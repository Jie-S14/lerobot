import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import List, Optional

import grpc

from lerobot.configs import parser
from lerobot.async_inference.configs import PolicyServerConfig, RobotClientConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2_grpc
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.control_utils import init_eval_keyboard_listener
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.random_utils import set_seed
from lerobot.utils.visualization_utils import init_rerun, stop_rerun

@dataclass
class EvalAsyncConfig:
    client: RobotClientConfig = field(default_factory=RobotClientConfig)
    server: PolicyServerConfig = field(default_factory=PolicyServerConfig)
    start_episode: int = 0;
    end_episode: Optional[int] = None;
    episode_time_s: float = 45.0
    seed: Optional[int] = None
    results_folder: str = field(default=".", metadata={"help": "Folder to save the inference results"})

@parser.wrap()
def eval_robot_client(
    cfg: EvalAsyncConfig,
) -> List[dict]:
    """
    Synchronous eval which starts a PolicyServer in-process and runs RobotClient-driven episodes.
    - cfg: EvalAsyncConfig (top-level CLI entry containing client and server configs)
    """
    register_third_party_plugins()
    init_logging()
    logging.info("Starting eval_robot_client")
    logging.info("Client cfg:\n%s", pformat(asdict(cfg.client)))
    logging.info("Server cfg:\n%s", pformat(asdict(cfg.server)))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # initialize eval keyboard listener (returns listener_obj, stop_event, events)
    listener, stdin_stop_evt, keyboard_events = init_eval_keyboard_listener()

    # Use server config from top-level cfg.server
    server_cfg = cfg.server
    # Create PolicyServer instance and start gRPC server (we keep server reference to stop it later)
    policy_server = PolicyServer(server_cfg)
    # Start rerun.io
    init_rerun("camera_demo")
    # create gRPC server with ThreadPoolExecutor like policy_server.serve does
    from concurrent import futures
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, grpc_server)
    bind_addr = f"{server_cfg.host}:{server_cfg.port}"
    grpc_server.add_insecure_port(bind_addr)
    logging.info(f"Starting PolicyServer on {bind_addr}")
    grpc_server.start()
    # run wait_for_termination in background thread so main thread continues
    server_wait_thread = threading.Thread(target=grpc_server.wait_for_termination, daemon=True)
    server_wait_thread.start()

    # Small wait to allow server to bind before client tries to connect
    time.sleep(0.3)

    # Create RobotClient using cfg.client (this will instantiate and connect robot)
    client = RobotClient(cfg.client)

    # Start action receiver thread (non-blocking)
    action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver_thread.start()

    # Start handshake with server
    if not client.start():
        # stop server and exit
        try:
            grpc_server.stop(0)
        except Exception:
            pass
        raise RuntimeError("Failed to start RobotClient (could not connect to policy server)")

    # release the start_barrier so receive_actions can run ----
    client.start_barrier.wait()

    # Determine number of episodes from robot JSONs (obj/goal poses) if available
    robot = client.robot
    try:
        n_obj = len(getattr(robot, "_obj_poses", []))
        n_goal = len(getattr(robot, "_goal_poses", []))
        default_episodes = min(n_obj if n_obj > 0 else float("inf"), n_goal if n_goal > 0 else float("inf"))
        if default_episodes == float("inf"):
            default_episodes = 1
    except Exception:
        default_episodes = 1

    total_episodes = int(cfg.end_episode) if (cfg.end_episode is not None) else int(default_episodes)
    logging.info(f"Will run {total_episodes} episodes (default from jsons: {default_episodes})")

    # timing parameters (use client fps if available)
    env_dt = 1.0 / float(getattr(cfg.client, "fps", 30))
    max_steps = int(cfg.episode_time_s / env_dt) if cfg.episode_time_s > 0 else 100000

    results = []
    try:
        for ep in range(int(cfg.start_episode), total_episodes):
            logging.info(f"=== Episode {ep}/{total_episodes - 1} ===")
            # Reset environment via robot.reset_env; accept both (ep, seed) and (ep,)
            try:
                robot.reset_env(ep=ep, seed=cfg.seed)
            except TypeError:
                robot.reset_env(ep=ep)
            except Exception as e:
                logging.warning(f"robot.reset_env failed: {e}")

            step = 0
            done = False

            # Clear action queue at start
            with client.action_queue_lock:
                while not client.action_queue.empty():
                    try:
                        client.action_queue.get_nowait()
                    except Exception:
                        break

            logging.info("Start episode loop (press Ctrl-C to abort whole eval)")
            # control loop: advance sim / consume actions / send observations
            while (not done) and (step < max_steps):
                if keyboard_events.get("succ"):
                    is_success = True
                    logging.info(f"User marked episode {ep} SUCCESS at step {step}")
                    results.append({"episode": ep, "steps": step, "success": is_success})
                    break
                if keyboard_events.get("fail"):
                    is_success = False
                    logging.info(f"User marked episode {ep} FAIL at step {step}")
                    results.append({"episode": ep, "steps": step, "success": is_success})
                    break

                loop_t0 = time.perf_counter()

                # Advance simulation timestep if available
                try:
                    if hasattr(robot, "world") and robot.world is not None:
                        # try to use world.step(render=...) if signature supports it
                        try:
                            robot.world.step(render=True)
                        except TypeError:
                            robot.world.step()
                except Exception as e:
                    logging.debug(f"world.step() warning: {e}")

                # If there are queued actions, perform one
                if client.actions_available():
                    try:
                        client.control_loop_action(verbose=False)
                    except Exception as e:
                        logging.warning(f"control_loop_action error: {e}")

                # Send observation to server if client ready
                try:
                    if client._ready_to_send_observation():
                        # control_loop_observation will add 'task' to raw_observation
                        _ = client.control_loop_observation(task=cfg.client.task, verbose=False)
                except Exception as e:
                    logging.warning(f"control_loop_observation error: {e}")


                step += 1
                # maintain fps
                dt = time.perf_counter() - loop_t0
                time.sleep(max(0.0, env_dt - dt))

            if not keyboard_events.get("succ") and not keyboard_events.get("fail"):
                is_success = False
                logging.info(f"Time out. Episode {ep} FAIL at step {step}")
                results.append({"episode": ep, "steps": step, "success": is_success})

            # small pause before next episode
            log_say("Resetting environment for next episode", blocking=False)
            keyboard_events["succ"] = False
            keyboard_events["fail"] = False
            time.sleep(0.5)

        keyboard_events.clear()
        logging.info("Eval complete")

    except KeyboardInterrupt:
        logging.info("Evaluation interrupted by user")

    finally:
        logging.info("Shutting down client and server")

        # 2) Wait for client-side threads to finish (action receiver)
        try:
            action_receiver_thread.join(timeout=0.5)
            logging.info("Shutting down action_receiver_thread")
        except Exception as e:
            logging.warning(f"Error joining action receiver thread, {e}")


        # 1) signal client threads to stop (do not close channel yet)
        try:
            if getattr(client, "shutdown_event", None) is not None:
                client.shutdown_event.set()
                logging.info("Shutting down client.shutdown_event.set()")
            # call client.stop() to run its disconnect logic (note: stop() no longer closes channel)
            try:
                stop_rerun()
                logging.info("Shutting down rerun")
                client.stop()
                logging.info("Shutting down robot and client")
            except Exception as e:
                logging.warning(f"Error while stopping client: {e}")
        except Exception as e:
            logging.warning(f"Error stopping client, {e}")

        # 3) Now that client threads exited, close the gRPC channel (this avoids aborting in-flight RPCs)
        try:
            if hasattr(client, "close_channel"):
                client.close_channel()
                logging.info("Shutting down client.close_channel()")
            else:
                # fallback: if no close_channel available, try to close directly
                try:
                    client.channel.close()
                    logging.info("Shutting down client.channel.close()")
                except Exception:
                    pass
        except Exception as e:
            logging.warning(f"Error closing client channel: {e}")

        # 4) Stop gRPC server with a short grace period so in-flight streams can finish cleanly
        try:
            stop_rerun()  # stop rerun before server to allow any pending logs to flush
            grpc_server.stop(2)  # give 2s grace for existing RPCs to complete
            logging.info("Shutting down grpc_server")
        except Exception as e:
            logging.warning(f"Error stopping gRPC server, {e}")

        # 5) Join server wait thread
        try:
            server_wait_thread.join(timeout=1.0)
            logging.info("Shutting down server_wait_thread")
        except Exception as e:
            logging.warning(f"Error joining server wait thread, {e}")

        successes = sum(1 for r in results if r.get("success"))
        total = len(results) or 1
        succ_ratio = successes / total
        logging.info(f"Success count: {successes}/{total} ({100.0 * succ_ratio:.2f}%)")

        datetime_str = time.strftime("%Y%m%d%H%M%S")
        m = re.search(r'smolvla_result_(.+)/checkpoints/(\d+)', cfg.client.pretrained_name_or_path)

        if m:
            policy = m.group(1)        # 20260624
            ck = m.group(2)  # 020000
        else:
            policy = "00000000"
            ck = "000000"

        checkpoint = ck[1:3] if ck else "00"
        result_filepath = f"{cfg.results_folder}/eval_results_{policy}_{cfg.client.actions_per_chunk}_{cfg.client.chunk_size_threshold}_{checkpoint}k_{cfg.start_episode}-{total_episodes}ep_{100.0 * succ_ratio:.2f}%_{datetime_str}.json"
        json.dump(results, open(result_filepath, "w"), indent=4)
        print(f"Results saved at {result_filepath}")

    return results

def main():
    eval_robot_client()


if __name__ == "__main__":
    main()