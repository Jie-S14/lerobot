import argparse
import json
import random
from pathlib import Path
import time
from lerobot.utils.random_utils import get_random_orientation, get_random_position
from lerobot.utils.utils import get_euclidean_distance

EP_CONF_PATH = Path(__file__).parent.parent / "robots/isaac_piper/config_episode_rule_OOD3.json"

def generate_holdout_pose(seed, episodes, is_record, dataset_name):
    with open(EP_CONF_PATH, "r", encoding="utf-8") as f:
        ep_conf = json.load(f)

    print(f"generate_holdout_pose: is_record={is_record}")

    obj_poses = { "object": {} }
    goal_poses = { "goal": {} }
    dir_path = Path("/home/shenjie/ws/data_viz/data")
    record_eval = "record" if is_record == "true" else "eval"
    dataset_name = dataset_name # "2cam_top_wst_goal_blu_1pos_15hz"
    obj_file_path = dir_path / f"obj_posori_{record_eval}_{dataset_name}_{seed}_{episodes}_" # obj_pos_ori_record/eval_2cam_top_wst_goal_blu_circle_pos.json
    goal_file_path = dir_path / f"goal_posori_{record_eval}_{dataset_name}_{seed}_{episodes}_"

    for ep in range(episodes):

        random.seed(ep+seed)
        distance_flag = False
        while not distance_flag:
            obj_pos_x, obj_pos_y = get_random_position(ep_conf["limits"]["object"]["angle"][0],
                                                        ep_conf["limits"]["object"]["angle"][1],
                                                    ep_conf["limits"]["object"]["radius"][0],
                                                    ep_conf["limits"]["object"]["radius"][1],
                                                    ep_conf["limits"]["origin"][0],
                                                    ep_conf["limits"]["origin"][1])
            # goal_pos_x, goal_pos_y = get_random_position(ep_conf["limits"]["goal"]["angle1"][0],
            #                                                 ep_conf["limits"]["goal"]["angle1"][1],
            #                                                 ep_conf["limits"]["goal"]["radius"][0],
            #                                                 ep_conf["limits"]["goal"]["radius"][1],
            #                                                 ep_conf["limits"]["origin"][0],
            #                                                 ep_conf["limits"]["origin"][1])
            goal_pos_x = ep_conf["limits"]["goal_fixed"]["x"]
            goal_pos_y = ep_conf["limits"]["goal_fixed"]["y"]

            distance = get_euclidean_distance(obj_pos_x, obj_pos_y,
                                          goal_pos_x, goal_pos_y)
            if distance < ep_conf["limits"]["min_distance"]:
                distance_flag = False
                print(f"Distance between object and goal: {distance}, less than the minimum distance. Re-generate.")
                continue
            else:
                distance_flag = True
                break

        obj_ori = get_random_orientation(ep_conf["limits"]["object"]["orientation"][0],
                                        ep_conf["limits"]["object"]["orientation"][1])
        # goal_ori = get_random_orientation(ep_conf["limits"]["goal"]["orientation"][0],
        #                                   ep_conf["limits"]["goal"]["orientation"][1])
        goal_ori = ep_conf["limits"]["goal_fixed"]["orientation"]

        obj_poses["object"][f"{ep}"] = {
            "position": [obj_pos_x, obj_pos_y, ep_conf["limits"]["object"]["z_axis"]],
            "orientation": obj_ori
        }
        goal_poses["goal"][f"{ep}"] = {
            "position": [goal_pos_x, goal_pos_y, ep_conf["limits"]["goal"]["z_axis"]],
            "orientation": goal_ori
        }

    datetime_str = time.strftime("%Y%m%d%H%M%S")
    json.dump(obj_poses, open(f"{obj_file_path}{datetime_str}.json", "w"), indent=4)
    json.dump(goal_poses, open(f"{goal_file_path}{datetime_str}.json", "w"), indent=4)

    print(f"Object poses saved to {obj_file_path}{datetime_str}.json")
    print(f"Goal poses saved to {goal_file_path}{datetime_str}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset/holdout poses for recording/evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--record", type=str, default="true", help="Whether the poses are for record or evaluate")
    parser.add_argument("--dataset_name", type=str, default="2cam_top_wst_goal_blu_1pos_15hz", help="The corresponding dataset in huggingface")
    args = parser.parse_args()

    generate_holdout_pose(seed=args.seed, 
                          episodes=args.episodes, 
                          is_record=args.record,
                          dataset_name=args.dataset_name)
    print("Poses saved!")