import argparse
import json
import random
from pathlib import Path
import time
from lerobot.utils.random_utils import get_random_orientation, get_random_position
from lerobot.utils.utils import get_euclidean_distance

EP_CONF_PATH = Path(__file__).parent.parent / "robots/isaac_piper/config_episode_rule.json"

def generate_holdout_pose(seed, episodes):
    with open(EP_CONF_PATH, "r", encoding="utf-8") as f:
        ep_conf = json.load(f)

    obj_poses = { "object": {} }
    goal_poses = { "goal": {} }
    obj_file_path = f"/home/shenjie/Documents/obj_pose_{seed}_{episodes}_"
    goal_file_path = f"/home/shenjie/Documents/goal_pose_{seed}_{episodes}_"

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
            goal_pos_x, goal_pos_y = get_random_position(ep_conf["limits"]["goal"]["angle1"][0],
                                                            ep_conf["limits"]["goal"]["angle1"][1],
                                                            ep_conf["limits"]["goal"]["radius"][0],
                                                            ep_conf["limits"]["goal"]["radius"][1],
                                                            ep_conf["limits"]["origin"][0],
                                                            ep_conf["limits"]["origin"][1])
            
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
        goal_ori = get_random_orientation(ep_conf["limits"]["goal"]["orientation"][0],
                                          ep_conf["limits"]["goal"]["orientation"][1])

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
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    args = parser.parse_args()

    generate_holdout_pose(seed=args.seed, episodes=args.episodes)
    print("Poses saved!")