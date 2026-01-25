import zmq
import os
import numpy as np
import io
import cv2
import torch
import time
from PIL import Image
from pathlib import Path

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action


MODEL_ID = "lerobot/smolvla_base"
MODEL_PATH = "/home/shenjie/Documents/smolvla_result_20260124/checkpoints/last/pretrained_model"

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20
N_ACTION_STEPS = 20 # n_action_steps in the training params
STORE_IMAGES = False # whether store preprocessed images from socket to /logs
IMAGES_STORE_PATH = Path("logs")

INPUT_OBSERVATION_SOCKET = "tcp://localhost:5556"
OUTPUT_ACTION_SOCKET = "tcp://127.0.0.1:5555"

JOINT_ORDER = [
    'joint1','joint2','joint3','joint4','joint5','joint6','joint7','joint8'
]

# define feature structure of output action
# i.e. target joint states
ACTION_FEATURES = {
    'action': {
        'dtype': 'float32',
        'shape': (8,),
        'names': JOINT_ORDER,
    },
}

# define feature structure of intput observation
# i.e. images and current joint states
OBS_FEATURES = {
    'observation.state': {
        'dtype': 'float32',
        'shape': (8,),
        'names': JOINT_ORDER,
    },
    'observation.images.top': {
        'dtype': 'video',
        'shape': (480, 640, 3),
        'names': [
            'height', 
            'width', 
            'channels',
        ],
    },
    'observation.images.wrist': {
        'dtype': 'video',
        'shape': (480, 640, 3),
        'names': [
            'height', 
            'width', 
            'channels',
        ],
    },
}

DATASET_FEATURES = {**ACTION_FEATURES, **OBS_FEATURES}
print("Dataset features:", DATASET_FEATURES)

# radian range of piper joints in Isaac Sim
SIMULATION_RANGE = {
    'joint1': {'sim_min': -2.618, 'sim_max': 2.618},
    'joint2': {'sim_min': 0.0, 'sim_max': 3.14},
    'joint3': {'sim_min': -2.697, 'sim_max': 0.0},
    'joint4': {'sim_min': -1.832, 'sim_max': 1.832},
    'joint5': {'sim_min': -1.22, 'sim_max': 1.22},
    'joint6': {'sim_min': -3.14, 'sim_max': 3.14},
    'joint7': {'sim_min': 0, 'sim_max': 1.5},
    'joint8': {'sim_min': 0.0, 'sim_max': 0.04},
}

def rad2pos(rad: float, joint_name: str):
    """Convert radians into motor pos.
    
    Isaac Sim joint state has real radian values, while SO100 robot hardware
    use -100 to 100 as "pos". Original script for SO100 follower, i.e. using_smolvla_example.py,
    use the default value of RobotConfig.use_degrees = False, so it use MotorNormMode.RANGE_M100_100,
    check norm_mode_body of class so100_foller.SO100Follower
    """
    sim_min = SIMULATION_RANGE[joint_name]['sim_min']
    sim_max = SIMULATION_RANGE[joint_name]['sim_max']
    # pos = ((rad - sim_min) / (sim_max - sim_min)) * 200 - 100

    # For piper
    if rad < sim_min:
        rad = sim_min
    if rad > sim_max:
        rad = sim_max
    pos = np.rad2deg(rad)  # convert to degrees
    return pos

def pos2rad(pos: float, joint_name: str):
    """Convert the motor pos into real radian."""
    sim_min = SIMULATION_RANGE[joint_name]['sim_min']
    sim_max = SIMULATION_RANGE[joint_name]['sim_max']
    # rad = sim_min + (pos + 100) / 200 * (sim_max - sim_min)

    # For piper
    rad = np.deg2rad(pos)  # convert degrees to radian
    if rad < sim_min:
        rad = sim_min
    if rad > sim_max:
        rad = sim_max
    return rad


if __name__ == "__main__":

    obs_context = zmq.Context()
    obs_socket = obs_context.socket(zmq.SUB)
    obs_socket.connect(INPUT_OBSERVATION_SOCKET)
    obs_socket.setsockopt(zmq.SUBSCRIBE, b"")

    act_context = zmq.Context()
    act_socket = act_context.socket(zmq.PUB)
    act_socket.bind(OUTPUT_ACTION_SOCKET)
    time.sleep(0.5) # Give subscribers a short time to connect

    device = torch.device("cuda")

    model = SmolVLAPolicy.from_pretrained(MODEL_PATH) # MODEL_ID if want to retrieve from hub
    # model = model.to(device)
    # model.eval()
    print("Model device:", next(model.parameters()).device)

    os.makedirs(IMAGES_STORE_PATH, exist_ok=True)

    # test other configs
    # model.config.n_action_steps = 5

    camera_feature_keys = list(model.config.image_features)
    max_supported_cameras = len(camera_feature_keys)
    if max_supported_cameras == 0:
        raise ValueError(f"Policy {MODEL_ID} exposes no camera inputs.")

    preprocess, postprocess = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=MODEL_PATH,   # MODEL_ID if want to retrieve from hub
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    task = "Pick up the red cube and place it in the box"
    robot_type = "piper_isaac_sim420"

    print(f"Current task: {task}")

    count = 0
    while True:

        payload = obs_socket.recv()   # one npz blob
        buf = io.BytesIO(payload)
        data = np.load(buf)
        print("Received data keys:", list(data.keys()))

        ts = float(data["ts"][0])
        joints = data["joints"].astype(np.float32)

        # image 1: from topic /camera1_rgb
        img1_vec = data["img1"]
        encoded_flag = int(data.get("img1_encoded", np.array([1]))[0])
        # convert JPEG bytes to numpy image
        if encoded_flag == 1:
            buf1 = np.frombuffer(img1_vec.tobytes(), dtype=np.uint8)
            img1 = cv2.imdecode(buf1, cv2.IMREAD_COLOR)
        else:
            img1 = img1_vec.reshape((480, 640, 3))
        # convert img into RGB
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        # image 2: from topic /camera2_rgb
        img2_vec = data["img2"]
        encoded_flag = int(data.get("img2_encoded", np.array([1]))[0])
        # convert JPEG bytes to numpy image
        if encoded_flag == 1:
            buf2 = np.frombuffer(img2_vec.tobytes(), dtype=np.uint8)
            img2 = cv2.imdecode(buf2, cv2.IMREAD_COLOR)
        else:
            img2 = img2_vec.reshape((480, 640, 3))
        # convert img into RGB
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # The expected obs should have this structure:
        # {
        #   "shoulder_pan.pos": joint_value,
        #   "shoulder_lift.pos": joint_value,
        #   "elbow_flex.pos": joint_value,
        #   "wrist_flex.pos": joint_value,
        #   "wrist_roll.pos": joint_value,
        #   "gripper.pos": joint_value,
        #   "camera1": cv2.RGB image with shape h, w, c, no rotation
        #   "camera2": cv2.RGB image with shape h, w, c, no rotation
        # }

        obs = {
            "joint1": rad2pos(rad=joints[0], joint_name="joint1"),      # When collecting data in Isaac Sim, we should just keep the radian!!
            "joint2": rad2pos(rad=joints[1], joint_name="joint2"),
            "joint3": rad2pos(rad=joints[2], joint_name="joint3"),
            "joint4": rad2pos(rad=joints[3], joint_name="joint4"),
            "joint5": rad2pos(rad=joints[4], joint_name="joint5"),
            "joint6": rad2pos(rad=joints[5], joint_name="joint6"),
            "joint7": joints[5] * 100, # scale up the gripper open/close command
            "joint8": joints[6] * 100,
            "top": img1,
            "wrist": img2,
        }

        # TODO: check what viewpoint angle and distance to robot (i.e. pose) should the camera have

        # the built obs_frame would have such structure:
        # {
        #   "observation.state": torch.Tensor with shape [1, 6]
        #   "observation.images.camera1": torch.Tensor of image batch with shape [1, c, h, w]
        #   "task": task string same as input,
        #   "robot_type": robot_type string same as input,
        # }
        obs_frame = build_inference_frame(
            observation=obs, 
            ds_features=DATASET_FEATURES, 
            device=device, 
            task=task, 
            robot_type=robot_type,
        )

        obs = preprocess(obs_frame)

        # save images in obs
        if STORE_IMAGES and count % N_ACTION_STEPS == 0:
            # cam1
            cam1_img = obs['observation.images.camera1'].cpu().detach().numpy().squeeze().reshape(480, 640, 3)
            cam1_img_uint8 = (cam1_img * 255).astype(np.uint8)
            cam1_img_uint8_save = Image.fromarray(cam1_img_uint8)  # expects RGB order
            cam1_img_uint8_save.save(IMAGES_STORE_PATH / f"obs_cam1_{count}.png")
            # cam2
            cam2_img = obs['observation.images.camera2'].cpu().detach().numpy().squeeze().reshape(480, 640, 3)
            cam2_img_uint8 = (cam2_img * 255).astype(np.uint8)
            cam2_img_uint8_save = Image.fromarray(cam2_img_uint8)  # expects RGB order
            cam2_img_uint8_save.save(IMAGES_STORE_PATH / f"obs_cam2_{count}.png")

        action = model.select_action(obs)
        action = postprocess(action)

        # the returned action would have such structure:
        # {
        #     'shoulder_pan.pos': abs_target_value,
        #     'shoulder_lift.pos': abs_target_value,
        #     'elbow_flex.pos': abs_target_value,
        #     'wrist_flex.pos': abs_target_value, 
        #     'wrist_roll.pos': abs_target_value, 
        #     'gripper.pos': abs_target_value, 
        # }
        action = make_robot_action(action, DATASET_FEATURES)

        action_array = np.array([
            pos2rad(pos=action["joint1"], joint_name="joint1"),
            pos2rad(pos=action["joint2"], joint_name="joint2"),
            pos2rad(pos=action["joint3"], joint_name="joint3"),
            pos2rad(pos=action["joint4"], joint_name="joint4"),
            pos2rad(pos=action["joint5"], joint_name="joint5"),
            pos2rad(pos=action["joint6"], joint_name="joint6"),
            action["joint7"] / 100.0,   # scale down the gripper
            action["joint8"] / 100.0,
        ],dtype=np.float32)

        if count % N_ACTION_STEPS == 0:
            print_action = ', '.join(f"{value:2f}" for value in action_array.tolist())
            print(f"Returned actions: [{print_action}]")

        # send actions to zmq socket
        act_socket.send(action_array.tobytes())

        count += 1

