"""
Phase 1 —— RTC 离线正确性验证（不碰 async pipeline）

这份脚本要做的事情，用大白话讲一遍：

    1. 从一条已有的 episode 里，挑两个时间点 t 和 t+k（比如相隔10步）。
    2. 假装自己是 policy_server：在 t 时刻，让模型正常预测一个动作 chunk，
       叫它 chunk_A。
    3. 假装机器人执行了 chunk_A 的前 k 步，剩下 chunk_A[k:] 就是"没执行完、
       留下来的动作"，也就是 RTC 需要的 prev_chunk_left_over。
    4. 在 t+k 时刻，用两种方式各预测一次新的 chunk：
         - 方式一：正常预测，不告诉模型"之前还剩什么动作"（RTC 关）
         - 方式二：把 chunk_A 剩下的部分喂给模型，让它做 RTC 引导去噪（RTC 开）
    5. 对比这两种方式的结果，检查 RTC 是否真的在按预期工作。

全程只调用 policy.predict_action_chunk()，不经过 gRPC、不经过
robot_client.py / policy_server.py，所以拿到的动作全程都在模型内部
的"归一化空间"里，不需要做反归一化的转换（这是我们上一轮讨论时确认过的坑）。

用法：
    python phase1_rtc_sanity_check.py \
        --checkpoint /path/to/your/checkpoint \
        --dataset-repo-id your_dataset_name_or_path \
        --episode-index 0 \
        --t 20 \
        --k 10
"""

import argparse
import copy
import tqdm
import torch
import torch.utils.data

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_dataset_viz import to_hwc_uint8_numpy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的 checkpoint 目录路径")
    parser.add_argument("--dataset-repo-id", type=str, required=True, help="用来抽取观测的数据集路径/repo_id")
    parser.add_argument("--episode-index", type=int, default=0, help="从哪条 episode 里抽取两帧")
    parser.add_argument("--t", type=int, default=20, help="第一个时间点，在该 episode 内的帧序号")
    parser.add_argument("--k", type=int, default=10, help="两个时间点之间相隔多少步")
    parser.add_argument("--device", type=str, default="cuda", help="推理用的设备")
    return parser.parse_args()


def build_single_frame_batch(dataset: LeRobotDataset, idx: int, device: str) -> dict:
    """从 LeRobotDataset 里取出一帧，拼成 predict_action_chunk 需要的 batch 格式。

    注：这里用的是 dataset[idx] 直接按下标取值，LeRobotDataset 本身实现了
    __getitem__，正常应该能直接随机访问。如果你的数据集因为视频解码后端
    的原因，直接按下标取值还是报错或者很慢，把这里换成用
    torch.utils.data.DataLoader(dataset, batch_size=1) 顺序遍历、
    取到第 idx 个 batch 即可，处理逻辑不用变，只是取数据的方式不同。

    dataset[idx] 拿到的是"单帧"（没有 batch 维度），而模型的预处理器
    (preprocessor) 期待输入是"一批"数据，哪怕这一批只有 1 条。所以这里
    要做的事情很简单：把每个 tensor 前面加一个 batch 维度（unsqueeze(0)），
    把 task 字符串包成一个长度为 1 的列表。
    """
    # for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
    #     for i in range(len(batch["index"])):
    #         for key in dataset.meta.camera_keys:
    #             img = to_hwc_uint8_numpy(batch[key][i])
    item = dataset[idx]

    batch = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
        elif key == "task":
            batch[key] = [value]  # 预处理器期待 task 是一个 list[str]
        # 其余字段（episode_index、timestamp 等元信息）预测时用不到，跳过即可

    return batch


def toggle_rtc(policy: SmolVLAPolicy, enabled: bool):
    """临时打开/关闭 RTC，用来做"检查一"。

    注意：这里只是把 config 里的 enabled 开关翻过来，
    不会重新创建 rtc_processor，所以翻回来的时候不用担心状态丢失。
    """
    policy.config.rtc_config.enabled = enabled


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """算两段动作序列有多"像"，返回一个数字，数字越小说明越接近。"""
    return torch.norm(a - b, dim=-1).mean().item()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 第 0 步：加载 policy，并且先确认 RTC 真的被正确加载了
    # ------------------------------------------------------------------
    print(f"正在加载 checkpoint: {args.checkpoint}")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
    policy.to(args.device)
    policy.eval()

    assert policy.config.rtc_config is not None, (
        "config.json 里没有读到 rtc_config，先检查一下 config.json 里的字段名"
        "是不是写对了（应该是 'enabled' 而不是 'enable'）"
    )
    assert policy.rtc_processor is not None, "rtc_processor 没有被初始化"
    print(f"RTC 配置确认: enabled={policy.config.rtc_config.enabled}, "
          f"execution_horizon={policy.config.rtc_config.execution_horizon}, "
          f"max_guidance_weight={policy.config.rtc_config.max_guidance_weight}")

    # ------------------------------------------------------------------
    # 第 1 步：准备预处理器（负责把原始观测归一化、tokenize 语言指令等）
    # 用同一个 checkpoint 目录加载，这样归一化统计量和训练时保持一致
    # ------------------------------------------------------------------
    preprocessor, _postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint,
    )
    # 注：这里我们不需要 _postprocessor（反归一化），因为我们全程留在
    # 模型内部的归一化空间里，不需要转换成机器人能执行的真实单位。

    # ------------------------------------------------------------------
    # 第 2 步：从数据集里挑两帧
    # ------------------------------------------------------------------
    # 关键点：构造 LeRobotDataset 时传 episodes=[episode_index]，这样
    # dataset 里就只包含这一条 episode 的帧，下标直接从 0 开始数到
    # (这条episode的长度-1)，不需要自己去算"这条episode在整个数据集里
    # 从哪个全局下标开始"——上一版脚本就是栽在这个自己算偏移量的逻辑上。
    # 这个用法参照了你项目里 lerobot_dataset_viz.py 里的写法。
    dataset = LeRobotDataset(args.dataset_repo_id, episodes=[args.episode_index])

    ep_length = len(dataset)
    assert args.t + args.k < ep_length, (
        f"t+k={args.t + args.k} 超出了这条 episode 的长度 {ep_length}，"
        f"换一个更小的 t/k，或者换一条更长的 episode"
    )

    batch_t = build_single_frame_batch(dataset, args.t, args.device)
    batch_t2 = build_single_frame_batch(dataset, args.t + args.k, args.device)

    batch_t = preprocessor(batch_t)
    batch_t2 = preprocessor(batch_t2)

    # ------------------------------------------------------------------
    # 第 3 步：在 t 时刻正常推理一次，得到 chunk_A
    # ------------------------------------------------------------------
    # 固定随机种子/噪声，这样每次重跑结果都可复现，方便你调试对比
    torch.manual_seed(0)
    chunk_size = policy.config.chunk_size
    action_dim = policy.config.max_action_dim
    noise_a = torch.randn(1, chunk_size, action_dim, device=args.device)

    with torch.no_grad():
        chunk_A = policy.predict_action_chunk(batch_t, noise=noise_a.clone())
    print(f"\nchunk_A 形状: {tuple(chunk_A.shape)}  (batch, chunk_size, action_dim)")

    # 假装机器人执行了前 k 步，剩下的就是 prev_chunk_left_over
    prev_chunk_left_over = chunk_A[:, args.k:, :].clone()
    print(f"prev_chunk_left_over 形状: {tuple(prev_chunk_left_over.shape)}  "
          f"(还剩 {chunk_size - args.k} 步没执行)")

    # ------------------------------------------------------------------
    # 第 4 步：在 t+k 时刻，分别用「RTC 关」和「RTC 开」各推理一次
    # ------------------------------------------------------------------
    noise_b = torch.randn(1, chunk_size, action_dim, device=args.device)

    # 4a. RTC 开，且真的把 leftover 传进去
    toggle_rtc(policy, enabled=True)
    with torch.no_grad():
        chunk_B_rtc = policy.predict_action_chunk(
            batch_t2,
            noise=noise_b.clone(),
            inference_delay=args.k,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=policy.config.rtc_config.execution_horizon,
        )

    # 4b. RTC 开，但不传 leftover（模拟"第一步没有历史可用"的情况）
    with torch.no_grad():
        chunk_B_rtc_no_leftover = policy.predict_action_chunk(
            batch_t2,
            noise=noise_b.clone(),
            inference_delay=None,
            prev_chunk_left_over=None,
            execution_horizon=None,
        )

    # 4c. RTC 彻底关掉（config 层面），作为最干净的对照组
    toggle_rtc(policy, enabled=False)
    with torch.no_grad():
        chunk_B_no_rtc = policy.predict_action_chunk(batch_t2, noise=noise_b.clone())
    toggle_rtc(policy, enabled=True)  # 用完记得翻回去，避免影响脚本后续逻辑

    # ------------------------------------------------------------------
    # 检查一：RTC 开着但不给 leftover，应该和"RTC 整个关掉"结果一致
    # ------------------------------------------------------------------
    # 为什么要查这个：modeling_rtc.py 里写的是，如果 prev_chunk_left_over
    # 是 None，就直接把原始的去噪结果原样返回，不做任何修正。所以这种情况
    # 下，"RTC开+没leftover" 和 "RTC关" 应该是完全一样的数字。如果这里对
    # 不上，说明某个地方的分支判断写错了，后面的检查就不用做了，先回去
    # 排查代码接线。
    same = torch.allclose(chunk_B_rtc_no_leftover, chunk_B_no_rtc, atol=1e-5)
    diff = (chunk_B_rtc_no_leftover - chunk_B_no_rtc).abs().max().item()
    print("\n[检查一] RTC开(无leftover) 是否等于 RTC关：", "通过 ✅" if same else "没通过 ❌")
    print(f"         两者最大逐元素差异: {diff:.8f}  (应该非常接近 0)")

    # ------------------------------------------------------------------
    # 检查二：RTC 开着且给了 leftover 时，新 chunk 的开头应该比"没有RTC"
    # 更贴近 prev_chunk_left_over
    # ------------------------------------------------------------------
    horizon = policy.config.rtc_config.execution_horizon
    horizon = min(horizon, prev_chunk_left_over.shape[1])  # 防止 leftover 比 horizon 短

    dist_with_rtc = l2_distance(
        chunk_B_rtc[:, :horizon, :], prev_chunk_left_over[:, :horizon, :]
    )
    dist_without_rtc = l2_distance(
        chunk_B_no_rtc[:, :horizon, :], prev_chunk_left_over[:, :horizon, :]
    )

    better = dist_with_rtc < dist_without_rtc
    print(f"\n[检查二] RTC 是否让新 chunk 更贴近上一段留下来的动作：", "通过 ✅" if better else "没通过 ❌")
    print(f"         开RTC时的距离:  {dist_with_rtc:.6f}")
    print(f"         关RTC时的距离:  {dist_without_rtc:.6f}")
    print(f"         (前者应该明显更小，说明 RTC guidance 真的在起作用)")

    # ------------------------------------------------------------------
    # 附加信息：把 debug tracker 记录的内部轨迹打印出来，方便你进一步排查
    # 只有 config 里 debug=True 才会有内容
    # ------------------------------------------------------------------
    debug_steps = policy.rtc_processor.get_all_debug_steps()
    if debug_steps:
        print(f"\n[附加信息] debug tracker 记录了 {len(debug_steps)} 个去噪 step。")
        first_step = debug_steps[0]
        mid_step = debug_steps[int(len(debug_steps)/2)]
        last_step = debug_steps[-1]
        print(f"         第一步 guidance_weight: {first_step.guidance_weight}")
        print(f"         第{int(len(debug_steps)/2)}步：{mid_step.guidance_weight}")
        print(f"         最后一步 guidance_weight: {last_step.guidance_weight}")
        print("         按设计，guidance_weight 应该两边大，中间小。")
    else:
        print("\n[附加信息] 没有采集到 debug 轨迹，如果需要看内部细节，"
              "确认 config.json 里 rtc_config.debug=true。")

    policy.rtc_processor.reset_tracker()


if __name__ == "__main__":
    main()