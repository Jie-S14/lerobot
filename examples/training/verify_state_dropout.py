"""
验证 state_dropout_prob 的 attention-mask 机制是否正确生效。

用法:
    logging.basicConfig(level=logging.DEBUG)  # 打开(a)里加的permanent log
    python verify_state_dropout.py

依赖: 你的 lerobot 环境里已装好 smolvla 相关代码（含本次修改）。
"""

import logging

import torch

from lerobot.configs import parser
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("verify_state_dropout")


def build_dummy_batch(policy: SmolVLAPolicy, bsize: int = 8, device: str = "cpu"):
    """构造一个满足 embed_prefix 输入形状要求的最小 dummy batch。"""
    cfg = policy.config
    img_size = cfg.resize_imgs_with_padding  # (H, W)

    # 假设单相机；如果你的 config 里 image key 数量 > 1，按需扩展成多个 tensor
    n_cams = 1
    images = [torch.rand(bsize, 3, img_size[0], img_size[1], device=device) for _ in range(n_cams)]
    img_masks = [torch.ones(bsize, dtype=torch.bool, device=device) for _ in range(n_cams)]

    seq_len = 12
    lang_tokens = torch.randint(0, 1000, (bsize, seq_len), device=device)
    # 模拟变长句子：每个样本随机长度、右侧pad
    lang_masks = torch.zeros(bsize, seq_len, dtype=torch.bool, device=device)
    for i in range(bsize):
        real_len = torch.randint(3, seq_len + 1, (1,)).item()
        lang_masks[i, :real_len] = True

    state_dim = cfg.max_state_dim
    state = torch.randn(bsize, state_dim, device=device)

    return images, img_masks, lang_tokens, lang_masks, state


def check_mask_values(policy: SmolVLAPolicy, batch, drop_prob: float = 0.5):
    """弱验证：直接检查 embed_prefix 返回的 pad_masks 数值。"""
    images, img_masks, lang_tokens, lang_masks, state = batch
    model = policy.model
    model.train()
    model.config.state_dropout_prob = drop_prob

    torch.manual_seed(0)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )

    # state token 是 prefix 里最后一个 token（紧跟 lang 之后），列 index 固定：
    state_col = prefix_pad_masks.shape[1] - 1  # states_seq_len == 1 时成立
    state_valid = prefix_pad_masks[:, state_col]  # (B,) True=保留 False=被丢弃

    logger.info("state_col index = %d", state_col)
    logger.info("state_valid per sample = %s", state_valid.tolist())
    dropped_ratio = (~state_valid).float().mean().item()
    logger.info("observed dropped ratio = %.3f (target %.3f)", dropped_ratio, drop_prob)

    # 严格检查：att_2d_masks 里，被丢弃样本的 state 那一行/列必须全 False（除非全pad场景另计）
    att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    for b in range(state_valid.shape[0]):
        if not state_valid[b]:
            row_all_false = not att_2d[b, state_col, :].any()
            col_all_false = not att_2d[b, :, state_col].any()
            assert row_all_false, f"sample {b}: state token 仍能 attend 到别的 token！"
            assert col_all_false, f"sample {b}: state token 仍被别的 token attend！"
    logger.info("✅ mask-value check passed: 所有被丢弃样本的 state token 在 attention 图中完全隔离")


def check_output_invariance(policy: SmolVLAPolicy, batch):
    """
    强验证（端到端）：state_dropout_prob=1.0 时，
    用两份完全不同的 state 值跑同一个 forward，输出应逐位相同（在数值精度内）。
    这是唯一能排除"某处代码悄悄还是读取了 state 值"的验证方式。
    """
    images, img_masks, lang_tokens, lang_masks, state_a = batch
    model = policy.model
    model.train()
    model.config.state_dropout_prob = 1.0  # 强制全丢，去掉随机性干扰

    state_b = torch.randn_like(state_a) * 100.0 + 50.0  # 故意用数值上完全不同的 state

    bsize = state_a.shape[0]
    actions = torch.randn(bsize, model.config.chunk_size, model.config.max_action_dim)
    torch.manual_seed(42)
    noise = model.sample_noise(actions.shape, actions.device)
    time = model.sample_time(actions.shape[0], actions.device)

    with torch.no_grad():
        losses_a = model.forward(
            images, img_masks, lang_tokens, lang_masks, state_a, actions, noise=noise, time=time
        )
        losses_b = model.forward(
            images, img_masks, lang_tokens, lang_masks, state_b, actions, noise=noise, time=time
        )

    max_diff = (losses_a - losses_b).abs().max().item()
    logger.info("max |losses_a - losses_b| with state fully dropped = %.3e", max_diff)
    assert max_diff < 1e-5, (
        f"state_dropout_prob=1.0 时，两份不同 state 的输出竟然不同 (max_diff={max_diff})！"
        f"说明 state 信息还是从某条路径泄漏进了模型。"
    )
    logger.info("✅ output-invariance check passed: state 值完全不影响输出，mask 机制严格生效")

@parser.wrap()
def main(cfg: SmolVLAConfig):
    policy = SmolVLAPolicy(cfg)

    batch = build_dummy_batch(policy, bsize=8)
    check_mask_values(policy, batch, drop_prob=0.5)
    check_output_invariance(policy, batch)


if __name__ == "__main__":
    # cfg = SmolVLAConfig()  # 按需替换成你实际用的 config（如需要加载 pretrained 权重路径等）
    main()
