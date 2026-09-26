import torch
import torch.nn as nn

def compute_apd_proxy(model, inputs):
    r"""
    NeurIPS 2025 论文 (Appendix Figure 16) 中公开的 APD 零成本代理指标基线:
    s_APD = ( \sum_{l \in B} ||M_l||_F^2 / ||M_l||_2^2 ) * ( \sum_{l \in C} ||W_l||_1 / ||W_l||_2 )
    
    特点:
    1. 零梯度开销 (全程在 torch.no_grad() 下运行)
    2. 左半部分: 衡量所有 BatchNorm 层的特征稳定秩 (Stable Rank，表征特征丰富度与抗坍塌能力)
    3. 右半部分: 衡量所有卷积层权重的 L1/L2 几何多样性
    """
    bn_ranks = []
    ratios = []
    hooks = []

    # 1. 注册 Forward Hook，抓取所有 BatchNorm2d 层的特征图输出
    def bn_hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            B, C, H, W = out.shape
            # 展平矩阵为 (B*H*W, C)
            mat = out.view(B, C, -1).permute(0, 2, 1).reshape(-1, C)
            # 计算 Frobenius 范数的平方 (总能量)
            frob_norm = torch.linalg.matrix_norm(mat, ord='fro') ** 2
            # 计算 谱范数 (最大奇异值) 的平方 (最强单项能量)
            spec_norm = torch.linalg.matrix_norm(mat, ord=2) ** 2
            # 稳定秩 = 总能量 / 最强单项能量
            stable_rank = frob_norm / (spec_norm + 1e-6)
            bn_ranks.append(stable_rank.mean())

    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            hooks.append(layer.register_forward_hook(bn_hook))
        elif isinstance(layer, nn.Conv2d):
            weights = layer.weight
            # 计算卷积核的 L1 范数与 L2 范数之比 (衡量参数多样性)
            l1_norm = weights.abs().sum(dim=(1, 2, 3)).mean()
            l2_norm = weights.norm(p=2, dim=(1, 2, 3)).mean()
            ratios.append((l1_norm / (l2_norm + 1e-6)).item())

    # 2. 核心特征：纯前向推断，无反向传播！
    model.eval()
    with torch.no_grad():
        model(inputs)

    # 3. 移除 Hook 释放内存
    for hook in hooks:
        hook.remove()

    # 4. 计算综合打分 (两项之和相乘)
    bn_sum = torch.stack(bn_ranks).sum().item() if bn_ranks else 0.0
    ratio_sum = sum(ratios) if ratios else 0.0
    score = bn_sum * ratio_sum

    return score
