# 由 APD 自主强化学习多代搜索闭环进化出的最新最优 ZCP 代码
# 进化代数: Generation 3
# Spearman 相关度: 0.9650
# 思考链: 杂交重组：稳定秩虽然能完美识别通道坍塌，但对卷积核本身的表达潜力缺乏约束。将 Gen 2 的'BN稳定秩'与卷积核'L1/L2几何多样性'进行对偶杂交，形成综合打分公式。

import torch
import torch.nn as nn
import numpy as np

def evaluate_proxy(model, inputs):
    stable_ranks = []
    ratios = []
    hooks = []
    def hook_fn(module, inp, out):
        if isinstance(out, torch.Tensor):
            B, C, H, W = out.shape
            mat = out.view(B, C, -1).permute(0, 2, 1).reshape(-1, C)
            frob = torch.linalg.matrix_norm(mat, ord='fro') ** 2
            spec = torch.linalg.matrix_norm(mat, ord=2) ** 2
            stable_ranks.append((frob / (spec + 1e-6)).mean().item())
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            hooks.append(layer.register_forward_hook(hook_fn))
        elif isinstance(layer, nn.Conv2d):
            w = layer.weight
            l1 = w.abs().sum(dim=(1,2,3)).mean()
            l2 = w.norm(p=2, dim=(1,2,3)).mean()
            ratios.append((l1 / (l2 + 1e-6)).item())
    with torch.no_grad():
        model(inputs)
    for h in hooks: h.remove()
    return sum(stable_ranks) * sum(ratios)
