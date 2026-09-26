# 由真实大模型在线自主强化学习闭环进化生成的 ZCP 冠军代码
# 进化代数: Generation 19
# 预测相关度 Spearman rho: 0.8902
# 评测时间: 2026-09-26 13:31:37

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def evaluate(model, inputs):
    import math
    import torch
    import torch.nn as nn

    was_training = model.training
    model.eval()

    activation_codes = []
    spatial_cvs = []
    channel_orths = []
    hooks = []

    def hook(module, inp, out):
        if not isinstance(out, torch.Tensor) or out.dim() != 4:
            return
        B, C, H, W = out.shape
        if B < 2 or C * H * W < 1:
            return
        # Float cast BEFORE binarize to avoid Bool dtype bug
        x = out.detach().float()

        # --- Primary signal source: binary activation signature (sample-level) ---
        A = (x > 0).to(torch.float32).reshape(B, -1)
        if A.shape[1] == 0:
            return
        activation_codes.append(A)

        if H > 1 and W > 1:
            # --- Spatial CV (validated, +0.10 rho boost) ---
            mu = x.mean(dim=(2, 3))
            sd = x.std(dim=(2, 3), unbiased=False)
            cv = sd / (mu.abs() + 1e-6)
            layer_cv = float(cv.median().item())
            if math.isfinite(layer_cv) and layer_cv >= 0.0:
                spatial_cvs.append(layer_cv)

        # --- NEW third dimension: Channel Orthogonality (COS) ---
        # Measures channel-redundancy: are different channels responding
        # to distinct input patterns (high orth) or collapsing onto the
        # same feature (low orth)? Purely (C, C)-space, orthogonal to both
        # sample-level (B, B) and spatial (H, W) signals.
        if C >= 2 and B * H * W >= 2:
            xf = x.permute(1, 0, 2, 3).reshape(C, -1)  # (C, B*H*W)
            xf = xf - xf.mean(dim=1, keepdim=True)
            n = xf.shape[1]
            if n >= 2:
                xn = xf / (xf.norm(dim=1, keepdim=True) + 1e-6)
                R = torch.mm(xn, xn.t())  # (C, C)
                if R.shape[0] >= 2:
                    eye = torch.eye(C, device=R.device, dtype=torch.bool)
                    off = R[~eye]
                    r_bar = float(off.abs().mean().item())
                    orth = 1.0 - min(max(r_bar, 0.0), 1.0)
                    if math.isfinite(orth):
                        channel_orths.append(orth)

    # Primary hook: every ReLU
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            hooks.append(m.register_forward_hook(hook))

    # Fallback 1: other non-linearities
    if not hooks:
        for m in model.modules():
            if isinstance(m, (nn.LeakyReLU, nn.SiLU, nn.GELU, nn.ReLU6)):
                hooks.append(m.register_forward_hook(hook))

    # Fallback 2: BatchNorm outputs (keeps evaluator well-defined)
    if not hooks:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                hooks.append(m.register_forward_hook(hook))

    try:
        with torch.no_grad():
            _ = model(inputs)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()
        else:
            model.eval()

    if not activation_codes:
        return 0.0

    # --- Primary: sample-kernel stable rank (validated rho=0.7669) ---
    A_all = torch.cat(activation_codes, dim=1).float()
    B_total, N_total = A_all.shape
    if B_total < 2 or N_total == 0:
        return 0.0

    norms = A_all.norm(dim=1, keepdim=True).clamp_min(1e-6)
    C_n = A_all / norms
    K = torch.mm(C_n, C_n.t())
    K = 0.5 * (K + K.t())
    try:
        s = torch.linalg.svdvals(K).clamp_min(0.0)
        sr_norm = float((s.sum() / (s[0] + 1e-6)) / B_total)
    except Exception:
        return 0.0

    # Global balance factor: kills dead (p->0) / saturated (p->1) features
    pos = float(A_all.mean().item())
    balance = 4.0 * pos * (1.0 - pos)

    # --- Secondary A: averaged spatial CV -> sqrt-mapped to (0,1) ---
    if spatial_cvs:
        cv_avg = sum(spatial_cvs) / len(spatial_cvs)
        cv_mapped = cv_avg / (1.0 + cv_avg)
        sp_cv = math.sqrt(max(cv_mapped, 0.0))
    else:
        sp_cv = 0.0

    # --- Secondary B: averaged channel orthogonality in [0, 1] ---
    if channel_orths:
        sp_orth = sum(channel_orths) / len(channel_orths)
        sp_orth = max(sp_orth, 0.0)
    else:
        sp_orth = 0.0

    # --- Conservative combination: validated primary (negative) + CV (0.2)
    #     + tiny channel-orthogonality refinement (0.05). ---
    score = -float(sr_norm * balance) + 0.2 * sp_cv + 0.05 * sp_orth

    if not math.isfinite(score):
        return 0.0
    return float(score)
