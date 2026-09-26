import torch
import torch.nn as nn

def compute_params(model):
    """统计模型参数量 (单位: M, 百万)"""
    return sum(p.numel() for p in model.parameters()) / 1e6

def compute_grad_norm(model, inputs):
    """
    经典传统 ZCP 指标：GradNorm (梯度范数)
    缺点:
    1. 必须执行反向传播 (backward), 计算开销大, 显存占用高
    2. 需要求偏导, 耗时远大于前向无梯度指标
    """
    model.eval()
    model.zero_grad()
    
    # 构造假目标以计算交叉熵损失
    outputs = model(inputs)
    # 取均值作为标量 loss 进行反向求导
    loss = outputs.sum()
    loss.backward()
    
    grad_norm_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm_sum += p.grad.data.norm(2).item()
            
    model.zero_grad()
    return grad_norm_sum
