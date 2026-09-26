import time
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

from apd_proxy import compute_apd_proxy
from baselines import compute_params, compute_grad_norm

# 设置中文字体与学术绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def run_benchmark():
    print("=" * 70)
    print("      NeurIPS 2025 APD 论文核心指标 (s_APD) 本地实验复现")
    print("=" * 70)
    
    # 1. 挑选具有代表性的、未经训练的经典视觉骨干网络
    model_factories = {
        "ShuffleNet-V2": models.shufflenet_v2_x1_0,
        "MobileNet-V2": models.mobilenet_v2,
        "ResNet-18": models.resnet18,
        "ResNet-34": models.resnet34,
        "ResNet-50": models.resnet50
    }
    
    # 2. 模拟论文中的标准测试配置 (Appendix B.1: CIFAR-10, Batch Size = 16, 32x32 分辨率)
    torch.manual_seed(42)
    dummy_input = torch.randn(16, 3, 32, 32)
    
    results = []
    
    for name, factory in model_factories.items():
        print(f"\n[正在体检] 实例化未训练模型: {name} ...")
        model = factory(weights=None)
        params = compute_params(model)
        
        # --- 测试 APD 指标 (纯前向无梯度) ---
        # 预热 1 次
        _ = compute_apd_proxy(model, dummy_input)
        
        # 测 5 次取平均耗时
        start_t = time.perf_counter()
        for _ in range(5):
            apd_score = compute_apd_proxy(model, dummy_input)
        apd_time = (time.perf_counter() - start_t) / 5.0 * 1000.0  # 转毫秒
        
        # --- 测试 传统 GradNorm 指标 (需反向求导) ---
        # 预热 1 次
        _ = compute_grad_norm(model, dummy_input)
        
        # 测 5 次取平均耗时
        start_t = time.perf_counter()
        for _ in range(5):
            grad_score = compute_grad_norm(model, dummy_input)
        grad_time = (time.perf_counter() - start_t) / 5.0 * 1000.0  # 转毫秒
        
        speedup = grad_time / (apd_time + 1e-6)
        
        results.append({
            "name": name,
            "params": params,
            "apd_score": apd_score,
            "apd_time": apd_time,
            "grad_score": grad_score,
            "grad_time": grad_time,
            "speedup": speedup
        })
        print(f"  -> 参数量: {params:.2f}M | APD得分: {apd_score:.1f} (耗时: {apd_time:.1f}ms) | GradNorm耗时: {grad_time:.1f}ms (提速 {speedup:.2f}x)")

    # 3. 打印完整科研对照表格
    print("\n" + "=" * 80)
    print(f"{'模型名称':<15} | {'参数量(M)':<10} | {'s_APD 得分':<12} | {'APD耗时(ms)':<12} | {'GradNorm耗时':<12} | {'提速倍数'}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<15} | {r['params']:<10.2f} | {r['apd_score']:<12.1f} | {r['apd_time']:<12.1f} | {r['grad_time']:<12.1f} | {r['speedup']:.2f}x")
    print("=" * 80)

    # 4. 绘制顶会学术级对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    width = 0.35
    
    # 图 1: 评估耗时对比 (证明 APD 的毫秒级与无梯度极速)
    ax1.bar(x - width/2, [r['apd_time'] for r in results], width, label='APD (论文指标, 纯前向)', color='#2ca02c', alpha=0.85)
    ax1.bar(x + width/2, [r['grad_time'] for r in results], width, label='GradNorm (传统指标, 需反向求导)', color='#d62728', alpha=0.85)
    ax1.set_ylabel('单次评估耗时 (毫秒 ms) - 越低越好', fontsize=12)
    ax1.set_title('核心优势一：零梯度计算带来的极致速度 (CIFAR-10 设定)', fontsize=13, pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 图 2: 模型深度/容量 vs APD 得分增长趋势 (证明特征丰富度单调感知)
    apd_scores = [r['apd_score'] for r in results]
    ax2.plot(x, apd_scores, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='s_APD 体检打分')
    for i, txt in enumerate(apd_scores):
        ax2.annotate(f"{txt:.1f}", (x[i], apd_scores[i]), textcoords="offset points", xytext=(0,10), ha='center')
    ax2.set_ylabel('APD 综合打分 (s_APD) - 衡量特征表达空间', fontsize=12)
    ax2.set_title('核心优势二：免训练下对模型拓扑容量的精准感知', fontsize=13, pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    output_fig = "apd_verification_results.png"
    plt.savefig(output_fig)
    print(f"\n[成果交付] 高清学术成果图已保存至: {output_fig}")

if __name__ == '__main__':
    run_benchmark()
