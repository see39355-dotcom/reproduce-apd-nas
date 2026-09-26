"""
大模型进化 ZCP 冠军代码泛化大考评测脚本
1. 在全部官方真实测试集 (Ground-Truth 涵盖 10.0% ~ 94.37%) 上进行盲测；
2. 在从 15,625 空间纯随机抽样的 50 个未知新架构上进行大规模机海扫描；
3. 输出完整相关度指标 (Spearman rho, Kendall tau, Pearson r) 与高清学术全景图。
"""
import os
import sys
import time
import json
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pearsonr

# 确保 Windows 终端 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 设置中文字体与学术绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入大模型自主生成的冠军代理评估函数
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from discovered_zcp_by_llm import evaluate as champion_evaluate

# =====================================================================
# 1. NAS-Bench-201 标准网络骨干定义
# =====================================================================
OP_NAMES = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

class NAS201Cell(nn.Module):
    def __init__(self, op_indices, C_in, C_out):
        super().__init__()
        self.op_indices = op_indices
        self.ops = nn.ModuleList()
        for idx in op_indices:
            name = OP_NAMES[idx]
            if name == 'none':
                self.ops.append(nn.Identity())
            elif name == 'skip_connect':
                self.ops.append(nn.Identity())
            elif name == 'nor_conv_1x1':
                self.ops.append(nn.Sequential(
                    nn.Conv2d(C_in, C_out, 1, bias=False),
                    nn.BatchNorm2d(C_out),
                    nn.ReLU(inplace=True)
                ))
            elif name == 'nor_conv_3x3':
                self.ops.append(nn.Sequential(
                    nn.Conv2d(C_in, C_out, 3, padding=1, bias=False),
                    nn.BatchNorm2d(C_out),
                    nn.ReLU(inplace=True)
                ))
            elif name == 'avg_pool_3x3':
                self.ops.append(nn.Sequential(
                    nn.AvgPool2d(3, stride=1, padding=1),
                    nn.BatchNorm2d(C_out)
                ))

    def forward(self, x):
        node0 = x
        node1 = self.ops[0](node0) if self.op_indices[0] != 0 else 0
        n2_0 = self.ops[1](node0) if self.op_indices[1] != 0 else 0
        n2_1 = self.ops[2](node1) if self.op_indices[2] != 0 and isinstance(node1, torch.Tensor) else 0
        node2 = n2_0 + n2_1
        n3_0 = self.ops[3](node0) if self.op_indices[3] != 0 else 0
        n3_1 = self.ops[4](node1) if self.op_indices[4] != 0 and isinstance(node1, torch.Tensor) else 0
        n3_2 = self.ops[5](node2) if self.op_indices[5] != 0 and isinstance(node2, torch.Tensor) else 0
        node3 = n3_0 + n3_1 + n3_2
        out = node3
        if not isinstance(out, torch.Tensor):
            out = torch.zeros_like(x)
        return out

class ResNetBasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c)
        )
    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class TinyNetwork201(nn.Module):
    def __init__(self, blueprint, num_classes=10):
        super().__init__()
        self.blueprint = blueprint
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.stage1 = nn.ModuleList([NAS201Cell(blueprint, 16, 16) for _ in range(5)])
        self.down1 = ResNetBasicBlock(16, 32, stride=2)
        self.stage2 = nn.ModuleList([NAS201Cell(blueprint, 32, 32) for _ in range(5)])
        self.down2 = ResNetBasicBlock(32, 64, stride=2)
        self.stage3 = nn.ModuleList([NAS201Cell(blueprint, 64, 64) for _ in range(5)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.stem(x)
        for cell in self.stage1: out = cell(out)
        out = self.down1(out)
        for cell in self.stage2: out = cell(out)
        out = self.down2(out)
        for cell in self.stage3: out = cell(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)

# =====================================================================
# 2. 泛化大考主逻辑
# =====================================================================
def run_generalization_test():
    print("=" * 85)
    print("      【大模型自主进化 ZCP 冠军指标】 - 泛化能力独立盲测大考")
    print("=" * 85)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 🚀 挂载加速设备: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    dummy_input = torch.randn(8, 3, 32, 32, device=device)

    # -------------------------------------------------------------
    # 第一部分：官方实测真值全量盲测 (Ground-Truth Hold-out Test)
    # -------------------------------------------------------------
    samples_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nas201_official_samples.json")
    with open(samples_file, "r", encoding="utf-8") as f:
        official_data = json.load(f)

    print(f"\n[测试一] 载入官方实测真值网络库: {len(official_data)} 个基准模型 (精度跨度: 10.0% ~ 94.37%)")
    
    true_accs = []
    proxy_scores = []
    latencies = []

    for item in official_data:
        model = TinyNetwork201(item["blueprint"]).to(device)
        model.eval()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            s = champion_evaluate(model, dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        lat = (time.perf_counter() - t0) * 1000.0
        
        true_accs.append(item["cifar10_acc"])
        proxy_scores.append(float(s))
        latencies.append(lat)

    rho, p_val = spearmanr(proxy_scores, true_accs)
    tau, tau_pval = kendalltau(proxy_scores, true_accs)
    r_linear, _ = pearsonr(proxy_scores, true_accs)
    avg_lat = sum(latencies) / len(latencies)

    print("\n" + "-" * 70)
    print("                 【测试一：官方真值全量盲测报告】")
    print("-" * 70)
    print(f"★ 斯皮尔曼秩相关系数 (Spearman rho): {rho:.4f} ({rho*100:.1f}%) [p-val: {p_val:.2e}]")
    print(f"★ 肯德尔等级相关系数 (Kendall tau)  : {tau:.4f} ({tau*100:.1f}%)")
    print(f"★ 皮尔逊线性相关系数 (Pearson r)   : {r_linear:.4f} ({r_linear*100:.1f}%)")
    print(f"★ 平均单网络硬件体检耗时           : {avg_lat:.2f} ms / network")
    print("-" * 70)

    # -------------------------------------------------------------
    # 第二部分：50 个从 15,625 搜索空间纯随机采样的未知新架构大考
    # -------------------------------------------------------------
    print(f"\n[测试二] 从 NAS-Bench-201 15,625 搜索空间中纯随机生成 50 个未知新架构...")
    random.seed(2026)  # 固定测试随机种子保证可复现
    
    test_blueprints = []
    test_blueprints.append([3, 3, 3, 3, 3, 3])  # 经典对照: 全3x3大卷积
    test_blueprints.append([0, 0, 0, 0, 0, 0])  # 经典对照: 全断路死网
    test_blueprints.append([1, 1, 1, 1, 1, 1])  # 经典对照: 全直连纯跳线
    test_blueprints.append([4, 4, 4, 4, 4, 4])  # 经典对照: 全池化模糊怪
    for _ in range(46):
        bp = [random.randint(0, 4) for _ in range(6)]
        test_blueprints.append(bp)

    scale_scores = []
    scale_conv3 = []
    scale_params = []
    scale_blueprints = []

    t_scale_start = time.perf_counter()
    for i, bp in enumerate(test_blueprints):
        model = TinyNetwork201(bp).to(device)
        model.eval()
        with torch.no_grad():
            score = float(champion_evaluate(model, dummy_input))
        
        num_conv3 = bp.count(3)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        
        scale_scores.append(score)
        scale_conv3.append(num_conv3)
        scale_params.append(params_m)
        scale_blueprints.append(bp)

    total_scale_time = time.perf_counter() - t_scale_start
    print(f"[*] 50 个未知网络批量体检全部完成！总耗时: {total_scale_time:.2f} 秒 (平均单网 {total_scale_time/50*1000.0:.1f} ms)！")

    # 排序排行榜
    indexed_results = list(zip(range(50), scale_blueprints, scale_scores, scale_conv3, scale_params))
    indexed_results.sort(key=lambda x: x[2], reverse=True)

    print("\n" + "=" * 90)
    print(f"{'名次':<8} | {'模型 ID':<10} | {'拓扑图纸编码 (6个插槽)':<22} | {'ZCP 得分':<14} | {'参数量(M)':<10} | {'3x3卷积数':<8} | {'系统判定'}")
    print("-" * 90)
    for rank, (mid, bp, sc, c3, pm) in enumerate(indexed_results[:5], 1):
        print(f"[TOP {rank:<2}] | Model #{mid:<6} | {str(bp):<22} | {sc:<14.4f} | {pm:<10.2f} | {c3:<8} | 优选高容量结构")
    print("..." + " " * 85)
    for rank, (mid, bp, sc, c3, pm) in enumerate(indexed_results[-5:], 46):
        print(f"[LAST{rank:<2}] | Model #{mid:<6} | {str(bp):<22} | {sc:<14.4f} | {pm:<10.2f} | {c3:<8} | 严重缺陷/淘汰")
    print("=" * 90)

    # -------------------------------------------------------------
    # 第三部分：绘制三合一综合泛化学术图谱
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # 1. 散点拟合图：官方真值 vs 代理得分
    ax1 = axes[0]
    ax1.scatter(proxy_scores, true_accs, color='#1f77b4', s=90, edgecolors='black', alpha=0.85, label='官方测试网络')
    z = np.polyfit(proxy_scores, true_accs, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(proxy_scores), max(proxy_scores), 100)
    ax1.plot(x_line, p(x_line), color='#d62728', linestyle='--', linewidth=2.0, label=f'线性回归拟合线 ($\\rho={rho:.3f}$)')
    ax1.set_title(f'官方测试集拟合度 (Spearman $\\rho = {rho:.4f}$)', fontsize=12, pad=10)
    ax1.set_xlabel('大模型 ZCP 预测打分', fontsize=11)
    ax1.set_ylabel('CIFAR-10 官方真实精度 (%)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=10)

    # 2. 直方图：50 个纯随机未知网络打分频数分布
    ax2 = axes[1]
    ax2.hist(scale_scores, bins=12, color='#2ca02c', edgecolor='black', alpha=0.75)
    ax2.axvline(np.median(scale_scores), color='red', linestyle='--', linewidth=2, label=f'中位数 ({np.median(scale_scores):.3f})')
    ax2.set_title(f'50 个未知新架构 ZCP 打分分布', fontsize=12, pad=10)
    ax2.set_xlabel('ZCP 打分区间', fontsize=11)
    ax2.set_ylabel('网络数量', fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10)

    # 3. 散点关系图：3x3 卷积数量 vs 代理得分 (单调性检验)
    ax3 = axes[2]
    sc = ax3.scatter(scale_conv3, scale_scores, c=scale_params, cmap='viridis', s=90, edgecolors='black', alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label('参数量 (M)', fontsize=10)
    z3 = np.polyfit(scale_conv3, scale_scores, 1)
    p3 = np.poly1d(z3)
    ax3.plot(np.unique(scale_conv3), p3(np.unique(scale_conv3)), color='#d62728', linestyle='--', linewidth=2, label='容量单调趋势线')
    ax3.set_title('核心 3x3 卷积数 vs ZCP 打分 (结构单调性)', fontsize=12, pad=10)
    ax3.set_xlabel('Cell 内 3x3 核心卷积数量', fontsize=11)
    ax3.set_ylabel('ZCP 打分', fontsize=11)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(fontsize=10)

    plt.tight_layout()
    output_png = "champion_generalization_50models.png"
    plt.savefig(output_png)
    print(f"\n[成果归档] 综合泛化学术图谱已保存至: {output_png}")

if __name__ == '__main__':
    run_generalization_test()
