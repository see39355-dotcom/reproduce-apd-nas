import time
import random
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 确保 Windows 终端 UTF-8 正常输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from apd_proxy import compute_apd_proxy

# 设置中文字体与学术绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 1. 严格实现 NAS-Bench-201 标准拓扑 Cell 与完整骨干网络 (TinyNetwork)
# =====================================================================

OP_NAMES = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

class NAS201Cell(nn.Module):
    """
    NAS-Bench-201 标准 4 节点、6 边有向无环图 (DAG) 拓扑 Cell
    边序号:
    0: (0 -> 1)
    1: (0 -> 2), 2: (1 -> 2)
    3: (0 -> 3), 4: (1 -> 3), 5: (2 -> 3)
    """
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
    """降采样下采样块 (用于 Stage 之间的过渡)"""
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
    """NAS-Bench-201 官方标准完整网络拓扑"""
    def __init__(self, blueprint, C=16, N=2, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        # Stage 1
        self.stage1 = nn.ModuleList([NAS201Cell(blueprint, C, C) for _ in range(N)])
        self.down1 = ResNetBasicBlock(C, C*2, stride=2)
        # Stage 2
        self.stage2 = nn.ModuleList([NAS201Cell(blueprint, C*2, C*2) for _ in range(N)])
        self.down2 = ResNetBasicBlock(C*2, C*4, stride=2)
        # Stage 3
        self.stage3 = nn.ModuleList([NAS201Cell(blueprint, C*4, C*4) for _ in range(N)])
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C*4, num_classes)

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
# 2. 批量生成 50 个具有真实结构代表性的候选网络并进行大规模 APD 体检
# =====================================================================

def run_large_scale_benchmark(num_candidates=50):
    print("=" * 80)
    print(f"      启动 NAS-Bench-201 大规模机海体检：批量评估 {num_candidates} 个候选网络")
    print("=" * 80)

    # 固定随机种子确保绝对可复现
    random.seed(42)
    torch.manual_seed(42)

    # 1. 随机生成 50 张图纸 (包含全卷积型、混合型、多直连型、甚至畸形残废型)
    candidate_blueprints = []
    
    # 强制放入几个经典的极端对照网络
    candidate_blueprints.append([3, 3, 3, 3, 3, 3])  # 0号: 全3x3卷积豪华配置 (预期顶流)
    candidate_blueprints.append([2, 2, 2, 2, 2, 2])  # 1号: 全1x1卷积配置
    candidate_blueprints.append([1, 1, 1, 1, 1, 1])  # 2号: 全直连纯跳线
    candidate_blueprints.append([0, 0, 0, 0, 0, 0])  # 3号: 全空置死路
    candidate_blueprints.append([4, 4, 4, 4, 4, 4])  # 4号: 全池化模糊怪
    
    # 其余 45 个从 15,625 种空间中纯随机采样
    for _ in range(num_candidates - 5):
        bp = [random.randint(0, 4) for _ in range(6)]
        candidate_blueprints.append(bp)

    dummy_input = torch.randn(16, 3, 32, 32)
    
    eval_results = []
    
    print(f"[开始体检] 正在对 {num_candidates} 个模型执行前向无梯度 APD 扫描...")
    start_total_t = time.perf_counter()
    
    for i, bp in enumerate(candidate_blueprints):
        model = TinyNetwork201(bp)
        
        # 统计图纸中的算子特征
        num_conv3 = bp.count(3)
        num_conv1 = bp.count(2)
        num_skip = bp.count(1)
        num_dead = bp.count(0) + bp.count(4)
        
        # 执行 APD 零成本体检
        t0 = time.perf_counter()
        score = compute_apd_proxy(model, dummy_input)
        lat = (time.perf_counter() - t0) * 1000.0  # ms
        
        # 估算参数量 (M)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        
        eval_results.append({
            "id": i,
            "blueprint": bp,
            "score": score,
            "latency": lat,
            "params": params,
            "num_conv3": num_conv3,
            "num_skip": num_skip,
            "num_dead": num_dead
        })
        
        if (i + 1) % 10 == 0 or i == num_candidates - 1:
            print(f"  -> 已完成 {i+1:>2}/{num_candidates} 个网络体检...")

    total_time = time.perf_counter() - start_total_t
    print(f"\n★ [体检收官] {num_candidates} 个神经网络体检完毕！总耗时: {total_time:.2f} 秒！平均每个网络耗时: {total_time/num_candidates*1000.0:.1f} 毫秒！")

    # =====================================================================
    # 3. 排行榜公示：谁是天选之子？谁是淘汰残废？
    # =====================================================================
    # 按 APD 得分降序排序
    eval_results.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "=" * 90)
    print(f"{'名次':<8} | {'模型 ID':<10} | {'图纸编码 (6个插槽)':<22} | {'s_APD 得分':<14} | {'参数量(M)':<10} | {'3x3卷积数':<8} | {'评价'}")
    print("-" * 90)
    for rank, r in enumerate(eval_results[:5], 1):
        bp_str = str(r['blueprint'])
        print(f"[TOP {rank:<2}] | Model #{r['id']:<6} | {bp_str:<22} | {r['score']:<14.1f} | {r['params']:<10.2f} | {r['num_conv3']:<8} | 优胜天选结构")
    print("..." + " " * 85)
    for rank, r in enumerate(eval_results[-5:], num_candidates - 4):
        bp_str = str(r['blueprint'])
        print(f"[LAST{rank:<2}] | Model #{r['id']:<6} | {bp_str:<22} | {r['score']:<14.1f} | {r['params']:<10.2f} | {r['num_conv3']:<8} | 严重缺陷/被淘汰")
    print("=" * 90)

    # =====================================================================
    # 4. 绘制包含 50 个真实候选网络的学术级全景图
    # =====================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=300)

    scores = [r['score'] for r in eval_results]
    conv3_counts = [r['num_conv3'] for r in eval_results]
    params_list = [r['params'] for r in eval_results]

    # 图 1: 50 个候选网络的体检打分分布直方图 (展示指标的分辨与分层能力)
    ax1.hist(scores, bins=15, color='#1f77b4', edgecolor='black', alpha=0.75)
    ax1.axvline(np.median(scores), color='red', linestyle='--', linewidth=2, label=f'中位数 ({np.median(scores):.1f})')
    ax1.set_title(f'50 个候选网络的 s_APD 体检打分频数分布', fontsize=13, pad=12)
    ax1.set_xlabel('s_APD 打分区间', fontsize=12)
    ax1.set_ylabel('候选网络数量', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # 图 2: 3x3 核心卷积数量 vs APD 得分散点图 (验证指标对骨干表征力的严格单调性)
    scatter = ax2.scatter(conv3_counts, scores, c=params_list, cmap='viridis', s=80, alpha=0.85, edgecolors='black')
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('模型参数量 (M)', fontsize=11)
    
    # 拟合一条趋势线
    z = np.polyfit(conv3_counts, scores, 1)
    p = np.poly1d(z)
    ax2.plot(np.unique(conv3_counts), p(np.unique(conv3_counts)), color='#d62728', linestyle='--', linewidth=2, label='表征能力趋势拟合线')

    ax2.set_title('核心特征数 (3x3卷积) 与 APD 得分的关系 (50个真实网络)', fontsize=13, pad=12)
    ax2.set_xlabel('每个 Cell 内 3x3 核心卷积的数量 (0 ~ 6 个)', fontsize=12)
    ax2.set_ylabel('s_APD 得分', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_png = "nas201_50_models_benchmark.png"
    plt.savefig(output_png)
    print(f"\n[成果归档] 50 个网络的批量体检全景图已保存至: {output_png}")

if __name__ == '__main__':
    run_large_scale_benchmark(50)
