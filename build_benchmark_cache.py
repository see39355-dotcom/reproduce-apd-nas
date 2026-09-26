"""
从官方 NATS-Bench 完整数据库中一键提取全部 15,625 个架构的拓扑编码与 CIFAR-10 真实测试准确率。
生成极速轻量缓存文件 nats201_cifar10_truth_all15625.json (~1.2 MB)。
"""
import os
import sys
import gc
import json
import pickle
import time

OP_MAP = {
    'none': 0,
    'skip_connect': 1,
    'nor_conv_1x1': 2,
    'nor_conv_3x3': 3,
    'avg_pool_3x3': 4
}

def parse_arch_to_blueprint(arch_str: str):
    """
    将 NATS-Bench / NAS-Bench-201 官方拓扑字符串解析为 6 维算子整数列表
    示例：|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
    解析对应边:
    0: (0->1)
    1: (0->2), 2: (1->2)
    3: (0->3), 4: (1->3), 5: (2->3)
    """
    nodes = arch_str.strip().split('+')
    edges = []
    for node_segment in nodes:
        parts = [p for p in node_segment.split('|') if p]
        for part in parts:
            op_name, in_node = part.split('~')
            edges.append(OP_MAP[op_name])
    return edges

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_json = os.path.join(script_dir, "nats201_cifar10_truth_all15625.json")
    pickle_path = os.path.join(os.path.dirname(script_dir), "NATS-tss-v1_0-3ffb9.pickle")
    if not os.path.exists(pickle_path):
        pickle_path = os.path.join("..", "NATS-tss-v1_0-3ffb9.pickle")
    if not os.path.exists(pickle_path):
        pickle_path = "NATS-tss-v1_0-3ffb9.pickle"

    if not os.path.exists(pickle_path):
        print(f"[错误] 未找到解压后的 pickle 文件: {pickle_path}")
        return

    print("=" * 70)
    print("【1/3】正在从您下载的官方全量数据库中提取全部 15,625 个模型真值...")
    print(f"       输入文件: {os.path.abspath(pickle_path)}")
    print(f"       目标输出: {os.path.abspath(out_json)}")
    t0 = time.perf_counter()
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    t1 = time.perf_counter()
    print(f"       载入成功！耗时: {t1 - t0:.2f} 秒")

    meta_archs = data["meta_archs"]
    arch2infos = data["arch2infos"]
    total = len(meta_archs)
    print(f"【2/3】正在解析全部 {total} 个神经网络拓扑结构与官方 CIFAR-10 实测准确率...")

    records = []
    for idx in range(total):
        arch_str = meta_archs[idx]
        bp = parse_arch_to_blueprint(arch_str)
        
        # 提取 CIFAR-10 最终测试准确率 (优先提取 200 epoch 的全训练真值)
        info_dict = arch2infos.get(idx, {})
        hp_key = 200 if 200 in info_dict else (12 if 12 in info_dict else list(info_dict.keys())[0] if info_dict else None)
        
        true_acc = 0.0
        if hp_key is not None and 'cifar10' in info_dict[hp_key]:
            c10 = info_dict[hp_key]['cifar10']
            seeds = [k for k in c10.keys() if k.startswith('seed@')]
            if seeds:
                accs = []
                for s in seeds:
                    eval_accs = c10[s].get('eval_acc1es', {}).get('ori-test', [])
                    if eval_accs:
                        accs.append(eval_accs[-1])
                if accs:
                    true_acc = round(sum(accs) / len(accs), 4)

        records.append({
            "idx": idx,
            "arch_str": arch_str,
            "blueprint": bp,
            "cifar10_acc": true_acc
        })

    # 释放原始巨型字典占用的内存
    del data, meta_archs, arch2infos
    gc.collect()

    print(f"【3/3】正在保存轻量极速真值库 -> {out_json}")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=None)

    # 自动清理临时解压出的 1.9GB 原始 pickle，仅保留 2MB 极速 json 和官方 .pbz2 压缩包
    try:
        if os.path.exists(pickle_path):
            os.remove(pickle_path)
            print("  已自动清理临时解压的巨大 pickle 缓存，系统空间干净清爽！")
    except Exception as e:
        print(f"  [提示] 临时 pickle 文件清理跳过: {e}")

    file_size_mb = os.path.getsize(out_json) / (1024 * 1024)
    print("=" * 70)
    print(f"  恭喜！全部 15,625 个官方模型真值已完整提取！")
    print(f"  真值文件大小: {file_size_mb:.2f} MB")
    print(f"  后续评测加载仅需 0.01 秒，彻底摆脱数分钟加载和内存膨胀问题！")
    print("=" * 70)

if __name__ == "__main__":
    main()
