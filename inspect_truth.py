import os
import json

print("Current Working Directory:", os.getcwd())
files = [f for f in os.listdir(".") if "nats201" in f or "truth" in f]
print("Matching files in CWD:", files)

truth_file = "nats201_cifar10_truth_all15625.json"
if os.path.exists(truth_file):
    with open(truth_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("=" * 65)
    print(f"【成功】已成功从官方 NATS-Bench 载入全部 {len(data)} 个神经网络拓扑！")
    print(f"最差拓扑 (死网) Arch #0: {data[0]['arch_str']}")
    print(f"官方实测 CIFAR-10 准确率: {data[0]['cifar10_acc']}%")
    print("-" * 65)
    print(f"中位拓扑 Arch #5000: {data[5000]['arch_str']}")
    print(f"官方实测 CIFAR-10 准确率: {data[5000]['cifar10_acc']}%")
    print("-" * 65)
    
    # 查找最高 SOTA 拓扑
    best_item = max(data, key=lambda x: x["cifar10_acc"])
    print(f"最高 SOTA 拓扑 Arch #{best_item['idx']}: {best_item['arch_str']}")
    print(f"官方实测 CIFAR-10 准确率: {best_item['cifar10_acc']}%")
    print("=" * 65)
else:
    print("Looking for file in parent and other dirs...")
    for root, dirs, fnames in os.walk(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        if truth_file in fnames:
            found = os.path.join(root, truth_file)
            print("Found at:", found)
            # Copy to current dir
            import shutil
            shutil.copy2(found, truth_file)
            print("Copied to current dir!")
            break
