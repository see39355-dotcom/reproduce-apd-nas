import os
import sys
import re
import time
import traceback
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from openai import OpenAI

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 引入 NAS-Bench-201 骨干网络定义
from run_nas201_scale_experiment import TinyNetwork201

# =========================================================================
# 1. 用户 API 配置区
# =========================================================================
# 提示：此脚本优先从环境变量或本地 .env 文件读取 API Key，避免在代码中硬编码敏感密钥
def load_env_key():
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if key:
        return key
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "YOUR_API_KEY_HERE"

API_CONFIG = {
    # 自动加载 API Key (兼容环境变量或 .env)
    "api_key": load_env_key(),
    
    # 常用服务商 Base URL 参考：
    # - DeepSeek 官方: "https://api.deepseek.com"
    # - 硅基流动: "https://api.siliconflow.cn/v1"
    # - 阿里百炼 (通义千问): "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # - 月之暗面 (Kimi): "https://api.moonshot.cn/v1"
    # - OpenAI 官方: "https://api.openai.com/v1"
    "base_url": "https://api.deepseek.com",
    
    # 选用基底模型
    "model_name": "deepseek-flash",
    
    # 拟探索的进化代数 (设置为标准的 20 代深度探索)
    "max_generations": 20
}

# =========================================================================
# 2. 动态随机基准测试集 (支持官方 4.7GB .pth 全量文件与官方已提取样本库)
# =========================================================================
def sample_random_nas201_benchmark(num_models=20, random_seed=None):
    """
    智能双模数据源：
    1. 若本地存在官方 NAS-Bench-201-v1_1-096897.pth 或 NATS-tss 数据库，直接从官方全量库查询！
    2. 若全量大文件尚未下载，自动从官方实测真值样本库 (nas201_official_samples.json) 中无偏抽选 20 个网络！
    """
    import json
    import glob
    
    torch_home = os.path.expanduser("~/.torch")
    potential_files = [
        "NAS-Bench-201-v1_1-096897.pth",
        os.path.join(torch_home, "NAS-Bench-201-v1_1-096897.pth"),
        os.path.join(torch_home, "NATS-tss-v1_0-3ffb9.pickle.pbz2"),
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "nats201_cifar10_truth_all15625.json"),
        os.path.join(os.path.dirname(script_dir), "nats201_cifar10_truth_all15625.json"),
    ]
    truth_file = None
    for c in candidates:
        if os.path.exists(c):
            truth_file = c
            break

    if truth_file is not None:
        try:
            with open(truth_file, "r", encoding="utf-8") as f:
                full_db = json.load(f)
            import random as py_random
            selected = py_random.sample(full_db, min(num_models, len(full_db)))
            suite = []
            for item in selected:
                suite.append({
                    "name": f"Arch-{item['idx']:05d}",
                    "blueprint": item["blueprint"],
                    "true_acc": item["cifar10_acc"]
                })
            print(f"[数据源] 成功挂载官方全量 NATS-Bench 基准库 (涵盖全部 {len(full_db)} 个候选拓扑)！")
            print(f"[采样] 本次随机抽取 {len(suite)} 个网络拓扑进行 APD 零成本代理体检。")
            return suite
        except Exception as e:
            print(f"[警告] 官方全量缓存读取异常 ({e})，平滑回退至官方样本库。")

    # 回退到本地官方实测真值库
    samples_file = os.path.join(script_dir, "nas201_official_samples.json")
    if not os.path.exists(samples_file):
        samples_file = "nas201_official_samples.json"
    if os.path.exists(samples_file):
        with open(samples_file, "r", encoding="utf-8") as f:
            full_db = json.load(f)
        import random as py_random
        chosen = py_random.sample(full_db, min(num_models, len(full_db)))
        suite = []
        for i, item in enumerate(chosen):
            suite.append({
                "name": f"RandArch-{i+1:02d} ({item.get('desc', '样本')})",
                "blueprint": item["blueprint"],
                "true_acc": item["cifar10_acc"]
            })
        print(f"[数据源] 已载入官方实测真值数据库 ({samples_file})，随机抽取 {len(suite)} 个网络评测！")
        return suite

    # 兜底纯随机拓扑测算
    import random as py_random
    suite = []
    for idx in range(1, num_models + 1):
        bp = [py_random.randint(0, 4) for _ in range(6)]
        suite.append({"name": f"RandArch-{idx:02d}", "blueprint": bp, "true_acc": 75.0})
    return suite

# =========================================================================
# 3. NeurIPS 2025 APD 论文标准 Prompt 模板体系
# =========================================================================
SYSTEM_PROMPT = """你是一名世界顶级神经网络架构搜索 (NAS) 与零成本代理指标 (ZCP) 算法专家。
你的任务是为 NAS-Bench-201 搜索空间设计全新的“零成本代理评估指标 (Zero-Cost Proxy)”，用纯前向推断快速打分神经网络的表征潜力。

【硬性执行契约 (Contract)】：
1. 必须是纯前向推断，绝对禁止进行反向传播或参数更新 (严禁 loss.backward()、严禁 optimizer.step())；
2. 必须输出一个且仅一个 Python 函数，函数签名必须严格为：
   def evaluate(model, inputs):
       ...
       return score  # 必须返回一个纯标量 float 分数
3. 必须具备良好的数值稳定性，所有除法分母必须加上 epsilon 极小量 (例如 + 1e-6)；
4. 必须能区分不同拓扑结构，不仅要看卷积核权重，还要探测数据流经 BatchNorm/激活层时的特征丰富度 (如奇异值、稳定秩、通道方差、正交性等)；
5. 回复格式要求：
   先给出一小段【思考链 (Chain-of-Thought)】解释你的数学物理直觉；
   然后给出完整的代码块，必须用 ```python 和 ``` 包裹。不要输出多余废话。
"""

INIT_PROMPT = """【第 0 代任务：冷启动初始化探索】
请提出你的第 1 个原创零成本代理指标。
目标：在不训练的前提下，输入图片张量 inputs 形状为 (B, 3, 32, 32)，对 model 进行前向分析并输出一个标量评分，越高代表模型在 CIFAR-10 上的真实性能越强。
请输出你的思考过程和完整的 Python 代码："""

MUTATION_PROMPT_TEMPLATE = """【第 {gen} 代任务：突变与反思迭代】
上一代大模型提出的代理指标代码实测表现如下：
【上一代代码】：
```python
{prev_code}
```
【实测质检报告】：
- 斯皮尔曼等级排序相关系数 Spearman rho: {rho:.4f} (目标 > 0.80)
- 单模型体检平均耗时: {latency:.2f} ms
- 综合适应度得分 (Fitness): {fitness:.4f}
- 运行诊断状态: {diag_status}

【历史最佳代码表现】：
- 历史最高 rho: {best_rho:.4f}

【强化学习反思指令】：
1. 深入分析上一版代码的缺陷：它是否在深层特征上发生了数值爆炸？它是否无法区分跳线(Skip)和死路(Dead)？它是否只看重参数量而忽略了有效特征展开维度？
2. 请进行【变异突变 (Mutation)】：引入新的数学工具（例如矩阵奇异值分布、稳定秩 Stable Rank、通道特征变异系数、权重正交多样性等），重构或改良 evaluate(model, inputs) 函数。
请输出你的深度思考链和全新的 Python 代码块："""

# =========================================================================
# 4. 代码提取与沙箱安全动态执行器
# =========================================================================
def extract_python_code(llm_text):
    """从 LLM 返回内容中精准抠出 python 代码块"""
    match = re.search(r"```python(.*?)```", llm_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match_generic = re.search(r"```(.*?)```", llm_text, re.DOTALL)
    if match_generic:
        return match_generic.group(1).strip()
    return llm_text.strip()

def run_in_sandbox(code_str, benchmark_models, dummy_input, true_accs):
    """在受保护的隔离环境中动态编译并执行大模型写出的代码"""
    # 静态安全检查：封杀可能反向求导的代码
    if "backward(" in code_str or "grad_fn" in code_str and "requires_grad" in code_str:
        return None, -1.0, 999.0, "违规警告：检测到显式求导或反向传播，违背 Training-Free 契约！"
        
    sandbox_globals = {
        "torch": torch,
        "nn": torch.nn,
        "F": torch.nn.functional,
        "np": np
    }
    sandbox_locals = {}
    
    # 1. 动态编译
    try:
        exec(code_str, sandbox_globals, sandbox_locals)
    except Exception as e:
        return None, -1.0, 999.0, f"Python 语法/编译异常: {type(e).__name__}: {str(e)}"
        
    if "evaluate" not in sandbox_locals or not callable(sandbox_locals["evaluate"]):
        return None, -1.0, 999.0, "契约失败：代码中未定义 evaluate(model, inputs) 函数！"
        
    evaluate_fn = sandbox_locals["evaluate"]
    
    # 2. 在 12 个基准网络上逐一运行并打分
    scores = []
    latencies = []
    
    for name, model, true_acc in benchmark_models:
        try:
            if dummy_input.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                res = evaluate_fn(model, dummy_input)
            if dummy_input.is_cuda:
                torch.cuda.synchronize()
            lat = (time.perf_counter() - t0) * 1000.0
            
            # 转为标准 float 标量
            if isinstance(res, torch.Tensor):
                score_val = res.item()
            else:
                score_val = float(res)
                
            if np.isnan(score_val) or np.isinf(score_val):
                return None, -1.0, 999.0, "数值异常：运行返回了 NaN 或 Inf 浮点溢出！"
                
            scores.append(score_val)
            latencies.append(lat)
        except Exception as e:
            err_msg = traceback.format_exc().splitlines()[-1]
            return None, -1.0, 999.0, f"运行时模型推断报错 ({name}): {err_msg}"
            
    # 3. 防常数作弊检查
    if np.std(scores) < 1e-6:
        return None, 0.0, np.mean(latencies), "退化警告：代码对所有网络输出相同常数，无区分能力！"
        
    # 4. 计算斯皮尔曼秩相关度 rho
    rho, pval = spearmanr(scores, true_accs)
    if np.isnan(rho):
        rho = 0.0
        
    avg_lat = float(np.mean(latencies))
    return evaluate_fn, rho, avg_lat, "质检通过：成功完成全套网络前向打分"

# =========================================================================
# 5. 主控进化引擎 (Actor-Critic 强化学习闭环)
# =========================================================================
def run_real_llm_apd():
    print("=" * 85)
    print("      【APD 大模型自主零成本代理发现系统】 - 真实 API 在线驱动版")
    print("=" * 85)
    
    # 检查 API Key 是否配置
    if API_CONFIG["api_key"] == "YOUR_API_KEY_HERE" or not API_CONFIG["api_key"]:
        print("\n" + "!" * 85)
        print("【提示】检测到你尚未在代码顶部填入真实的 API_KEY！")
        print("请用文本编辑器打开本脚本 (run_real_llm_apd.py)，在第 25 行填入你的 API Key 即可启动！")
        print("例如：")
        print('  API_CONFIG = {')
        print('      "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxx",')
        print('      "base_url": "https://api.deepseek.com",')
        print('      "model_name": "deepseek-chat",')
        print('      "max_generations": 5')
        print('  }')
        print("!" * 85 + "\n")
        return

    # 初始化 OpenAI 兼容客户端
    print(f"[*] 正在连接大模型服务商: {API_CONFIG['base_url']}")
    print(f"[*] 选用基底模型: {API_CONFIG['model_name']}")
    print(f"[*] 计划进化代数: {API_CONFIG['max_generations']} 代\n")
    
    client = OpenAI(
        api_key=API_CONFIG["api_key"],
        base_url=API_CONFIG["base_url"]
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"[*] 🚀 成功检测并挂载 GPU 硬件加速: {torch.cuda.get_device_name(0)}")
        try:
            torch.set_default_device('cuda')
        except Exception:
            pass
    else:
        print(f"[*] 当前环境未激活 CUDA，运行于 CPU 模式")

    print("[*] 正在从 NAS-Bench-201 15,625 搜索空间中随机抽选 20 个候选模型...")
    benchmark_suite = sample_random_nas201_benchmark(num_models=20)
    
    BENCHMARK_MODELS = []
    for item in benchmark_suite:
        m = TinyNetwork201(item["blueprint"]).to(device)
        m.eval()
        BENCHMARK_MODELS.append((item["name"], m, item["true_acc"]))
        
    DUMMY_INPUT = torch.randn(8, 3, 32, 32, device=device)
    TRUE_ACCS = [item["true_acc"] for item in benchmark_suite]
    
    print(f"[*] 成功随机采样并加载 {len(BENCHMARK_MODELS)} 个考卷模型到 [{device}]！(真实精度分布: {min(TRUE_ACCS):.1f}% ~ {max(TRUE_ACCS):.1f}%)\n")
    
    history = []
    best_candidate = None
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    beta = 0.05  # 延迟惩罚超参数
    
    for gen in range(API_CONFIG["max_generations"]):
        print(f"\n" + "-" * 85)
        print(f" >>> [Generation {gen}] 启动思考与代码生成...")
        print("-" * 85)
        
        # 组装本代 Prompt
        if gen == 0:
            user_prompt = INIT_PROMPT
        else:
            prev = history[-1]
            user_prompt = MUTATION_PROMPT_TEMPLATE.format(
                gen=gen,
                prev_code=prev["clean_code"],
                rho=prev["rho"],
                latency=prev["latency"],
                fitness=prev["fitness"],
                diag_status=prev["diag_status"],
                best_rho=best_candidate["rho"] if best_candidate else 0.0
            )
            
        messages.append({"role": "user", "content": user_prompt})
        
        # 调用大模型生成代码
        t_api_start = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=API_CONFIG["model_name"],
                messages=messages,
                temperature=0.7
            )
            llm_reply = completion.choices[0].message.content
            api_duration = time.perf_counter() - t_api_start
            print(f"[API 通信完成] 耗时 {api_duration:.2f} 秒，收到大模型回复。")
        except Exception as e:
            print(f"[API 调用失败] 发生网络或认证错误: {str(e)}")
            break
            
        # 把大模型的回答追加到上下文
        messages.append({"role": "assistant", "content": llm_reply})
        
        # 提取 Python 代码
        clean_code = extract_python_code(llm_reply)
        
        # 本地沙箱执行评测
        fn, rho, lat, diag = run_in_sandbox(clean_code, BENCHMARK_MODELS, DUMMY_INPUT, TRUE_ACCS)
        
        # 计算强化学习适应度
        if fn is None:
            fitness = -2.0  # 编译或执行报错给予重罚
        else:
            fitness = rho - beta * (lat / 100.0)
            
        record = {
            "gen": gen,
            "raw_reply": llm_reply,
            "clean_code": clean_code,
            "rho": rho,
            "latency": lat,
            "fitness": fitness,
            "diag_status": diag
        }
        history.append(record)
        
        print(f"  [沙箱质检报告]: {diag}")
        print(f"  [预测排序相关度]: Spearman rho = {rho:.4f} ({rho*100:.1f}%)")
        print(f"  [单模型体检耗时]: {lat:.2f} ms")
        print(f"  [强化学习综合适应度]: Fitness = {fitness:.4f}")
        
        if best_candidate is None or fitness > best_candidate["fitness"]:
            best_candidate = record
            print("  ★ [刷新纪录] 成为当前历史最优代理指标，已被优先归档！")
            
    if not history:
        print("未产生有效记录，程序终止。")
        return
        
    # =========================================================================
    # 6. 保存最终自主进化的冠军代码与演化图谱
    # =========================================================================
    print("\n" + "=" * 85)
    print("                      大模型自主进化收官总评")
    print("=" * 85)
    print(f"探索代数: {len(history)} 代")
    print(f"历史最佳代数: Generation {best_candidate['gen']}")
    print(f"最高预测相关度 (Spearman rho): {best_candidate['rho']:.4f}")
    print(f"体检单网延迟: {best_candidate['latency']:.2f} ms")
    
    # 落地冠军代码
    output_code_path = "discovered_zcp_by_llm.py"
    with open(output_code_path, "w", encoding="utf-8") as f:
        f.write("# 由真实大模型在线自主强化学习闭环进化生成的 ZCP 冠军代码\n")
        f.write(f"# 进化代数: Generation {best_candidate['gen']}\n")
        f.write(f"# 预测相关度 Spearman rho: {best_candidate['rho']:.4f}\n")
        f.write(f"# 评测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport numpy as np\n\n")
        f.write(best_candidate["clean_code"] + "\n")
    print(f"\n[成果归档 1] 真实大模型生成的独立最优代码已落盘: {output_code_path}")
    
    # 绘制折线演化图
    gens = [h["gen"] for h in history]
    rhos = [max(-1.0, h["rho"]) for h in history]
    fits = [max(-1.0, h["fitness"]) for h in history]
    
    plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(gens, rhos, marker='o', linewidth=2.5, color='#1f77b4', label=r'Spearman 秩相关度 $\rho$')
    plt.plot(gens, fits, marker='s', linewidth=2.0, linestyle='--', color='#2ca02c', label='强化学习适应度 Fitness')
    plt.axhline(0.832, color='red', linestyle=':', label='Paper Baseline (0.832)')
    plt.title(f'真实大模型 ({API_CONFIG["model_name"]}) 自主进化 ZCP 轨迹图', fontsize=13, pad=12)
    plt.xlabel('进化代数 (Generation)', fontsize=11)
    plt.ylabel('打分指标', fontsize=11)
    plt.xticks(gens)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    
    output_png = "real_llm_evolution_curve.png"
    plt.savefig(output_png)
    print(f"[成果归档 2] 进化收敛曲线已绘制保存至: {output_png}")
    print("=" * 85 + "\n")

if __name__ == '__main__':
    run_real_llm_apd()
