# -*- coding: utf-8 -*-
"""
============================================================
 CommonsenseQA 评测 —— log-likelihood 打分版
============================================================

【和原版的核心区别】

  原版：让模型"生成"一个字母 → 再用正则抓出来
        ✗ 依赖模型"听得懂指令"，base 模型（distilgpt2）做不到
        ✗ 13/20 条压根没输出字母，准确率 10% 是噪声

  新版：对每个选项算"模型认为它有多合理" → 取最高分
        ✓ 不需要模型听话，只需要它给概率
        ✓ 不需要正则解析，不会"解析失败"

【打分公式】
    对选项 o，计算：  (1/n) * Σ log P(token_i | 前面所有内容)
    其中 n 是该选项的 token 数（长度归一化，避免偏爱短选项）

【怎么跑】
    cd D:\\AI_Research\\projects\\hello_llm
    python scripts/evaluate_commonsenseqa_logprob.py

【输出】
    results/predictions_logprob.jsonl   逐题明细（含 5 个选项的分数）
    results/summary_logprob.json        汇总指标（含元指标）
============================================================
"""

import json
import math
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===== 1. 实验设置（和原版完全一致：读同一个 config）=====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "commonsenseqa.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

EXPERIMENT_NAME = config["experiment"]["name"] + "_logprob"
MODEL_NAME = config["model"]["name"]
DATASET_NAME = config["dataset"]["name"]
SPLIT = config["dataset"]["split"]
NUM_SAMPLES = config["dataset"]["num_samples"]
SEED = config["generation"]["seed"]

RUN_NAME = "02_logprob_baseline"
OUTPUT_DIR = PROJECT_ROOT / "results" / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)

print("=" * 64)
print("Experiment :", EXPERIMENT_NAME)
print("Model      :", MODEL_NAME)
print("Dataset    :", DATASET_NAME, "| split:", SPLIT)
print("Samples    :", NUM_SAMPLES)
print("Scoring    : log-likelihood (length-normalized)")
print("Seed       :", SEED)
print("=" * 64)

# ===== 2. 加载数据 =====
dataset = load_dataset(DATASET_NAME)
split_data = dataset[SPLIT]
validation_data = split_data.select(range(min(NUM_SAMPLES, len(split_data))))

# ===== 3. 加载模型 =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()


# ===== 4. 核心：给一个选项打分 =====
def _common_prefix_len(a, b):
    """两个 token 列表从头开始相同的长度（处理分词边界差异）"""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@torch.no_grad()
def option_logprob(context, option):
    """
    返回 (该选项的 token 对数似然之和, token 数)

    做法：
      1. 把「题干 + 选项」拼成一句话
      2. 找到"从哪个 token 开始属于选项"
      3. 把选项每个 token 的对数概率加起来
    """
    prefix_text = context + " "
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prefix_text + option, add_special_tokens=False)["input_ids"]

    n_prefix = _common_prefix_len(prefix_ids, full_ids)
    if n_prefix >= len(full_ids):
        # 选项没带来任何新 token（极罕见），给个极低分
        return -1e9, 1

    ids = torch.tensor([full_ids])
    logits = model(ids).logits                      # (1, T, vocab)
    logprobs = torch.log_softmax(logits, dim=-1)    # 对数概率

    total, count = 0.0, 0
    for pos in range(n_prefix, len(full_ids)):
        # 位置 pos 的 token，由 pos-1 位置的输出预测
        total += logprobs[0, pos - 1, full_ids[pos]].item()
        count += 1

    return total, max(count, 1)


def softmax(xs):
    """把分数转成概率，方便看置信度"""
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def predict(example):
    """
    输入一条数据，返回：
        prediction : 模型选的标签（A-E）
        scores     : 5 个选项的平均对数似然
        probs      : 5 个选项归一化后的概率
    """
    context = example["question"]
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]

    scores = []
    for text in texts:
        total, n = option_logprob(context, " " + text.strip() + ".")
        scores.append(total / n)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    probs = softmax(scores)
    return labels[best_idx], scores, probs


# ===== 5. 主循环：逐题预测并记录 =====
results = []

for index, example in enumerate(validation_data):
    prediction, scores, probs = predict(example)
    gold = example["answerKey"]

    results.append({
        "id": example["id"],
        "question": example["question"],
        "choices": example["choices"]["text"],
        "labels": example["choices"]["label"],
        "option_scores": [round(float(s), 4) for s in scores],
        "option_probs": [round(float(p), 4) for p in probs],
        "prediction": prediction,
        "gold_answer": gold,
        "correct": prediction == gold,
        "confidence": round(float(probs[max(range(len(probs)), key=lambda i: scores[i])]), 4),
    })

    print(f"\n--- Q{index + 1} ---")
    print("题目 :", example["question"])
    for lab, txt, sc in zip(example["choices"]["label"],
                            example["choices"]["text"],
                            scores):
        mark = " <<<" if lab == prediction else ""
        print(f"  {lab}. {txt:<28} score={sc:8.4f}{mark}")
    print(f"预测 : {prediction}  |  正确 : {gold}  |  {'✅' if prediction == gold else '❌'}")

# ===== 6. 计算指标（主指标 + 元指标）=====
total = len(results)
correct = sum(r["correct"] for r in results)
accuracy = correct / total if total else 0.0

# 元指标 1：模型的选择分布（是不是总选同一个字母？）
label_counts = {}
for r in results:
    label_counts[r["prediction"]] = label_counts.get(r["prediction"], 0) + 1

# 元指标 2：平均置信度（越高说明越"笃定"，但可能越固执）
mean_conf = sum(r["confidence"] for r in results) / total if total else 0.0

# 元指标 3：选项分数的平均间隔（第一 vs 第二，间隔小说明模型在犹豫）
margins = []
for r in results:
    s = sorted(r["option_scores"], reverse=True)
    margins.append(s[0] - s[1])
mean_margin = sum(margins) / len(margins) if margins else 0.0

print("\n" + "=" * 64)
print("Model      :", MODEL_NAME)
print("Samples    :", total)
print("Correct    :", correct)
print(f"Accuracy   : {accuracy:.4f}   （随机猜 5 选 1 = 0.2000）")
print("Prediction distribution:", label_counts)
print(f"Mean confidence: {mean_conf:.4f}")
print(f"Mean margin (top1-top2): {mean_margin:.4f}")
print("=" * 64)

# ===== 7. 打印失败样本（最重要的分析材料）=====
wrong = [r for r in results if not r["correct"]][:5]
if wrong:
    print("\n===== 失败的 5 个样本（重点看这里）=====")
    for r in wrong:
        print(f"\nQ: {r['question']}")
        print(f"   模型选了 {r['prediction']}，正确答案 {r['gold_answer']}")
        ranked = sorted(zip(r["labels"], r["choices"], r["option_scores"]),
                        key=lambda x: x[2], reverse=True)
        for lab, txt, sc in ranked:
            print(f"     {lab}. {txt:<30} {sc:8.4f}")

# ===== 8. 存档（逐题明细 + 汇总）=====
with open(OUTPUT_DIR / "predictions_logprob.jsonl", "w", encoding="utf-8") as file:
    for result in results:
        file.write(json.dumps(result, ensure_ascii=False) + "\n")

summary = {
    "experiment_name": EXPERIMENT_NAME,
    "model": MODEL_NAME,
    "dataset": DATASET_NAME,
    "split": SPLIT,
    "num_samples": total,
    "correct": correct,
    "accuracy": accuracy,
    "scoring_method": "length_normalized_log_likelihood",
    "seed": SEED,
    "prediction_distribution": label_counts,
    "mean_confidence": round(mean_conf, 4),
    "mean_margin": round(mean_margin, 4),
}

with open(OUTPUT_DIR / "summary_logprob.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2, ensure_ascii=False)

print("\nSaved:", OUTPUT_DIR.resolve())
