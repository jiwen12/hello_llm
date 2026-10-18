# !!! DEPRECATED !!! 本脚本用'生成+正则解析'，在 base 模型上失效（准确率 0.10，13/20 无法解析）。
# 请改用 scripts/evaluate_commonsenseqa_logprob.py
import json
import re
from pathlib import Path
import yaml

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 1. 实验设置 =====

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "commonsenseqa.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

EXPERIMENT_NAME = config["experiment"]["name"]
MODEL_NAME = config["model"]["name"]
DATASET_NAME = config["dataset"]["name"]
SPLIT = config["dataset"]["split"]
NUM_SAMPLES = config["dataset"]["num_samples"]

MAX_NEW_TOKENS = config["generation"]["max_new_tokens"]
DO_SAMPLE = config["generation"]["do_sample"]
SEED = config["generation"]["seed"]

RUN_NAME = "01_generate_baseline"
OUTPUT_DIR = PROJECT_ROOT / "results" / RUN_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)

print("Experiment:", EXPERIMENT_NAME)
print("Config:", CONFIG_PATH)
print("Model:", MODEL_NAME)
print("Dataset:", DATASET_NAME)
print("Samples:", NUM_SAMPLES)

# ===== 2. 加载验证集 ====
dataset = load_dataset(DATASET_NAME)
split_data = dataset[SPLIT]

validation_data = split_data.select(
    range(min(NUM_SAMPLES, len(split_data)))
)

# ===== 3. 加载模型和分词器 =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# GPT-2 默认没有单独的 padding token。
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===== 4. 构造提示词 =====
def build_prompt(example):
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]

    choices_text = "\n".join(
        f"{label}. {text}"
        for label, text in zip(labels, texts)
    )

    return (
        "Answer the following multiple-choice question "
        "using only one letter: A, B, C, D, or E.\n"
        f"Question: {example['question']}\n"
        f"{choices_text}\n"
        "Answer:"
    )

# ===== 5. 从模型输出中提取答案 =====
def parse_answer(text):
    match = re.match(
        r"\s*([A-E])\b",
        text.upper()
    )
    return match.group(1) if match else None

# ===== 6. 执行预测并记录结果 =====
results = []

for index, example in enumerate(validation_data):
    prompt = build_prompt(example)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
       output_ids = model.generate(
    **inputs,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=DO_SAMPLE,
    pad_token_id=tokenizer.pad_token_id,
)

    # 只解码新生成的部分，不包含原始提示词。
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    prediction = parse_answer(generated_text)
    gold_answer = example["answerKey"]

    results.append({
        "id": example["id"],
        "question": example["question"],
        "prompt": prompt,
        "generated_text": generated_text,
        "prediction": prediction,
        "gold_answer": gold_answer,
        "correct": prediction == gold_answer,
    })

    print(f"\nQuestion {index + 1}")
    print("Prediction:", prediction)
    print("Gold answer:", gold_answer)
    print("Correct:", prediction == gold_answer)

# ===== 7. 计算指标 =====
total = len(results)
correct = sum(
    result["correct"] for result in results
)
accuracy = correct / total if total else 0.0

print("\n===== Evaluation Summary =====")
print("Model:", MODEL_NAME)
print("Samples:", total)
print("Correct:", correct)
print(f"Accuracy: {accuracy:.4f}")

# ===== 8. 保存逐题结果和实验摘要 =====
with open(
    OUTPUT_DIR / "predictions.jsonl",
    "w",
    encoding="utf-8"
) as file:
    for result in results:
        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

summary = {
    "experiment_name": EXPERIMENT_NAME,
    "model": MODEL_NAME,
    "dataset": DATASET_NAME,
    "split": SPLIT,
    "num_samples": total,
    "correct": correct,
    "accuracy": accuracy,
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": DO_SAMPLE,
    "seed": SEED,
}

with open(
    OUTPUT_DIR / "summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=2, ensure_ascii=False)

print("\nSaved results to:", OUTPUT_DIR.resolve())