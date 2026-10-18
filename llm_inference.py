
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 实验配置
model_name = "distilbert/distilgpt2"
prompt = "The future of artificial intelligence is"
max_new_tokens = 30
seed = 42

# 2. 固定随机种子，便于复现实验
torch.manual_seed(seed)

# 3. 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 4. 准备输入
inputs = tokenizer(prompt, return_tensors="pt")

# 5. 生成文本
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

# 6. 解码输出
result = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

# 7. 记录实验信息
print("Model:", model_name)
print("Prompt:", prompt)
print("Max new tokens:", max_new_tokens)
print("Seed:", seed)
print("Generated text:", result)
