from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilbert/distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text = "The future of artificial intelligence is"

inputs = tokenizer(text, return_tensors="pt")

input_ids = inputs["input_ids"]

# 获取 Token Embedding 层
embedding_layer = model.get_input_embeddings()

# Token ID → Embedding
embeddings = embedding_layer(input_ids)

print("Embedding shape:")
print(embeddings.shape)

# 获取第一层 Attention
attn = model.transformer.h[0].attn

# Embedding → QKV
qkv = attn.c_attn(embeddings)

print("\nQKV shape:")
print(qkv.shape)

# 拆分 Q、K、V
q, k, v = qkv.split(768, dim=-1)

print("\nQ shape:")
print(q.shape)

print("\nK shape:")
print(k.shape)

print("\nV shape:")
print(v.shape)

# GPT-2 的 Attention 头数量
num_heads = model.config.n_head

# 每个 head 的维度
head_dim = model.config.n_embd // num_heads

print("\nNumber of heads:")
print(num_heads)

print("\nHead dimension:")
print(head_dim)

# 把 Q、K、V 从 [batch, sequence, hidden]
# 变成 [batch, heads, sequence, head_dim]
q = q.view(1, 6, num_heads, head_dim).transpose(1, 2)
k = k.view(1, 6, num_heads, head_dim).transpose(1, 2)
v = v.view(1, 6, num_heads, head_dim).transpose(1, 2)

print("\nQ after splitting heads:")
print(q.shape)

print("\nK after splitting heads:")
print(k.shape)

print("\nV after splitting heads:")
print(v.shape)

import torch

# 计算 Attention Score
attention_scores = torch.matmul(q, k.transpose(-1, -2))

# 缩放
attention_scores = attention_scores / (head_dim ** 0.5)

print("\nAttention scores shape:")
print(attention_scores.shape)

print("\nAttention scores for head 0:")
print(attention_scores[0, 0])

# 创建 causal mask
mask = torch.tril(torch.ones(6, 6))

print("\nCausal mask:")
print(mask)

# 将未来位置的 Attention Score 设置为负无穷
masked_scores = attention_scores.masked_fill(
    mask == 0,
    float("-inf")
)

print("\nMasked attention scores for head 0:")
print(masked_scores[0, 0])

# 对最后一个维度进行 Softmax
attention_weights = torch.softmax(masked_scores, dim=-1)

print("\nAttention weights for head 0:")
print(attention_weights[0, 0])

print("\nRow sums:")
print(attention_weights[0, 0].sum(dim=-1))

# Attention weights × V
attention_output = torch.matmul(attention_weights, v)

print("\nAttention output shape:")
print(attention_output.shape)

print("\nAttention output for head 0, first token:")
print(attention_output[0, 0, 0])

# 合并 12 个 attention heads
attention_output = attention_output.transpose(1, 2)

print("\nAfter transpose:")
print(attention_output.shape)

# 将 12 个 head 的 64 维向量拼接起来
attention_output = attention_output.contiguous().view(
    1, 6, 12 * 64
)

print("\nAfter merging heads:")
print(attention_output.shape)

# 输出投影
projected_output = attn.c_proj(attention_output)

print("\nAfter c_proj:")
print(projected_output.shape)

# 第一步：交换 head 和 sequence 两个维度
attention_output = attention_output.transpose(1, 2)

print("\nAfter transpose:")
print(attention_output.shape)

# 第二步：把 12 个 head 的向量拼接起来
attention_output = attention_output.contiguous().view(
    1, 6, 12 * 64
)

print("\nAfter merging heads:")
print(attention_output.shape)

# 多头 Attention 输出投影
projected_output = attn.c_proj(attention_output)

print("\nAfter c_proj:")
print(projected_output.shape)


# 取出第一个 GPT-2 Block
block = model.transformer.h[0]

# 使用当前 embeddings 做演示
x = embeddings

print("输入形状：", x.shape)

# 1. LayerNorm
normalized_x = block.ln_1(x)
print("ln_1 后：", normalized_x.shape)

# 2. MLP 的第一层
mlp_hidden = block.mlp.c_fc(normalized_x)
print("MLP c_fc 后：", mlp_hidden.shape)

# 3. GELU 激活
mlp_hidden = block.mlp.act(mlp_hidden)
print("GELU 后：", mlp_hidden.shape)

# 4. MLP 的第二层
mlp_output = block.mlp.c_proj(mlp_hidden)
print("MLP c_proj 后：", mlp_output.shape)


import torch

model.eval()

text = "The future of artificial intelligence is"
inputs = tokenizer(text, return_tensors="pt")

input_ids = inputs["input_ids"]

# Token Embedding + Position Embedding
position_ids = torch.arange(
    input_ids.shape[1]
).unsqueeze(0)

token_embeds = model.transformer.wte(input_ids)
position_embeds = model.transformer.wpe(position_ids)

hidden_states = token_embeds + position_embeds

print("Block 输入：", hidden_states.shape)

# 运行第一个 GPT-2 Block
block = model.transformer.h[0]

print("hidden_states 形状：", hidden_states.shape)
print("block 类型：", type(block))
print("block forward：", block.forward)

with torch.no_grad():
    result = block(hidden_states)

print("result 形状：", result.shape)