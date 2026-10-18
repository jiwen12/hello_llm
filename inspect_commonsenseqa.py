from datasets import load_dataset

# 1. 加载 CommonsenseQA
dataset = load_dataset("tau/commonsense_qa")

# 2. 查看数据集有哪些划分
print("Dataset splits:")
print(dataset)

# 3. 读取训练集
train_dataset = dataset["train"]

print("\nNumber of training examples:")
print(len(train_dataset))

# 4. 查看第一条样本
example = train_dataset[0]

print("\nFirst example:")
print(example)

# 5. 单独读取关键字段
print("\nQuestion:")
print(example["question"])

print("\nChoices:")
for label, text in zip(
    example["choices"]["label"],
    example["choices"]["text"]
):
    print(f"{label}. {text}")

print("\nGold answer:")
print(example["answerKey"])