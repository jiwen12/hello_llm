from transformers import AutoTokenizer

model_name = "distilbert/distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "The future of artificial intelligence is"

inputs = tokenizer(text, return_tensors="pt")

print("原始文本：")
print(text)

print("\nToken IDs：")
print(inputs["input_ids"])

print("\nToken：")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))