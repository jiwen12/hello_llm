from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilbert/distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text = "The future of artificial intelligence is"

inputs = tokenizer(text, return_tensors="pt")

input_ids = inputs["input_ids"]

print("Token IDs:")
print(input_ids)

embedding_layer = model.get_input_embeddings()

print("\nEmbedding layer:")
print(embedding_layer)

print("\nEmbedding dimension:")
print(embedding_layer.embedding_dim)

embeddings = embedding_layer(input_ids)

print("\nEmbedding shape:")
print(embeddings.shape)

print("\nFirst token embedding:")
print(embeddings[0, 0])