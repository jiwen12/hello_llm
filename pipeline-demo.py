from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilbert/distilgpt2"
)

prompt = "The future of artificial intelligence is"

result = generator(
    prompt,
    max_new_tokens=30,
    do_sample=False,
    return_full_text=False
)

print(result[0]["generated_text"])