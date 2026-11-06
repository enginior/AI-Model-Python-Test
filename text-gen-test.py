from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

result = generator("Once upon a time,", max_length=30, num_return_sequences=1)
print(result[0]['generated_text'])
