from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def generate_tagline(text: str) -> str:
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=50, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True, top_p=0.9)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)