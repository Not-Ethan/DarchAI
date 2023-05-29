from transformers import BartTokenizer, BartForConditionalGeneration

test_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
test_model = BartForConditionalGeneration.from_pretrained("services/text_process/ai/taglines/model_loader/models/bart_checkpoint_2")

def generate_tagline(text: str) -> str:
    inputs = test_tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    try:
        outputs = test_model.generate(inputs, max_length=200, min_length=20, length_penalty=0.5, num_beams=16, early_stopping=True, repetition_penalty=50.0, num_return_sequences=1)
    except IndexError as e:
        print(f"Error occurred with input: {text}")
        raise e

    return test_tokenizer.decode(outputs[0], skip_special_tokens=True)