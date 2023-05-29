from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('services/text_process/ai/sentence_models/model_loader/models/distilbert_mnli_one_output', num_labels=1)

def distilbert(argument, sentence, max_len=256):
    inputs = tokenizer.encode_plus(
        argument,
        sentence,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    model.eval()

    # Get the model's output
    with torch.no_grad():
        outputs = model(**inputs.to(model.device))

    # Get the model's prediction
    prediction = torch.sigmoid(outputs.logits).item()

    return prediction
