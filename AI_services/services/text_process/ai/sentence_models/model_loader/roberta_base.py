import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

xtr_model_roberta = RobertaForSequenceClassification.from_pretrained("services/text_process/ai/sentence_models/model_loader/models/xtr_mvm_roberta")
test_xtr_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def roberta_base(sentence1: str, sentence2: str) -> float:

    inputs = test_xtr_tokenizer.encode_plus(
        sentence1, sentence2, 
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        output = xtr_model_roberta(**inputs.to(xtr_model_roberta.device))
    probs = torch.nn.functional.softmax(output.logits, dim=-1)
    relevance_probability = probs[0, 1].item()  # Probability of 'relevant'
    return relevance_probability