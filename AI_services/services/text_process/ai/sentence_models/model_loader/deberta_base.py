from transformers import DebertaForSequenceClassification, DebertaTokenizer
import torch
from torch.nn.functional import softmax

deberta_model = DebertaForSequenceClassification.from_pretrained('services/text_process/ai/sentence_models/model_loader/models/deberta-base-mnli', num_labels=2)
deberta_tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base-mnli')

def deberta_base(argument, sentence):
    # Encode the argument and sentence
    inputs = deberta_tokenizer.encode_plus(
        argument,
        sentence,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Get the model's output
    with torch.no_grad():
        outputs = deberta_model(**inputs)

    # The output logits are in the first element of the outputs tuple
    logits = outputs[0]

    # Apply the softmax function to convert the logits into probabilities
    probs = softmax(logits, dim=-1)

    # The second element of the probs tensor is the probability that the argument supports the sentence
    support_prob = probs[0, 1].item()

    return support_prob