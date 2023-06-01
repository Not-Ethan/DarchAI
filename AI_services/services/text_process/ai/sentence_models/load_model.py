from services.text_process.ai_utils.similarity import sbert_cosine_similarity
from services.text_process.ai.sentence_models.model_loader.roberta_base import roberta_base
from services.text_process.ai.sentence_models.model_loader.deberta_base import deberta_base
from services.text_process.ai.sentence_models.model_loader.distilbert import distilbert

def load_model(sentence1: str, sentence2: str, sentence_model=0):
    if sentence_model == 0:
        return (distilbert(sentence1, sentence2), 0.75)
    if sentence_model == 1:
        return (roberta_base(sentence1, sentence2), 0.5)
    elif sentence_model == 2:
        return (sbert_cosine_similarity(sentence1, sentence2), 0.5)
    elif sentence_model == 3:
        return (deberta_base(sentence1, sentence2), 0.5)
    else:
        return (sbert_cosine_similarity(sentence1, sentence2), 0.5)