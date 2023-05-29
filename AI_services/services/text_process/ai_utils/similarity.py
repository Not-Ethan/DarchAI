import re
from services.text_process.ai_utils.detection import contains_named_entities
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v2')

def weight(sentence, similarity):
    if re.search(r'(\d+)?(\.\d+)?( ?million| ?billion| ?trillion| ?percent| ?%)', sentence.text, flags=re.IGNORECASE):
        similarity *= 2
    elif re.search(r'\d|%', sentence.text):
        similarity *= 1.1
    elif re.search(r'because|since|so', sentence.text):
        similarity *= 1.25
    if contains_named_entities(sentence):
        similarity *= 1.25

    return similarity

def sbert_cosine_similarity(sentence1: str, sentence2: str) -> float:
    embedding1 = get_sentence_embedding_sbert(sentence1)
    embedding2 = get_sentence_embedding_sbert(sentence2)
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

def get_sentence_embedding_sbert(sentence: str):
    sentence_embedding = sbert_model.encode(sentence, convert_to_tensor=True)
    return sentence_embedding