import spacy
import sys
from services.text_process.ai_utils.detection import is_informative
from services.text_process.ai_utils.similarity import weight
from util.preprocess_text import preprocess_text
from typing import Tuple, List
sys.path.append('ai')
from services.text_process.ai.sentence_models.load_model import load_model
from services.text_process.ai_utils.build_rel import build

nlp = spacy.load('en_core_web_lg')

def find_relevant_sentences(text:str, query:str, context:int=4, similarity_threshold:float=0.75, sentence_model=0):

    if(len(text)>1000000):
        raise Exception("Text is too long")

    doc = nlp(text)
    relevant_sentences = []
    sentences = list(doc.sents)
    included_in_context = set()
    char_index = 0

    for i, sentence in enumerate(sentences):

        if not is_informative(sentence):
            char_index += len(sentence.text) + 1
            continue

        text = preprocess_text(sentence.text)
        if(len(text)==0):
            char_index += len(sentence.text) + 1
            continue

        similarity, similarity_threshold = load_model(query, text, sentence_model=sentence_model)

        similarity = weight(sentence, similarity)

        build(relevant_sentences, sentences, included_in_context, char_index, i, sentence, context, similarity_threshold, similarity)

        char_index += len(sentence.text) + 1

    relevant_text = [sentence.text for sentence, _, _, _, _, _ in relevant_sentences]

    return relevant_sentences, relevant_text