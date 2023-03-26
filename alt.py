import os
import spacy
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
import torch

# Load the BERT model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')

nlp = spacy.load('en_core_web_sm')

def load_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

def generate_query(topic, side, link, impact, uniqueness):
    return f"{topic} {side} {link} {impact} {uniqueness}"

def preprocess(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not (token.is_stop or token.is_punct or token.is_space)]
    return ' '.join(tokens)

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.detach().numpy().mean(axis=1)

def find_relevant_sentences(query, text, threshold=0.5):
    query = preprocess(query)
    query_embedding = get_bert_embedding(query)

    relevant_sentences = []
    doc = nlp(text)

    for sent in doc.sents:
        sent_text = preprocess(sent.text)
        sent_embedding = get_bert_embedding(sent_text)
        similarity = 1 - cosine(query_embedding, sent_embedding)

        if similarity >= threshold:
            relevant_sentences.append((sent.text, similarity))
    
    return sorted(relevant_sentences, key=lambda x: x[1], reverse=True)

# Load the text from a .txt file
file_path = 'text.txt'
text = load_text_file(file_path)

# Set up the debate topic, side, and argument components
topic = "dogs are better than cats"
side = "affirmative"
link = "dogs are more loyal"
internal_link = "dogs are more willing to protect people"
impact = "keeping people safe"
uniqueness = "dogs are more loyal and better at protecting people than cats"

# Generate the query
query = generate_query(topic, side, link, impact, uniqueness)

# Find relevant sentences
relevant_sentences = find_relevant_sentences(query, text)
for sentence, similarity in relevant_sentences:
    if(similarity > 0.8):
        print(f"{sentence} (Similarity: {similarity:.2f})")

