import requests
from bs4 import BeautifulSoup
import spacy
import torch
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
from newspaper import Article
from sentence_transformers import SentenceTransformer

nlp = spacy.load('en_core_web_lg');
sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v2')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def get_sentence_embedding_sbert(sentence):
    sentence_embedding = sbert_model.encode(sentence, convert_to_tensor=True)
    return sentence_embedding

def sbert_cosine_similarity(sentence1, sentence2):
    embedding1 = get_sentence_embedding_sbert(sentence1)
    embedding2 = get_sentence_embedding_sbert(sentence2)
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    sentence_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return sentence_embedding

def bert_cosine_similarity(sentence1, sentence2):
    embedding1 = get_sentence_embedding(sentence1)
    embedding2 = get_sentence_embedding(sentence2)
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

def extract_article_content(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def preprocess_text(text):
    # Create a spaCy doc object
    doc = nlp(text)
    
    # Preprocess tokens: lowercase, remove stopwords and punctuation, and lemmatize
    preprocessed_tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct
    ]
    
    # Join the preprocessed tokens to create the preprocessed text
    preprocessed_text = ' '.join(preprocessed_tokens)
    
    return preprocessed_text

def get_article_content(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    content = ' '.join([p.text for p in soup.find_all('p')])
    return content

def find_relevant_sentences(text ,query):
    doc = nlp(text)
    query_doc = nlp(query)
    relevant_sentences = []
    i = 0
    for sentence in doc.sents:
        text = preprocess_text(sentence.text)
        similarity = sbert_cosine_similarity(text, query)
        if similarity > 0.7:
            relevant_sentences.append(sentence.text)

        print(i, sentence.text, "Similiarity: ", similarity)
        i+=1
    
    return relevant_sentences

def generate_queries(topic, side, argument):
    side_keywords = {
        'pro': ['advantages', 'benefits', 'positive aspects', 'strengths'],
        'con': ['disadvantages', 'drawbacks', 'negative aspects', 'weaknesses']
    }

    # Combine topic, side, and argument with relevant keywords
    queries = []
    for keyword in side_keywords[side]:
        queries.append(f"{topic} {keyword} {argument}")
        queries.append(f"{topic} {argument} {keyword}")

    # Add queries that focus only on the argument
    queries.append(f"{topic} {argument}")
    queries.append(f"{argument} {topic}")

    return queries

def generate_query(topic, side, argument):
    side_keywords = {
        'pro': ['advantages', 'benefits', 'positive aspects', 'strengths'],
        'con': ['disadvantages', 'drawbacks', 'negative aspects', 'weaknesses']
    }
    
    # Combine topic, side, and argument with relevant keywords
    query = f"{topic} {' '.join(side_keywords[side])} {argument} {argument} {argument}"
    
    return query


def main(topic, side, argument, urls):
    query = generate_query(topic, side, argument)
    processed_query = preprocess_text(query)
    extracted_sentences = []
    
    for url in urls:
        content = get_article_content(url)
        if content:
            relevant_sentences = find_relevant_sentences(content, processed_query)
            extracted_sentences.extend(relevant_sentences)

    return extracted_sentences



topic = "Dogs are better than cats"
side = "pro"
argument = "loyalty"
urls = [
    "https://www.thesprucepets.com/reasons-dogs-are-better-than-cats-1118371"
]

output = main(topic, side, argument, urls)
print("\n".join(output))