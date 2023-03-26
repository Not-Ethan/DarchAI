import os
import time
import re
import requests
from bs4 import BeautifulSoup
import spacy
import torch
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
from newspaper import Article
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build
import PyPDF4
import io
import requests
from docx import Document
from docx import Document
from docx.shared import Pt
import pdfplumber


startTime = time.time();

api_key = os.environ.get('API_KEY');
CSE = os.environ.get('CSE');

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
    _, file_extension = os.path.splitext(url)
    file_extension = file_extension.lower()

    print(file_extension)
    # Download the file
    response = requests.get(url)
    response.raise_for_status()

    # If the file is a PDF
    if file_extension == '.pdf':
        content = extract_pdf_content_from_response(response)
        content = preprocess_pdf_text(content)

    # If the file is a Word document
    elif file_extension in ['.doc', '.docx']:
        content = extract_word_document_content_from_response(response)

    # If the file is a text file
    elif file_extension == '.txt':
        content = extract_text_file_content_from_response(response)

    # If the file is an RTF file
    elif file_extension == '.rtf':
        content = extract_word_document_content_from_response(response)

    # If the file is HTML or an unsupported type
    else:
        content = extract_article_content(url)

    return content


def extract_pdf_content_from_response(response):    
    with io.BytesIO(response.content) as f:
        return extract_pdf_content(f)

def extract_word_document_content_from_response(response):
    with io.BytesIO(response.content) as f:
        return extract_word_document_content(f)

def extract_text_file_content_from_response(response):
    return response.text
    
def extract_word_document_content(file_obj):
    doc = Document(file_obj)
    content = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    return content


def find_relevant_sentences(text, query, context=4):
    doc = nlp(text)
    query_doc = nlp(query)
    relevant_sentences = []
    sentences = list(doc.sents)
    included_in_context = set()
    
    for i, sentence in enumerate(sentences):
        text = preprocess_text(sentence.text)
        similarity = sbert_cosine_similarity(text, query)
        print(i, sentence.text, f"Similarity: {similarity:.2f}")
        i+=1
        if re.search(r'(\d+)?(\.\d+)?( ?million| ?billion| ?trillion| ?percent| ?%)',sentence.text, flags=re.IGNORECASE):
            similarity *= 2
        elif re.search(r'\d|%', sentence.text):
            similarity *= 1.5 
        elif re.search(r'because|since|so', sentence.text):
            similarity *= 1.25
        

    if similarity > 0.5:
            if i not in included_in_context:
                start_index = max(0, i - context)
                end_index = min(len(sentences), i + context + 1)
                before_context = sentences[start_index:i]
                after_context = sentences[i + 1:end_index]

                # Add the sentences from before_context and after_context to the set
                for j in range(start_index, end_index):
                    if j != i:
                        included_in_context.add(j)

                relevant_sentences.append((sentence, True, before_context, after_context))
            else:
                # If the sentence is already part of the context, mark it as relevant without changing the context
                for j, (rel_sentence, is_relevant, before, after) in enumerate(relevant_sentences):
                    if rel_sentence == sentence:
                        relevant_sentences[j] = (rel_sentence, True, before, after)
                        break

    print(i, sentence.text, f"Similarity: {similarity:.2f}")

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

#Generate query from topic side and argument provided
def generate_query(topic, side, argument):
    side_keywords = {
        'pro': ['advantages', 'benefits', 'positive aspects', 'strengths'],
        'con': ['disadvantages', 'drawbacks', 'negative aspects', 'weaknesses']
    }
    
    # Combine topic, side, and argument with relevant keywords
    query = f"{topic} {' '.join(side_keywords[side])} {argument} {argument} {argument}"
    
    return query

def generate_search_query(topic, side, argument):
    query = f"{argument}"
    return query

#extract text from pdf
def extract_pdf_content(file_obj):
     # Read the PDF file
    with pdfplumber.open(file_obj) as pdf:
        # Extract text from each page and combine it
        content = ""
        for page in pdf.pages:
            content += page.extract_text()

    return content

def search_articles(query, api_key, CSE, num_results=10):
    service = build("customsearch", "v1", developerKey=api_key)
    urls = []
    start_index = 1

    while len(urls) < num_results:
        response = service.cse().list(q=query, cx=CSE, start=start_index).execute()
        results = response.get('items', [])

        for result in results:
            url = result.get('link')
            if url:
                urls.append(url)

        # Handle pagination
        start_index += 10
        if start_index > 10 or not response.get('queries').get('nextPage'):
            break

    return urls[:num_results]

import re

def preprocess_pdf_text(text):
    # Remove non-alphanumeric characters, except for spaces, periods, and commas
    cleaned_text = re.sub(r"[^a-zA-Z0-9,. ]", " ", text)

    # Remove multiple spaces
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)

    # Remove multiple periods
    cleaned_text = re.sub(r"\.+", ".", cleaned_text)

    # Remove multiple commas
    cleaned_text = re.sub(r",+", ",", cleaned_text)

    # Remove spaces before commas and periods
    cleaned_text = re.sub(r"\s+([,.])", r"\1", cleaned_text)

    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)

    # Remove extra spaces at the beginning and end of the text
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def main(topic, side, argument, num_results=10):
    query = generate_query(topic, side, argument)
    processed_query = preprocess_text(query)
    sQuery = generate_search_query(topic, side, argument)
    processed_sQuery = preprocess_text(sQuery)

    urls = search_articles(processed_sQuery, api_key, CSE, num_results=num_results)
    url_sentence_map = {}
    
    print(urls)

    for url in urls:
        try:
            content = get_article_content(url)
            if content:
                relevant_sentences = find_relevant_sentences(content, processed_query)
                if relevant_sentences:
                    url_sentence_map[url] = relevant_sentences
        except Exception as e:
            print(f"Error processing URL {url}: {e}")

    return url_sentence_map

def apply_style(paragraph, font_size, bold=False, underline=False):
    run = paragraph.runs[0]
    run.font.size = Pt(font_size)
    run.bold = bold
    run.underline = underline

def write_sentences_to_word_doc(file_path, url_sentence_map):
    doc = Document()

    for url, relevant_sentences in url_sentence_map.items():
        doc.add_paragraph(url)

        for i, (sentence, is_relevant, before_context, after_context) in enumerate(relevant_sentences):
            # Write before context
            for context_sentence in before_context:
                para = doc.add_paragraph(context_sentence)
                apply_style(para, font_size=6)

            # Write the relevant sentence
            para = doc.add_paragraph(sentence.text)
            apply_style(para, font_size=12, bold=True)

            # Write after context
            for context_sentence in after_context:
                para = doc.add_paragraph(context_sentence)
                apply_style(para, font_size=6)

            doc.add_paragraph("\n")

    doc.save(file_path)


topic = "The United States Federal Government should ban the collection of personal data through biometric recognition technology."
side = "con"
argument = "banks use biometrics"

url_sentence_map = main(topic, side, argument, 10)

print(url_sentence_map)

write_sentences_to_word_doc("output.docx", url_sentence_map)

timeElapsed = time.time() - startTime
print("\nTIME: "+str(timeElapsed))