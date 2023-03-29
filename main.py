import os
import time
from datetime import date
import re
import requests
from bs4 import BeautifulSoup
import spacy
import torch
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
from newspaper import Article
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build
import PyPDF4
import io
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_UNDERLINE
import pdfplumber
from citeproc import CitationStylesStyle, CitationStylesBibliography, Citation, CitationItem, formatter
from citeproc.source.json import CiteProcJSON
import random
import json

startTime = time.time();
times  = []
global weightedTimeTotal
weightedTimeTotal = 0
global urlTimeTotal
global totalUrls
urlTimeTotal = 0
totalUrls = 0

api_key = os.environ.get('API_KEY');
CSE = os.environ.get('CSE');

nlp = spacy.load('en_core_web_lg');
sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v2')

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

def generate_tagline(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=50, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True, top_p=0.9)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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

    user_agent_list = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) like Gecko',
        'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134'
    ]
    user_agent = random.choice(user_agent_list)

    headers = {'User-Agent': user_agent}

    # Download the file
    response = requests.get(url, headers=headers, timeout=30)
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


def find_relevant_sentences(text, query, context=4, similarity_threshold=0.5):
    start_time = time.time()
    global weightedTimeTotal

    if(len(text)>1000000):
        raise Exception("Text is too long");
    doc = nlp(text)
    query_doc = nlp(query)
    relevant_sentences = []
    sentences = list(doc.sents)
    included_in_context = set()
    for i, sentence in enumerate(sentences):
        text = preprocess_text(sentence.text)
        if(len(text)==0):
            continue

        similarity = sbert_cosine_similarity(text, query)



        if re.search(r'(\d+)?(\.\d+)?( ?million| ?billion| ?trillion| ?percent| ?%)', sentence.text, flags=re.IGNORECASE):
            similarity *= 2
        elif re.search(r'\d|%', sentence.text):
            similarity *= 1.5
        elif re.search(r'because|since|so', sentence.text):
            similarity *= 1.25

        if similarity > similarity_threshold:
            if i not in included_in_context:
                start_index = max(0, i - context)
                end_index = min(len(sentences), i + context + 1)
                prev_context = [(sentences[j].text.strip(), sbert_cosine_similarity(sentences[j].text, sentence.text)) for j in range(start_index, i)]
                next_context = [(sentences[j].text.strip(), sbert_cosine_similarity(sentences[j].text, sentence.text)) for j in range(i + 1, end_index)]

                # Add the sentences from prev_context and next_context to the set
                for j in range(start_index, end_index):
                    if j != i:
                        included_in_context.add(j)

                relevant_sentences.append((sentence, True, prev_context, next_context))
            else:
                # If the sentence is already part of the context, mark it as relevant without changing the context
                for j, (rel_sentence, is_relevant, prev, next) in enumerate(relevant_sentences):
                    if rel_sentence == sentence:
                        relevant_sentences[j] = (rel_sentence, True, prev, next)
                        break

    relevant_text = " ".join([sentence.text for sentence, _, _, _ in relevant_sentences])


    delta_t = time.time() - start_time;
    weightedTimeTotal += delta_t / len(sentences);
    return relevant_sentences, relevant_text


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
        'con': ['disadvantages', 'drawbacks', 'negative aspects', 'weaknesses'],
        'sup': ['supporting']
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
        if start_index > 100 or not response.get('queries').get('nextPage'):
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
    global totalUrls
    global urlTimeTotal
    if side == "sup":
        topic = " "
    query = generate_query(topic, side, argument)
    processed_query = preprocess_text(query)
    sQuery = generate_search_query(topic, side, argument)
    processed_sQuery = preprocess_text(sQuery)

    urls = search_articles(processed_sQuery, api_key, CSE, num_results=num_results)
    url_sentence_map = {}
    url_text_map = {}
    print(urls)
    for url in urls:
        try:

            urlTimeStart = time.time()
            content = get_article_content(url)
            urlDeltaTime = time.time() - urlTimeStart
            urlTimeTotal += urlDeltaTime
            totalUrls += 1

            if content:
                url_text_map[url] = content
        except Exception as e:
            print(f"Error fetching URL {url}: {e}")
    i = 0   
    for url, text in url_text_map.items():
        individualStart = time.time()
        try:
            relevant_sentences, relevant_text = find_relevant_sentences(text, query)
        except Exception as e:
            print(f"Error finding relevant content in {url}: {e}")
            continue
        deltaTime = time.time() - individualStart
        times.append(deltaTime)

        tagline = generate_tagline(relevant_text)
        url_sentence_map[url] = (tagline, relevant_sentences)
        print(f"Finished processing URL: {url}, Relevant Sentences: {len(relevant_sentences)}, URL Number: {i}")
        i+=1

    return url_sentence_map

def apply_style(paragraph, font_size, bold=False, underline=False):
    run = paragraph.runs[0]
    run.font.size = Pt(font_size)
    run.bold = bold
    run.underline = underline

def add_table_of_contents(doc, current_url_map):
    # Add "Table of Contents" title
    para = doc.add_paragraph("Table of Contents", style='Title')
    apply_style(para, font_size=16, bold=True)
    doc.add_paragraph("\n")

    # Add taglines to the table of contents
    for url, (tagline, relevant_sentences) in current_url_map.items():
        if len(relevant_sentences) > 0:
            para = doc.add_paragraph(tagline, style='Heading 1')
    
    doc.add_page_break()


def get_mla_citation(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else url

        reference = {
            'id': '1',
            'type': 'webpage',
            'title': title,
            'URL': url,
            'accessed': {'date-parts': [[date.fromtimestamp(time.time()).year, date.fromtimestamp(time.time()).month, date.fromtimestamp(time.time()).day]]}
        }

        bib_source = CiteProcJSON([reference])
        bib_style = CitationStylesStyle('modern-language-association', locale='en-US')
        bibliography = CitationStylesBibliography(bib_style, bib_source, formatter.plain)

        citation = Citation([CitationItem('1')])
        bibliography.register(citation)

        return bibliography.bibliography()[0]
    except Exception as e:
        print(f"Error generating MLA citation for URL {url}: {e}")
        return url

def write_sentences_to_word_doc(file_path, url_sentence_map, info):
    num_urls_per_doc = 100
    docNum = 0
    url_count = 0

    while url_count < len(url_sentence_map):
        current_url_map = dict(list(url_sentence_map.items())[url_count:url_count + num_urls_per_doc])
        doc = Document()
        (topic, side, argument) = info
        doc.add_paragraph("Topic: " + topic + "\nSide: " + side + "\n" + "Arg: " + argument, style='Title')
        doc.add_page_break()

        add_table_of_contents(doc, current_url_map)

        for url, (tagline, relevant_sentences) in current_url_map.items():
            if len(relevant_sentences) == 0:
                url_count += 1
                continue

            para = doc.add_paragraph("Tagline: " + tagline, style='Heading 1')
            apply_style_run(para.runs[0], font_size=14, bold=True, underline=True)
            doc.add_paragraph(url, style='Heading 2')

            url_para = doc.add_paragraph()
            for sentence, is_relevant, before_context, after_context in relevant_sentences:
                if is_relevant:
                    r = url_para.add_run(sentence.text.strip() + " ")
                    apply_style_run(r, font_size=12, bold=True)

                for context_sentence, context_similarity in before_context:
                    r = url_para.add_run(context_sentence.strip() + " ")
                    if context_similarity > 0.5:
                        apply_style_run(r, font_size=12, underline=True)
                    else:
                        apply_style_run(r, font_size=7)

                for context_sentence, context_similarity in after_context:
                    r = url_para.add_run(context_sentence.strip() + " ")
                    if context_similarity > 0.5:
                        apply_style_run(r, font_size=12, underline=True)
                    else:
                        apply_style_run(r, font_size=7)

            url_count += 1
            print("Finished writing URL: " + url)

        doc.save(file_path + str(docNum) + ".docx")
        docNum += 1

def apply_style_run(run, font_size=None, bold=False, underline=False):
    if font_size:
        run.font.size = Pt(font_size)
    if bold:
        run.bold = bold
    if underline:
        run.underline = WD_UNDERLINE.SINGLE

import json

def save_to_json(file_path, url_sentence_map, info):
    data = []
    (topic, side, argument) = info
    for url, (tagline, relevant_sentences) in url_sentence_map.items():
        if len(relevant_sentences) == 0:
            continue

        temp_data = {}
        for sentence, is_relevant, prev_context, next_context in relevant_sentences:
            sentence_data = {
                'url': url,
                'tagline': tagline,
                'relevant_sentence': sentence.text,
                'is_relevant': is_relevant,
                'prev_context': [{"text": t, "similarity": s} for t, s in prev_context],
                'next_context': [{"text": t, "similarity": s} for t, s in next_context],
                'topic': topic,
                'side': side,
                'argument': argument
            }
            temp_data['data'] = sentence_data
            temp_data['url'] = url
            data.append(sentence_data)

    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


topic = "Should not ban the collection of personal data through biometric recognition technology"
'''
arguments = [
    {
        "side": "pro",
        "argument": "Biometric technology is necessary for national security"
    },
    {
        "side": "pro",
        "argument": "Biometric technology is necessary for US hegemony"
    },
    {
        "side": "pro",
        "argument": "Biometric technology is necessary for banks"
    },
    {
        "side": "con",
        "argument": "Biometric techonolgy is not accurate"
    },
    {
        "side": "con",
        "argument": "Biometric technology allows identity theft"
    },
    {
        "side": "con",
        "argument": "Biometric tecohoogy used by authoritarian governments"
    },
    {
        "side": "con",
        "argument": "Biometric technology is necessary for the metaverse"
    },
]
'''

arguments = [
    {
        "side": "con",
        "argument": "Biometric technology is necessary for cloning"
    },
    {
        "side": "sup",
        "argument": "cloning technology powerful"
    },
    {
        "side": "sup",
        "argument": "china steal biometric data"
    },
    {
        "side": "sup",
        "argument": "china clones people"
    },
    {
        "side": "sup",
        "argument": "china clones joe biden"
    }
]    
for item in arguments:
    side = item["side"]
    argument = item["argument"]
    file_path = side+"_"+("_".join(argument.split(" ")[-3:]))
    url_sentence_map = main(topic, side, argument, 10)

    write_sentences_to_word_doc(file_path, url_sentence_map, (topic, side, argument))
    save_to_json(file_path+".json", url_sentence_map, (topic, side, argument))

timeElapsed = time.time() - startTime
print("\nTIME: "+str(timeElapsed))
print("AVERAGE TIME: "+str(sum(times)/len(times)));
print("Weighted time: " + str(weightedTimeTotal/len(times)))
print("Average query time: " + str(sum(urlTimeTotal)/len(totalUrls)))