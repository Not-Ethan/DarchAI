import os
import time
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from typing import Tuple, List, Dict
from collections import defaultdict
from threading import Lock
from services.scraper.web_query import generate_query, generate_search_query
from services.scraper.process_article import get_article_content
from services.scraper.search import search_articles
from util.preprocess_text import preprocess_text
from services.text_process.service import find_relevant_sentences
from services.text_process.clustering.cluster import cluster_relevant_sentences
from services.text_process.ai.taglines.load_model import load_model as tagline

# Global dictionary to store progress information for each request
progress = {}
progress_lock = Lock()

def update_progress(request_id, stage, num, outof):
    with progress_lock:
        progress[request_id] = {'stage': stage, 'progress': f"{num} / {outof}", 'as_num': num/outof, 'num': num, 'outof': outof}


startTime = time.time()
times  = []
global urlTimeTotal
global totalUrls
urlTimeTotal = 0
totalUrls = 0

class ContextSentence:
    def __init__(self, text: str, similarity: float):
        self.text = text
        self.similarity = similarity

class RelevantSentence:
    def __init__(self, sentence: str, is_relevant: bool, before_context: List[ContextSentence], after_context: List[ContextSentence]):
        self.text = sentence
        self.is_relevant = is_relevant
        self.before_context = before_context
        self.after_context = after_context

class Evidence:
    def __init__(self, url: str, tagline: str, relevant_sentences: List[RelevantSentence]):
        self.url = url
        self.tagline = tagline
        self.relevant_sentences = relevant_sentences

api_key = os.environ.get('API_KEY')
CSE = os.environ.get('CSE')

sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v2')
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def main(topic: str, side: str, argument: str, num_results: int = 10, request_id=None, sentence_model=0, tagline_model=0) -> Dict[str, List[Dict[str, List[RelevantSentence]]]]:
    url_sentence_map = defaultdict(list)
    raw_data = {}
    relevant_sentences:List[List[Tuple[str, bool, List[Tuple(str, float)], List[Tuple(str, float)]]]] = []
    resulting_sentences:List[Evidence] = []
    global totalUrls
    global urlTimeTotal
    if side == "sup":
        topic = " "
    query = generate_query(topic, side, argument)
    sQuery = generate_search_query(topic, side, argument)
    processed_sQuery = preprocess_text(sQuery)

    urls = search_articles(processed_sQuery, api_key, CSE, num_results=num_results)
    url_text_map = {}

    print(urls)
    total_urls = len(urls)

    #Threading for backend
    if request_id is not None:
        update_progress(request_id, "fetching", 0, total_urls)  # Initialize progress to 0

    curURL = 0
    for url in urls:
        try:
            #logging
            urlTimeStart = time.time()

            #extract url content
            content, cite = get_article_content(url)

            # Update progress percentage
            update_progress(request_id, "fetching", (curURL + 1), total_urls)

            #logging
            urlDeltaTime = time.time() - urlTimeStart
            urlTimeTotal += urlDeltaTime
            totalUrls += 1

            if content:
                url_text_map[url] = content, cite
        except Exception as e:
            print(f"Error fetching URL {url}: {e}")

    curURL = 0

    for url, (text, citation) in url_text_map.items():

        individualStart = time.time()
        try:
            if request_id is not None:

                # Update progress percentage
                update_progress(request_id, "processing", (curURL + 1), total_urls)

            res_sentence, relevant_text = find_relevant_sentences(text, query, sentence_model=sentence_model)
        except Exception as e:
            print(f"Error finding relevant content in {url}: {e}")
            continue
        deltaTime = time.time() - individualStart
        times.append(deltaTime)

        rel_sentences = {}

        for i, (sentence, is_relevant, prev, after, start, end) in enumerate(res_sentence):
            rel_sentences[i] = (sentence, is_relevant, prev, after, start, end)

        # Perform clustering on the relevant sentences
        clusters_indices, representative_sentences = cluster_relevant_sentences([sent.text for sent, _, _, _, _, _ in res_sentence])

        # Generate taglines for each cluster
        taglines = [tagline(rep_sentence, tagline_model) for rep_sentence in representative_sentences]

        # Store the information in the url_sentence_map, using the new structure
        for i in range(len(clusters_indices)):
            # Convert Span objects to strings for each sentence in the cluster
            true_sentence = [(rel_sentences[j][0].text, rel_sentences[j][1], rel_sentences[j][2], rel_sentences[j][3]) for j in clusters_indices[i]]

            url_sentence_map[url].append({
                'tagline': taglines[i],
                'relevant_sentences': true_sentence,
                'citation': citation
            })
        raw_data[url] = {'full_text': text, 'prompt': query}
        print(f"Finished processing URL: {url}, Relevant Sentences: {len(res_sentence)}, URL Number: {curURL}")
        curURL += 1


    del progress[request_id]  # Remove the request progress on completion
    return url_sentence_map, raw_data
