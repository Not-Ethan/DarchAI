from typing import List, Tuple
import re
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline

ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")


from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess_text(text: str) -> str:
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    return ' '.join(filtered_words)


def get_names_from_html(html_content: str, text_limit: int = 500) -> List[str]:
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove any script and style elements from the content
    for element in soup(["script", "style"]):
        element.extract()

    text = soup.get_text(" ")

    # Preprocess the text by filtering out stop words
    preprocessed_text = preprocess_text(text)

    # Limit the text to a specified number of characters for NER
    limited_text = preprocessed_text[:text_limit]

    # Use NER pipeline to get all names
    ner_results = ner_pipeline(limited_text)
    names = [result["word"] for result in ner_results if result["entity"] == "B-PER" or result["entity"] == "I-PER"]

    cleaned_names = []
    for name in names:
        cleaned_name = name.replace(" ##", "")
        cleaned_names.append(cleaned_name)

    return cleaned_names

def extract_article_content(url: str) -> Tuple[str, str]:
    article = Article(url)
    article.download()
    article.parse()

    plain_text = article.text
    html_names = get_names_from_html(article.html)

    print("NAMES: ", ", ".join(html_names))

    return plain_text, html_names

# Test the function with a sample URL
url = "https://www.rutgers.edu/news/nuclear-war-would-cause-global-famine-and-kill-billions-rutgers-led-study-finds"
plain_text, names = extract_article_content(url)
#print(f"Plain text: {plain_text}\n")
print(f"Names: {names}")
