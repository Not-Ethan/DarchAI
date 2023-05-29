from newspaper import Article
from typing import List
import re
from transformers import pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def extract_article_content(url: str) -> str:
    article = Article(url)
    article.download()
    article.parse()

    cleaned_authors = []

    if article.authors:
        cleaned_authors = [extract_author_name(author) for author in article.authors]
        cleaned_authors = [author for author in cleaned_authors if author != ""]
        author_part = " & ".join(cleaned_authors)
    else:
        author_part = "Unknown Author"

    year_part = ("'"+article.publish_date.strftime("%Y")) if article.publish_date else "Unknown Year"
    citation = f"{author_part} {year_part}"

    if flag_for_manual_review(article.text, cleaned_authors):
        citation += " (manual review recommended)"

    return article.text, citation

def extract_author_name(author_info: str) -> str:
    ner_results = ner_pipeline(author_info)
    names = [result["word"] for result in ner_results if result["entity"] == "B-PER" or result["entity"] == "I-PER"]
    name = " ".join(names).replace(" ##", "")
    return name if name else ""

def flag_for_manual_review(text: str, authors: List[str]) -> bool:
    # Set a threshold for the number of authors that warrant manual review
    multiple_authors_threshold = 3

    if len(authors) >= multiple_authors_threshold:
        return True

    # Check if there are any other patterns indicating multiple authors
    multiple_authors_patterns = [
        r'\band\b',  # Look for the word "and" between names
        r',',  # Look for commas between names
    ]

    for pattern in multiple_authors_patterns:
        if re.search(pattern, text):
            return True

    return False