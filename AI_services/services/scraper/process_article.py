import random
import os
import requests

from services.scraper.file_formats.process_pdf import extract_pdf_content_from_response, preprocess_pdf_text
from services.scraper.file_formats.word import extract_word_document_content_from_response
from services.scraper.file_formats.text import extract_text_file_content_from_response
from services.scraper.file_formats.file import extract_article_content

def get_article_content(url: str) -> str:
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
    
    cite = "Citation Unavailable"

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
        content, cite = extract_article_content(url)

    return content, cite