from docx import Document
import io
import requests

def extract_word_document_content_from_response(response: requests.Response) -> str:
    with io.BytesIO(response.content) as f:
        return extract_word_document_content(f)
    
def extract_word_document_content(file_obj: io.BytesIO) -> str:
    doc = Document(file_obj)
    content = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    return content