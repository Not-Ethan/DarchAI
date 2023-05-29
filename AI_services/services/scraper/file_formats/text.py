import requests

def extract_text_file_content_from_response(response: requests.Response) -> str:
    return response.text