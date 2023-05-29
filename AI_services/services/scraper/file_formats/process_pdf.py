import io
import re
import pdfplumber
import requests

def extract_pdf_content_from_response(response: requests.Response) -> str:    
    with io.BytesIO(response.content) as f:
        return extract_pdf_content(f)
    
def preprocess_pdf_text(text: str):
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

def extract_pdf_content(file_obj: io.BytesIO) -> str:
    def group_chars_by_x(characters, threshold=1.0):
        groups = []
        current_group = []
        prev_x = None
        for char in characters:
            if prev_x is not None and char["x0"] - prev_x > threshold:
                groups.append(current_group)
                current_group = []
            current_group.append(char)
            prev_x = char["x1"]
        groups.append(current_group)
        return groups

    with pdfplumber.open(file_obj) as pdf:
        text = []

        for page in pdf.pages:
            current_line = []
            current_y = -1

            for char in sorted(page.chars, key=lambda x: (x["top"], x["x0"])):
                if char["top"] != current_y:
                    if current_line:
                        words = group_chars_by_x(current_line)
                        text.append(" ".join("".join(c["text"] for c in word) for word in words))
                        current_line = []
                    current_y = char["top"]

                
                current_line.append(char)

            if current_line:
                words = group_chars_by_x(current_line)
                text.append(" ".join("".join(c["text"] for c in word) for word in words))

        return "\n".join(text)