import spacy

nlp = spacy.load('en_core_web_lg')

def preprocess_text(text: str) -> str:
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