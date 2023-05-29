import re

def is_informative(sentence):
    text = sentence.text.strip()
    token_count = len(sentence)

    if token_count <= 2:
        return False

    if re.match(r'^\s*(chapter|section|introduction|conclusion|acknowledgment|reference|table of contents)\s*$', text, flags=re.IGNORECASE):
        return False

    return True

def contains_named_entities(sentence):
    return any([token.ent_type_ for token in sentence])