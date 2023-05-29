def generate_query(topic: str, side: str, argument: str) -> str:
    side_keywords = {
        'pro': ['advantages', 'benefits', 'positive aspects', 'strengths'],
        'con': ['disadvantages', 'drawbacks', 'negative aspects', 'weaknesses'],
        'sup': ['']
    }
    
    # Combine topic, side, and argument with relevant keywords
    side_kw_string = ' '.join(side_keywords[side])
    
    query = ""

    # Assign different weights to topic, side, and argument
    if not side == "sup": 
        query = f"{topic} {side_kw_string} {argument} {argument}"
    else:
        query = f"{argument}"
    return query
def generate_search_query(topic:str, side:str, argument:str) -> str: 
    query = f"{argument}"
    return query