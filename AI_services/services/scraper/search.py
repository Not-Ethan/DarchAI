from googleapiclient.discovery import build

def search_articles(query: str, api_key: str, CSE: str, num_results: int=10):
    service = build("customsearch", "v1", developerKey=api_key)
    urls = []
    start_index = 1

    while len(urls) < num_results:
        response = service.cse().list(q=query, cx=CSE, start=start_index).execute()
        results = response.get('items', [])

        for result in results:
            url = result.get('link')
            if url:
                urls.append(url)

        # Handle pagination
        start_index += 10
        if start_index > 100 or not response.get('queries').get('nextPage'):
            break

    return urls[:num_results]

