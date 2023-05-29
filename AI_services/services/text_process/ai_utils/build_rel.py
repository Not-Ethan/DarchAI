from services.text_process.ai_utils.similarity import sbert_cosine_similarity

def build(relevant_sentences, sentences, included_in_context, char_index, i, sentence, context, similarity_threshold, similarity):
    if similarity > similarity_threshold:
        start_index_char = char_index
        end_index_char = char_index + len(sentence.text)
        if i not in included_in_context:
            start_index = max(0, i - context)
            end_index = min(len(sentences), i + context + 1)
            prev_context = [(sentences[j].text.strip(), sbert_cosine_similarity(sentences[j].text, sentence.text)) for j in range(start_index, i)]
            next_context = [(sentences[j].text.strip(), sbert_cosine_similarity(sentences[j].text, sentence.text)) for j in range(i + 1, end_index)]

            # Add the sentences from prev_context and next_context to the set
            for j in range(start_index, end_index):
                if j != i:
                    included_in_context.add(j)

            relevant_sentences.append((sentence, True, prev_context, next_context, start_index_char, end_index_char))
        else:
            # If the sentence is already part of the context, mark it as relevant without changing the context
            for j, (rel_sentence, is_relevant, prev, after, _, _) in enumerate(relevant_sentences):
                if rel_sentence == sentence:
                    relevant_sentences[j] = (rel_sentence, True, prev, after, start_index_char, end_index_char)
                    break