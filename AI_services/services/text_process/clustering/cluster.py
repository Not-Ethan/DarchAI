from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('paraphrase-distilroberta-base-v2')
from typing import List

def cluster_relevant_sentences(sentences: List[str], eps: float = 0.22, min_samples: int = 2, outlier_eps: float = 0.7):
    if not sentences:
        return [], []

    # Generate sentence embeddings using Universal Sentence Encoder
    embeddings = sbert_model.encode(sentences)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    dbscan.fit(embeddings)

    # Group sentence indices by cluster
    unique_labels = set(dbscan.labels_)
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    clustered_indices = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(dbscan.labels_):
        if label != -1:
            clustered_indices[label].append(idx)

    # Combine all sentences in each cluster
    representative_sentences = []
    for cluster in clustered_indices:
        cluster_sentences = [sentences[idx] for idx in cluster]
        combined_sentence = ' '.join(cluster_sentences)
        representative_sentences.append(combined_sentence)

    # Perform a second pass of DBSCAN on the outliers
    outlier_sentences = [sentences[i] for i, label in enumerate(dbscan.labels_) if label == -1]
    if outlier_sentences:
        outlier_embeddings = sbert_model.encode(outlier_sentences)
        outlier_dbscan = DBSCAN(eps=outlier_eps, min_samples=1, metric='cosine')
        outlier_dbscan.fit(outlier_embeddings)

        # Include clustered outlier sentences in the `clustered_indices` as separate clusters
        for idx, label in enumerate(outlier_dbscan.labels_):
            original_idx = sentences.index(outlier_sentences[idx])
            if label != -1:
                new_label = num_clusters + label
                if new_label < len(clustered_indices):
                    clustered_indices[new_label].append(original_idx)
                else:
                    clustered_indices.append([original_idx])
            else:
                new_cluster_index = len(clustered_indices)
                clustered_indices.append([original_idx])

        # Combine all outlier sentences in each cluster
        for cluster in clustered_indices[num_clusters:]:
            cluster_sentences = [sentences[idx] for idx in cluster]
            combined_sentence = ' '.join(cluster_sentences)
            representative_sentences.append(combined_sentence)

    return clustered_indices, representative_sentences