"""
Queries the FAISS index for the nearest designs to a given part.

Given a design's filename (its manifest entry), looks up its precomputed
embedding, searches the index for the closest matches, and excludes the
query design itself from the results (distance 0 to itself is not useful).
"""

import json
import numpy as np
import faiss

EMBEDDINGS_PATH = "../outputs/embeddings.npy"
MANIFEST_PATH = "../outputs/embeddings_manifest.json"
INDEX_PATH = "../outputs/faiss_index.bin"

TOP_K = 6  # 5 real neighbors + the query itself, which gets filtered out


def load_resources():
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    manifest = json.load(open(MANIFEST_PATH))
    index = faiss.read_index(INDEX_PATH)
    return embeddings, manifest, index


def query_by_stem(query_stem, embeddings, manifest, index):
    query_idx = manifest.index(query_stem)
    query_vector = embeddings[query_idx : query_idx + 1]

    distances, indices = index.search(query_vector, TOP_K)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == query_idx:
            continue
        results.append((manifest[idx], float(dist)))

    return results[:5]


if __name__ == "__main__":
    embeddings, manifest, index = load_resources()

    query_stem = manifest[100]  # placeholder: querying the first design for now
    print("Query design:", query_stem)

    matches = query_by_stem(query_stem, embeddings, manifest, index)
    print("Nearest matches:")
    for stem, dist in matches:
        print(f"  {stem}  (distance: {dist:.4f})")