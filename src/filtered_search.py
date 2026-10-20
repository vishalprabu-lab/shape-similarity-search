"""
Search the existing FAISS index with an embedding, optionally keeping only
matches whose size is close to the query's, ranked by embedding distance.
Returns an empty list if nothing passes the filter.

Works for stored parts (query_by_stem) and new parts (search with a vector).
Never writes to the index or embeddings.
"""

import csv
import json
import sys
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = ROOT / "outputs" / "embeddings.npy"
MANIFEST_PATH = ROOT / "outputs" / "embeddings_manifest.json"
INDEX_PATH = ROOT / "outputs" / "faiss_index.bin"
PROPERTIES_CSV = ROOT / "outputs" / "part_properties.csv"

TOP_K = 5
DIM_KEYS = ["dim1", "dim2", "dim3"]


def load_resources():
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    manifest = json.load(open(MANIFEST_PATH))
    index = faiss.read_index(str(INDEX_PATH))
    with open(PROPERTIES_CSV) as fh:
        props = {}
        for row in csv.DictReader(fh):
            stem = row.pop("stem")
            props[stem] = {k: float(v) for k, v in row.items()}
            props[stem]["num_faces"] = int(props[stem]["num_faces"])
    return embeddings, manifest, index, props


def within_tolerance(query_props, match_props, tolerance):
    """True if each sorted bounding-box dimension is within +/- tolerance of the query's."""
    return all(
        abs(match_props[k] - query_props[k]) <= tolerance * query_props[k] for k in DIM_KEYS
    )


def search(vector, query_props, index, manifest, props,
           size_filter=False, tolerance=0.2, exclude_stem=None):
    """Top 5 matches for an embedding vector (1 x 84 float32), with properties."""
    # Search the whole index, so the size filter has every part to choose from
    distances, indices = index.search(vector, index.ntotal)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        stem = manifest[idx]
        if stem == exclude_stem:
            continue
        if size_filter and not within_tolerance(query_props, props[stem], tolerance):
            continue
        results.append({"stem": stem, "distance": float(dist), **props[stem]})
    return results[:TOP_K]


def query_by_stem(stem, embeddings, manifest, index, props, size_filter=False, tolerance=0.2):
    vector = embeddings[manifest.index(stem)][None, :]
    return search(vector, props[stem], index, manifest, props,
                  size_filter, tolerance, exclude_stem=stem)


def print_table(results):
    print(f"{'stem':24} {'dist':>7} {'dim1':>8} {'dim2':>8} {'dim3':>8} {'volume':>10} {'area':>10} {'faces':>5}")
    for r in results:
        print(f"{r['stem']:24} {r['distance']:7.4f} {r['dim1']:8.1f} {r['dim2']:8.1f} {r['dim3']:8.1f} "
              f"{r['volume']:10.0f} {r['area']:10.0f} {r['num_faces']:5d}")


if __name__ == "__main__":
    embeddings, manifest, index, props = load_resources()
    stem = manifest[int(sys.argv[1]) if len(sys.argv) > 1 else 0]
    p = props[stem]
    print(f"Query: {stem}  dims {p['dim1']:.1f} x {p['dim2']:.1f} x {p['dim3']:.1f}  "
          f"vol {p['volume']:.0f}  area {p['area']:.0f}  faces {p['num_faces']}")
    print("\nNo size filter:")
    print_table(query_by_stem(stem, embeddings, manifest, index, props))
    print("\nWith size filter (+/-20%):")
    print_table(query_by_stem(stem, embeddings, manifest, index, props, size_filter=True))
