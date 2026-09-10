"""
Builds a FAISS flat L2 index from the BRepNet part embeddings.

IndexFlatL2 stores vectors as-is and computes exact Euclidean distance
against every stored vector per query — no approximation. Sufficient at
this scale (500 vectors); FAISS's approximate index types exist for
million-scale datasets and are not needed here.
"""

import numpy as np
import faiss

EMBEDDINGS_PATH = "../outputs/embeddings.npy"
INDEX_OUT_PATH = "../outputs/faiss_index.bin"

embeddings = np.load(EMBEDDINGS_PATH).astype("float32")  # FAISS requires float32
print("Loaded embeddings:", embeddings.shape)

dimension = embeddings.shape[1]  # 84 = BRepNet per-face feature size
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("Vectors indexed:", index.ntotal)

# saved in FAISS's own file format instead of numpy's, 
# plus a tiny bit of metadata (how many vectors, what dimension) so FAISS knows how to read it back correctly. 
# It's a binary format (not human-readable text) because it's optimized for FAISS's internal math operations,
#  not for you to open and inspect.
faiss.write_index(index, INDEX_OUT_PATH)
print("Saved index to:", INDEX_OUT_PATH)