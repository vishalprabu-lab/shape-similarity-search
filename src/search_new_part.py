"""
Search the existing index with a STEP file that is not in it (brepnet env).

The embedding comes from the same BRepNet pipeline as run_batch_embeddings.py
(its evaluate_folder call and model/standardization paths are reused, and the
per-face embeddings are mean-pooled the same way). Everything runs in a
temporary folder; the index, embeddings and data folders are never modified.

Usage: python src/search_new_part.py path/to/part.step [--filter] [--tol 0.2]
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from filtered_search import ROOT, load_resources, print_table, search
from part_properties import get_part_properties

BREPNET_DIR = ROOT / "external" / "BRepNet"


def embed_step_file(step_path):
    """Pooled 84-dim BRepNet embedding for one STEP file, as a 1 x 84 float32 array."""
    step_path = Path(step_path).resolve()  # before the chdir below

    # BRepNet's code and its model/standardization paths are relative to its own folder
    sys.path.insert(0, str(BREPNET_DIR))
    old_cwd = os.getcwd()
    os.chdir(BREPNET_DIR)
    try:
        import run_batch_embeddings as rbe  # reused only for evaluate_folder and the model paths

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            shutil.copy(step_path, work_dir / step_path.name)
            rbe.evaluate_folder(work_dir, rbe.FEATURE_STANDARDIZATION, model=rbe.PRETRAINED_MODEL)

            embedding_files = list((work_dir / "temp_working" / "embeddings").glob("*.embeddings"))
            if len(embedding_files) != 1:
                raise RuntimeError("BRepNet produced no embedding (file could not be processed)")
            arr = np.loadtxt(embedding_files[0])
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr.mean(axis=0).astype("float32")[None, :]  # same pooling as run_batch_embeddings
    finally:
        os.chdir(old_cwd)
        sys.path.remove(str(BREPNET_DIR))


def search_new_part(step_path, index, manifest, props, size_filter=False, tolerance=0.2):
    """Returns (query properties, top 5 matches) for any STEP file."""
    query_props = get_part_properties(step_path)
    vector = embed_step_file(step_path)
    return query_props, search(vector, query_props, index, manifest, props, size_filter, tolerance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step_file")
    parser.add_argument("--filter", action="store_true", help="apply the size filter")
    parser.add_argument("--tol", type=float, default=0.2)
    args = parser.parse_args()

    _, manifest, index, props = load_resources()
    query_props, matches = search_new_part(args.step_file, index, manifest, props, args.filter, args.tol)
    print(f"Query: {args.step_file}")
    print("dims {dim1:.1f} x {dim2:.1f} x {dim3:.1f}  vol {volume:.0f}  area {area:.0f}  faces {num_faces}".format(**query_props))
    if matches:
        print_table(matches)
    else:
        print("No matches within tolerance")
