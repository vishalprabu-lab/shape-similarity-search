"""
Part properties (size/geometry) for STEP files, using occwl (brepnet env).

get_part_properties() is the one shared function used for both the 500 stored
parts and any new uploaded part. Running this file writes
outputs/part_properties.csv, keyed by file stem like the embeddings manifest.
"""

import csv
import json
from pathlib import Path

from occwl.compound import Compound

ROOT = Path(__file__).resolve().parent.parent
SUBSET_DIR = ROOT / "data" / "processed" / "fusion360_subset"
MANIFEST_PATH = ROOT / "outputs" / "embeddings_manifest.json"
PROPERTIES_CSV = ROOT / "outputs" / "part_properties.csv"

FIELDS = ["stem", "dim1", "dim2", "dim3", "volume", "area", "num_faces"]


def get_part_properties(step_path):
    """Bounding-box dims (largest to smallest), volume, surface area, face count."""
    part = Compound.load_from_step(step_path)
    # exact_box is tight; box() pads curved surfaces
    dims = sorted(part.exact_box().diagonal(), reverse=True)
    return {
        "dim1": float(dims[0]),
        "dim2": float(dims[1]),
        "dim3": float(dims[2]),
        "volume": part.volume(),
        "area": part.area(),
        "num_faces": part.num_faces(),
    }


def main():
    stems = json.load(open(MANIFEST_PATH))
    rows = []
    for i, stem in enumerate(stems):
        row = {"stem": stem}
        try:
            row.update(get_part_properties(SUBSET_DIR / f"{stem}.step"))
        except Exception as e:
            print(f"FAILED {stem}: {e}")
            continue
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(stems)}")

    with open(PROPERTIES_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)}/{len(stems)} rows to {PROPERTIES_CSV}")


if __name__ == "__main__":
    main()
