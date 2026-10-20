# Shape Similarity Search

GenAI/geometric-ML proof of concept: given a CAD part (STEP file), retrieve the most structurally similar existing designs from a repository, using a deep learning embedding of B-Rep geometry and vector similarity search.

Query is by **example part**, not text. There is no text-to-shape capability in this pipeline.

## How it works

```
STEP file → occwl (parses B-Rep topology) → BRepNet (generates embedding) → FAISS (nearest-neighbor search) → ranked list of similar designs
```

- **occwl** ([AutodeskAILab/occwl](https://github.com/AutodeskAILab/occwl), v3.0.0) parses STEP files via OpenCascade and extracts B-Rep topology (faces, edges, how they connect).
- **BRepNet** ([AutodeskAILab/BRepNet](https://github.com/AutodeskAILab/BRepNet)) is a pretrained model that operates directly on B-Rep topology (topological message passing / custom convolution, not a transformer). We use its pretrained segmentation model and pull the per-face hidden representation from the layer before the final classification head, mean-pooled into one 84-dim vector per part.
- **FAISS** (`IndexFlatL2`) indexes these vectors and returns nearest neighbors by Euclidean distance. Flat/brute-force is intentional — sufficient at 500 vectors, no need for approximate search yet.

**What "similar" means here**: similar *feature composition* (mix of extrudes, fillets, chamfers, etc.), not necessarily similar dimensions or overall proportions. A thin rod and a thick block both made of a single plain extrude will match well. Confirmed via manual visual spot-checks (see Known Issues).

## Capabilities

- **Search by an existing part** or **by a new STEP file** that isn't in the index (embedded on the fly with the same BRepNet pipeline; the index is never modified).
- **Part properties next to every match**: bounding-box size (largest to smallest, so orientation doesn't matter), volume, surface area, face count. Computed with occwl by one shared function for both indexed and uploaded parts.
- **Optional size filter**: keeps only matches whose three sorted bounding-box dimensions are each within ±N% (default 20%) of the query's. The whole index is searched, survivors are ranked by embedding distance, top 5 returned. Returns nothing if no part fits.
- **Minimal Streamlit app** combining all of the above, with thumbnails and distances.

Launch it from the project root in the `brepnet` conda env: `streamlit run app.py` (see Usage, step 7).

## Project status

- [x] Data pipeline: filtered subset built from Fusion 360 Gallery Reconstruction Dataset
- [x] Embedding generation: BRepNet running end-to-end, patched for modern dependencies
- [x] Retrieval: FAISS index + query script working, spot-checked visually
- [x] Part properties + optional size filter
- [x] Search with a new (non-indexed) STEP file
- [x] Demo UI (Streamlit)
- [ ] Reproducibility docs: `environment.yml` for the BRepNet environment, `PATCHES.md` documenting applied fixes, un-nesting the git repos under `external/`
- [ ] Root-causing weaker match quality on geometrically complex parts (see Known Issues)

Related: a second project (manufacturability/machining-feature screening) shares this same BRepNet backbone and is planned as a parallel effort.

## Project structure

```
shape-similarity-search/
├── app.py                  # Streamlit demo (brepnet env)
├── src/                    # Core pipeline code
│   ├── inventory_data.py       # Filters raw dataset down to the 500-design working subset
│   ├── build_faiss_index.py    # Builds the FAISS index from embeddings.npy
│   ├── query_similar_designs.py # Basic query: given a design, return nearest matches (.venv)
│   ├── part_properties.py      # Size/volume/area/face count per part; builds part_properties.csv (brepnet)
│   ├── filtered_search.py      # Search with properties + optional size filter (brepnet)
│   └── search_new_part.py      # Embed a new STEP file and search the index (brepnet)
├── notebooks/               # Exploratory / diagnostic scripts, not part of the pipeline
│   ├── check_build_sequence.py
│   └── inspect_retrieval.ipynb  # Visual query tool: inspect query + top matches inline
├── data/
│   ├── raw/                    # Untouched downloaded datasets (gitignored)
│   └── processed/               # Filtered 500-design subset + thumbnails
├── outputs/                 # Generated artifacts (gitignored)
│   ├── embeddings.npy           # 500 x 84 array, one row per design
│   ├── embeddings_manifest.json # Row index -> filename mapping for embeddings.npy
│   ├── faiss_index.bin
│   └── part_properties.csv      # Size/geometry properties, keyed by file stem
├── external/                # Third-party code, not ours (currently improperly nested
│   ├── BRepNet/                # git repos - see Known Issues)
│   └── occwl/
├── tests/                   # Placeholder, no formal tests yet
├── requirements.txt          # Main .venv dependencies
└── README.md
```

## Environments

This project uses **two separate, isolated environments** — they are not interchangeable and serve different parts of the pipeline.

### `.venv` — main pipeline (data prep, indexing, basic retrieval)

```
python -m venv .venv
.venv\Scripts\activate.bat      # Windows cmd
pip install -r requirements.txt
```

Core dependency: `cadquery` (OCP bindings to OpenCascade) — used for STEP handling in `src/`.

### `brepnet` (conda) — embedding generation, properties, search app

Required because `occwl` depends on `pythonocc-core`, which has no working pip wheel on Windows and is only reliably available via conda-forge. Embedding generation runs from `external/BRepNet/`. The properties, new-part search and Streamlit app also run here (occwl computes the properties; `cadquery` is deliberately not installed in this env to avoid an OpenCascade conflict). The basic scripts in `src/` (`build_faiss_index.py`, `query_similar_designs.py`) still use `.venv`.

```
conda create -n brepnet python=3.10 -c conda-forge -y
conda activate brepnet
conda install -c conda-forge pythonocc-core=7.7.0 igl -y
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytorch-lightning==2.6.5 pyDeprecate
pip install faiss-cpu streamlit
cd external/occwl
pip install .
```

BRepNet's own `environment.yml` (Python 3.7, PyTorch 1.7.1, pytorch-lightning 1.5.10) is stale — its `occwl` submodule reference also points to an internal Autodesk server and is not usable externally. The versions above are the confirmed-working modern substitute; BRepNet's code was patched to match (full patch list pending in `PATCHES.md` — see Known Issues).

## Usage

1. **Build the dataset subset** (`.venv`):
   ```
   cd src
   python inventory_data.py
   ```
   Filters to final-stage designs only (3-token filenames), randomly samples 500 with a fixed seed (42) for reproducibility, copies STEP files + thumbnails into `data/processed/fusion360_subset/`.

2. **Generate embeddings** (`brepnet` conda env, from `external/BRepNet/`):
   ```
   python run_batch_embeddings.py
   ```
   Reads all STEP files in `data/processed/fusion360_subset/`, runs BRepNet inference, writes `outputs/embeddings.npy` and `outputs/embeddings_manifest.json`. Takes several minutes on CPU.

3. **Build the FAISS index** (`.venv`):
   ```
   python build_faiss_index.py
   ```

4. **Query** (`.venv`):
   ```
   python query_similar_designs.py
   ```
   Or, for visual inspection with inline thumbnails, use `notebooks/inspect_retrieval.ipynb`.

5. **Build part properties** (`brepnet`, from the project root; already generated):
   ```
   python src/part_properties.py
   ```
   Writes `outputs/part_properties.csv`. Properties with/without the size filter for a stored part: `cd src && python filtered_search.py <manifest index>`.

6. **Search with a new STEP file** (`brepnet`, from `src/`):
   ```
   python search_new_part.py path/to/part.step [--filter] [--tol 0.2]
   ```
   Runs BRepNet in a temporary folder; nothing in `data/` or `outputs/` is touched.

7. **Run the app** (`brepnet`, from the project root):
   ```
   conda activate brepnet
   streamlit run app.py
   ```
   Pick an existing part or upload a STEP file, toggle the size filter and tolerance, and see the top 5 matches with thumbnail, distance and properties. Uploads show "No preview" (no 3D renderer).

## Data source

[Fusion 360 Gallery Reconstruction Dataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset) (Autodesk Research). **Non-commercial research use only** — this dataset cannot be used as-is for anything client-facing or commercial without separately checking licensing terms.

## Known issues

- **Match quality is inconsistent across part complexity.** Simple parts (e.g., plain extrusions) retrieve well-matched neighbors; more geometrically complex parts sometimes return visually dissimilar matches. Suspected cause: mean-pooling per-face embeddings may lose distinguishing detail on parts with many varied features. Not yet root-caused.
- **Embeddings barely encode size.** With the size filter on, the best size-compatible matches are often far away in embedding space (distances of ~20+ versus ~0.1–0.5 unfiltered). Distances are always shown so this stays visible. Distances are FAISS `IndexFlatL2` values (squared L2).
- **Part scale is not perfectly uniform.** Sizes look like consistent millimetres for most parts, but a few outliers are authored at extreme scales (e.g. `23856_6be75f62_0007`, ~4.2 million units), so the app shows sizes without a unit label.
- **A STEP file that loads but is malformed can still crash BRepNet's pipeline** (segfault in pythonocc), which would stop the app. Unreadable files are caught before BRepNet runs and shown as an error.
- **`run_batch_embeddings.py` still has hardcoded paths** (`SRC_DIR` points to `data/processed/test_subset`) and writes straight to `outputs/embeddings.npy`. Check them before re-running, or it will overwrite the embeddings. The new-part search only imports it for `evaluate_folder` and the model paths and never calls `main()`.
- **`external/BRepNet` and `external/occwl` are their own git repositories**, nested inside this project's repo. As-is, `git add .` from the project root likely does not correctly capture patches made inside them. Needs fixing (strip nested `.git` folders, track as plain files) before this is reliably reproducible by cloning.
- **`environment.yml` for the `brepnet` conda environment does not exist yet** — the recipe above is transcribed from manual setup and hasn't been captured as a file.