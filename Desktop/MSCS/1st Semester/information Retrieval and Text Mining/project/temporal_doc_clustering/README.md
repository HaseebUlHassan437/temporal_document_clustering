Temporal Document Clustering (Enron email demo)

Overview
-
This repository contains a Dash app and supporting scripts to explore temporal document clustering and retrieval over the Enron email corpus. The dashboard shows 2D cluster visualizations, ranked retrieval using TF‑IDF, document inspection, selection for RAG-style prompts, and an integrated Gemini query helper (API key required).

Quick Start (Windows PowerShell)
-
1. Create and activate a virtual environment (recommended Python 3.11):

```powershell
python -m venv venv
.\venv\Scripts\Activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. (Optional) If you want the original email bodies to appear in the UI, run the merge script to create `data/clustered_emails_raw.csv` (this will stream the large raw CSV):

```powershell
python scripts\merge_raw_into_clustered.py
```

4. Run the Dash app:

```powershell
python app.py
# then open http://127.0.0.1:8050 in your browser
```

Notes on environment and dependencies
-
- Recommended Python: 3.11 (pandas binary wheels are not always available for newer CPython versions; using 3.11 or a conda environment avoids build-from-source failures on Windows).
- If you prefer conda:

```powershell
conda create -n temporal-cluster python=3.11 -y; conda activate temporal-cluster
pip install -r requirements.txt
```

Gemini / LLM key
-
The app reads `GEMINI_API_KEY` from the environment (via `python-dotenv` if you place a `.env` file). Create a `.env` with:

```
GEMINI_API_KEY=your_api_key_here
```

Data files and where to look
-
- `data/clustered_emails.csv` — clustered/processed emails used by the app.
- `data/clustered_emails_raw.csv` — (optional) merged file containing the original `message` bodies. The app will prefer `clustered_emails_raw.csv` if present.
- `data/processed_emails.csv` — preprocessed text used to compute TF‑IDF.

Key scripts and modules
-
- `app.py` — Dash app and callbacks (main demo). Prefer `raw_body` when available for document display.
- `scripts/merge_raw_into_clustered.py` — stream-merge raw `emails.csv` into `clustered_emails_raw.csv` (handles large CSVs).
- `src/preprocessing.py` — preprocessing utilities and dataset processing.
- `src/clustering.py` — TF‑IDF, KMeans/T-SVD and PCA helper utilities.

Runtime caveats & recommendations
-
- TF‑IDF uses `sklearn.TfidfVectorizer` and the app currently uses PCA on a dense matrix for visualization; for larger corpora consider `TruncatedSVD` to work directly on sparse matrices.
- If `pip install -r requirements.txt` attempts to compile `pandas` from source on Windows, switch to Python 3.11 or use conda to get prebuilt binaries.

Git and pushing
-
This repo includes a `.gitignore` (see below). To push to GitHub (replace `<repo-url>`):

```powershell
git remote add origin <repo-url>
git branch -M main
git add README.md .gitignore
git commit -m "chore: add README and .gitignore"
git push -u origin main
```

Privacy / large files
-
- The repo by default excludes `data/` large files and `.env`. If you want to publish `data/clustered_emails_raw.csv`, consider Git LFS or provide a small sample instead.
- `project_deliverable.txt` and `propsal.txt` are ignored by default per your request (they will not be pushed).

Where to go next
-
- If you want, I can: (a) commit these files for you and prepare the exact `git` commands to run locally; (b) run additional cleanup (line-by-line comments in `app.py`); (c) implement model persistence with `joblib`.

License & credits
-
This work was prepared as part of a course project. Check course policy for sharing restrictions before publishing any dataset or private keys.
