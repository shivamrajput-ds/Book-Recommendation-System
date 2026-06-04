"""
recommender.py
--------------
Pure ML logic — no Streamlit, no FastAPI here.
Uses CSV + npy format (version-independent, works on any Python/pandas).
"""

import os
import numpy as np
import pandas as pd

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")


def _load_artifacts():
    pivot_path = os.path.join(ARTIFACT_DIR, "book_pivot.csv")
    sim_path   = os.path.join(ARTIFACT_DIR, "similarity.npy")

    if not os.path.exists(pivot_path) or not os.path.exists(sim_path):
        raise FileNotFoundError(
            "Artifacts missing. Run:  python src/train.py"
        )

    pivot      = pd.read_csv(pivot_path, index_col=0)
    similarity = np.load(sim_path)

    return pivot, similarity


# Load once at import time (cached by Python's module system)
_pivot, _similarity = _load_artifacts()


def get_book_list() -> list:
    """Return all books available for collaborative filtering."""
    return _pivot.index.tolist()


def recommend(book_name: str, n: int = 5) -> list:
    """
    Return top-n similar books for a given book title.
    Returns a list of dicts: [{"title": ..., "score": ...}, ...]
    """
    if book_name not in _pivot.index:
        return []

    idx       = np.where(_pivot.index == book_name)[0][0]
    distances = _similarity[idx]

    similar_items = sorted(
        enumerate(distances), key=lambda x: x[1], reverse=True
    )[1 : n + 1]

    return [
        {"title": _pivot.index[i], "score": round(float(s), 4)}
        for i, s in similar_items
    ]


def get_top_books() -> pd.DataFrame:
    """Load and return the pre-computed top-20 books dataframe."""
    path = os.path.join(ARTIFACT_DIR, "top_books.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Artifacts missing. Run:  python src/train.py")
    return pd.read_csv(path)
