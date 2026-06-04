"""
api/main.py
-----------
FastAPI backend.

Run:
    uvicorn api.main:app --reload

Swagger UI:
    http://127.0.0.1:8000/docs
"""

import sys
import os

# Make sure src/ is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from src.recommender import recommend, get_top_books, get_book_list

app = FastAPI(
    title="Book Recommendation System",
    description="Collaborative Filtering + Popularity Based Recommendations",
    version="1.0.0",
)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Book Recommendation API is running"}


@app.get("/top-books", tags=["Popularity"])
def top_books():
    """Return the top 20 books based on weighted popularity score."""
    df = get_top_books()
    return df.to_dict(orient="records")


@app.get("/books", tags=["Collaborative"])
def list_books():
    """Return all book titles available for collaborative filtering."""
    return {"books": get_book_list()}


@app.get("/recommend/{book_name}", tags=["Collaborative"])
def get_recommendations(book_name: str, n: int = 5):
    """
    Return top-n similar books for a given title.

    - **book_name**: exact book title (URL-encoded if it has spaces)
    - **n**: number of recommendations (default 5)
    """
    results = recommend(book_name, n=n)
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Book '{book_name}' not found in the collaborative filtering dataset.",
        )
    return {"book": book_name, "recommendations": results}
