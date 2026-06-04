"""
train.py
---------
Run this ONCE to generate artifacts:
    python src/train.py

Outputs:
    artifacts/top_books.csv
    artifacts/book_pivot.pkl
    artifacts/similarity.pkl
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_data():
    print("📂 Loading CSVs ...")
    books   = pd.read_csv(os.path.join(DATA_DIR, "Books.csv"), low_memory=False)
    ratings = pd.read_csv(os.path.join(DATA_DIR, "Ratings.csv"))
    return books, ratings


def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    # Fill known-missing authors / publishers / image URLs
    fixes_author = {
        "0751352497": "DK",
        "9627982032": "Edinburgh Financial Publishing",
    }
    for isbn, author in fixes_author.items():
        books.loc[books["ISBN"] == isbn, "Book-Author"] = author

    fixes_publisher = {
        "193169656X": "Novelbooks Inc",
        "1931696993": "Novelbooks Inc",
    }
    for isbn, pub in fixes_publisher.items():
        books.loc[books["ISBN"] == isbn, "Publisher"] = pub

    fixes_img = {
        "078946697X": "https://images.amazon.com/images/P/078946697X.0",
        "2070426769": "https://images.amazon.com/images/P/2070426769.0",
        "0789466953": "https://images.amazon.com/images/P/0789466953.0",
    }
    for isbn, url in fixes_img.items():
        books.loc[books["ISBN"] == isbn, "Image-URL-L"] = url

    return books


def build_top_books(merged: pd.DataFrame) -> pd.DataFrame:
    """Popularity-based recommendation using IMDb weighted rating."""
    print("🏆 Building Top 20 Books ...")
    book_stats = merged.groupby("Book-Title").agg(
        avg_rating=("Book-Rating", "mean"),
        total_rating=("Book-Rating", "count"),
    )

    C = merged["Book-Rating"].mean()
    m = 150  # minimum ratings threshold

    def weighted_rating(row):
        R, v = row["avg_rating"], row["total_rating"]
        return (v / (v + m) * R) + (m / (v + m) * C)

    book_stats["score"] = book_stats.apply(weighted_rating, axis=1)
    book_stats = book_stats.sort_values("score", ascending=False).reset_index()

    top20 = book_stats.head(20)
    top20_detailed = merged[merged["Book-Title"].isin(top20["Book-Title"])]
    top20 = top20.merge(top20_detailed, on="Book-Title").drop_duplicates(
        subset=["Book-Title"]
    )
    top20 = top20[
        [
            "Book-Title",
            "Book-Author",
            "Year-Of-Publication",
            "Publisher",
            "Image-URL-M",
            "avg_rating",
            "total_rating",
            "score",
        ]
    ]
    return top20


def build_collaborative_artifacts(merged: pd.DataFrame):
    """Collaborative Filtering: pivot table + cosine similarity."""
    print("🤝 Building Collaborative Filtering artifacts ...")

    # Keep only users with > 150 ratings
    active_users = merged.groupby("User-ID")["Book-Rating"].count()
    active_users = active_users[active_users > 150].index
    exp_df = merged[merged["User-ID"].isin(active_users)]

    # Keep only books with > 50 ratings (among active users)
    popular_books = exp_df.groupby("Book-Title")["Book-Rating"].count()
    popular_books = popular_books[popular_books > 50].index
    filtered = exp_df[exp_df["Book-Title"].isin(popular_books)]

    # Pivot table
    pivot = pd.pivot_table(
        data=filtered,
        index="Book-Title",
        columns="User-ID",
        values="Book-Rating",
    ).fillna(0)

    # Cosine similarity
    similarity = cosine_similarity(pivot)

    return pivot, similarity


def save_artifacts(top20, pivot, similarity):
    print("💾 Saving artifacts ...")
    top20.to_csv(os.path.join(ARTIFACT_DIR, "top_books.csv"), index=False)

    # CSV + npy: version-independent (works on Python 3.12, 3.13, any pandas)
    pivot.to_csv(os.path.join(ARTIFACT_DIR, "book_pivot.csv"))
    np.save(os.path.join(ARTIFACT_DIR, "similarity.npy"), similarity)

    print(f"✅ top_books.csv   → {len(top20)} books")
    print(f"✅ book_pivot.csv  → {pivot.shape}")
    print(f"✅ similarity.npy  → {similarity.shape}")


def main():
    books, ratings = load_data()
    books = clean_books(books)
    merged = books.merge(ratings, on="ISBN")

    top20 = build_top_books(merged)
    pivot, similarity = build_collaborative_artifacts(merged)

    save_artifacts(top20, pivot, similarity)
    print("\n🎉 Training complete! Run: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
