"""
streamlit_app.py
----------------
Run:
    streamlit run streamlit_app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from src.recommender import recommend, get_top_books, get_book_list
from src.google_books import get_book_info

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
)

# ── Light Styling ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background-color: #f5f5f5; }
    .book-card { text-align: center; padding: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Cache heavy loads ──────────────────────────────────────────────────────
@st.cache_data
def load_top_books():
    return get_top_books()

@st.cache_data
def load_book_list():
    return get_book_list()

# ── Title ──────────────────────────────────────────────────────────────────
st.title("📚 Book Recommendation System")
st.markdown("Powered by **Collaborative Filtering** + **Popularity Score**")

# ── Navigation ─────────────────────────────────────────────────────────────
option = st.radio("Choose an option:", ["🏆 Top 20 Books", "🔍 Find Similar Books"], horizontal=True)

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1 — TOP 20 BOOKS
# ══════════════════════════════════════════════════════════════════════════
if option == "🏆 Top 20 Books":
    st.subheader("Top 20 Books to Read")
    st.caption("Ranked by IMDb-style weighted rating (avg rating × popularity)")

    top_books = load_top_books()

    for i in range(0, len(top_books), 4):
        cols = st.columns(4)
        for j in range(4):
            idx = i + j
            if idx < len(top_books):
                book = top_books.iloc[idx]
                with cols[j]:
                    with st.container(border=True):
                        st.image(book["Image-URL-M"], width=130)
                        st.markdown(f"**{book['Book-Title']}**")
                        st.caption(f"✍️ {book['Book-Author']}")
                        st.caption(f"⭐ Avg: {book['avg_rating']:.2f}  |  🔥 Score: {book['score']:.2f}")

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2 — SEARCH & RECOMMEND
# ══════════════════════════════════════════════════════════════════════════
else:
    st.subheader("Find Books Similar to One You Like")

    book_list = load_book_list()

    selected_book = st.selectbox(
        "Start typing to search a book:",
        options=["-- Select a book --"] + book_list,
    )

    if selected_book and selected_book != "-- Select a book --":
        st.success(f"✅ Selected: **{selected_book}**")

        col1, col2 = st.columns([1, 2])

        # ── Left: Book details ─────────────────────────────────────────
        with col1:
            # Load top_books to get image/author for selected book if available
            top_books_df = load_top_books()
            match = top_books_df[top_books_df["Book-Title"] == selected_book]

            if not match.empty:
                book_row = match.iloc[0]
                st.image(book_row["Image-URL-M"], width=160)
                st.markdown(f"**Author:** {book_row['Book-Author']}")
            else:
                st.info("Cover not available in top-books dataset.")

        # ── Right: Summary & Genre via Google Books API ────────────────
        with col2:
            with st.spinner("Fetching book info ..."):
                description, genre = get_book_info(selected_book)

            st.markdown(f"**🎭 Genre:** {genre}")
            st.markdown("**📖 Summary:**")
            st.write(description)

        # ── Recommendations ────────────────────────────────────────────
        st.divider()
        st.subheader("📌 Similar Books You Might Like")

        recs = recommend(selected_book, n=5)

        if not recs:
            st.warning("No recommendations found for this book.")
        else:
            rec_cols = st.columns(5)
            for k, rec in enumerate(recs):
                with rec_cols[k]:
                    with st.container(border=True):
                        st.markdown(f"**{k+1}. {rec['title']}**")
                        st.caption(f"Similarity: {rec['score']:.2f}")
