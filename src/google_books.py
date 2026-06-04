"""
google_books.py
---------------
Fetches book description and genre from Google Books API.
API key is read from .env  (GOOGLE_BOOKS_API_KEY).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


def get_book_info(book_name: str) -> tuple[str, str]:
    """
    Returns (description, genre) for a book title.
    Falls back gracefully if API fails or key is missing.
    """
    api_key = os.getenv("GOOGLE_BOOKS_API_KEY", "")

    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": book_name}
    if api_key:
        params["key"] = api_key

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return "Could not fetch description.", "Unknown"

    if "items" not in data or not data["items"]:
        return "No description available.", "Unknown"

    volume = data["items"][0].get("volumeInfo", {})
    description = volume.get("description", "No description available.")
    genre = volume.get("categories", ["Unknown"])[0]

    return description, genre
