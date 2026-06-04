# 📚 Book Recommendation System

A production-ready Book Recommendation System built using **Collaborative Filtering** and **Popularity-Based** scoring, deployed with **Streamlit** and **FastAPI**.

---

## 🔍 Overview

This project recommends books using two approaches:

| Approach | Description |
|---|---|
| **Popularity Based** | IMDb-style weighted rating — top 20 books ranked by score |
| **Collaborative Filtering** | Cosine similarity on user-book rating pivot table |

---

## 📁 Project Structure

```
Book-Recommendation-System/
│
├── data/                        # Raw CSVs (Books, Ratings, Users)
│   ├── Books.csv
│   ├── Ratings.csv
│   └── Users.csv
│
├── artifacts/                   # Auto-generated after training
│   ├── top_books.csv
│   ├── book_pivot.pkl
│   └── similarity.pkl
│
├── src/
│   ├── train.py                 # Data processing + model training
│   ├── recommender.py           # Core ML logic
│   └── google_books.py          # Google Books API integration
│
├── api/
│   └── main.py                  # FastAPI backend
│
├── streamlit_app.py             # Streamlit frontend
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Book-Recommendation-System.git
cd Book-Recommendation-System
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your Google Books API key
```

---

## 🚀 Run

### Step 1 — Train the model (run once)
```bash
python src/train.py
```
This generates all artifacts inside `artifacts/`.

### Step 2 — Start Streamlit app
```bash
streamlit run streamlit_app.py
```

### Step 3 — Start FastAPI (optional)
```bash
uvicorn api.main:app --reload
```
Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/top-books` | Top 20 books |
| GET | `/books` | All available book titles |
| GET | `/recommend/{book_name}` | 5 similar books |

---

## 📊 Dataset

[Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

- `Books.csv` — 271,360 books
- `Ratings.csv` — 1,149,780 ratings
- `Users.csv` — 278,858 users

**Collaborative Filtering filters:**
- Users with > 150 ratings
- Books with > 50 ratings (from active users)

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Pandas / NumPy** — Data processing
- **Scikit-learn** — Cosine similarity
- **Streamlit** — Frontend UI
- **FastAPI** — REST API backend
- **Google Books API** — Book summaries and genres

---

## 👤 Author

**Shivam** — CSE Student | ML & Data Science Portfolio Project
