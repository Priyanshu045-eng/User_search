from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

app = FastAPI(title="Smart User Search API")

# -----------------------------
# 🔹 User Model
# -----------------------------
class User(BaseModel):
    user_id: int
    name: str
    branch: str
    year: int
    interests: str

# -----------------------------
# 🔹 Search Users by Multiple Fields
# -----------------------------
@app.post("/smart_user_search/")
def smart_user_search(users: List[User], name: str, year: int = None, branch: str = None, top_n: int = 5):
    users_df = pd.DataFrame([u.dict() for u in users])

    # --- Step 1: Compute similarity for name ---
    name_vectorizer = TfidfVectorizer(stop_words="english")
    name_tfidf = name_vectorizer.fit_transform(users_df["name"])
    query_name_vec = name_vectorizer.transform([name])
    name_sim = cosine_similarity(query_name_vec, name_tfidf).flatten()

    # --- Step 2: Compute similarity for branch (if provided) ---
    if branch:
        branch_vectorizer = TfidfVectorizer(stop_words="english")
        branch_tfidf = branch_vectorizer.fit_transform(users_df["branch"])
        query_branch_vec = branch_vectorizer.transform([branch])
        branch_sim = cosine_similarity(query_branch_vec, branch_tfidf).flatten()
    else:
        branch_sim = np.zeros(len(users_df))

    # --- Step 3: Compute similarity for interests ---
    interests_vectorizer = TfidfVectorizer(stop_words="english")
    interests_tfidf = interests_vectorizer.fit_transform(users_df["interests"])
    query_interests_vec = interests_vectorizer.transform([" ".join([name, branch or "", str(year or "")])])
    interests_sim = cosine_similarity(query_interests_vec, interests_tfidf).flatten()

    # --- Step 4: Combine similarities with weights ---
    combined_score = (
        0.5 * name_sim +      # Name is most important
        0.2 * branch_sim +    # Then branch
        0.2 * interests_sim + # Then interests
        0.1 * (users_df["year"] == year).astype(float)  # Small boost if year matches
    )

    # --- Step 5: Sort and select top N users ---
    top_indices = combined_score.argsort()[::-1][:top_n]

    results = [
        {
            "user_id": int(users_df.iloc[i]["user_id"]),
            "name": users_df.iloc[i]["name"],
            "branch": users_df.iloc[i]["branch"],
            "year": int(users_df.iloc[i]["year"]),
            "interests": users_df.iloc[i]["interests"],
            "similarity_score": round(float(combined_score[i]), 3)
        }
        for i in top_indices if combined_score[i] > 0
    ]

    if not results:
        return {"message": f"No user found similar to '{name}'"}

    return {
        "searched_query": {"name": name, "branch": branch, "year": year},
        "matched_users": results
    }

# Example root
@app.get("/")
def home():
    return {"message": "Smart User Search API is running 🚀"}
