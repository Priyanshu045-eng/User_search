from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import motor.motor_asyncio
import numpy as np
from dotenv import load_dotenv
import os
from bson import ObjectId

load_dotenv()

app = FastAPI(title="Smart User Search API")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["auth-db"]
users_collection = db.users


class SearchUsersRequest(BaseModel):
    name: str
    top_n: int = 5


class UserSearchResponse(BaseModel):
    user_id: str
    name: str
    email: str
    branch: str
    year: str
    interests: List[str]
    similarity_score: float


async def fetch_all_users():
    users = await users_collection.find({}).to_list(length=1000)
    return users


@app.post("/smart_user_search/", response_model=List[UserSearchResponse])
async def smart_user_search(request: SearchUsersRequest):
    users = await fetch_all_users()
    if not users:
        return []

    names = [u["name"] for u in users]

    vectorizer = TfidfVectorizer(stop_words="english")
    name_vectors = vectorizer.fit_transform(names)
    query_vec = vectorizer.transform([request.name])

    similarity_scores = cosine_similarity(query_vec, name_vectors).flatten()
    top_indices = similarity_scores.argsort()[::-1][:request.top_n]

    results = [
        UserSearchResponse(
            user_id=str(users[i]["_id"]),
            name=users[i]["name"],
            email=users[i]["email"],
            branch=users[i].get("branch", ""),
            year=users[i].get("year", ""),
            interests=users[i].get("interests", []),
            similarity_score=round(float(similarity_scores[i]), 3),
        )
        for i in top_indices if similarity_scores[i] > 0
    ]

    return results


@app.get("/")
def home():
    return {"message": "Smart User Search API is running 🚀"}
