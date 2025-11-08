from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import motor.motor_asyncio
import numpy as np
from dotenv import load_dotenv
import os
from bson import ObjectId

load_dotenv()

app = FastAPI(title="Smart User Search API (MongoDB-Compatible)")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["auth-db"]
users_collection = db.users

class SearchUsersRequest(BaseModel):
    name: str
    branch: Optional[str] = None
    year: Optional[str] = None
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
    branches = [u["branch"] for u in users]
    interests_list = [" ".join(u.get("interests", [])) for u in users]
    years = [u["year"] for u in users]

    name_vec = TfidfVectorizer(stop_words="english").fit_transform(names)
    query_name_vec = TfidfVectorizer(stop_words="english").fit(names).transform([request.name])
    name_sim = cosine_similarity(query_name_vec, name_vec).flatten()

    if request.branch:
        branch_vec = TfidfVectorizer(stop_words="english").fit_transform(branches)
        query_branch_vec = TfidfVectorizer(stop_words="english").fit(branches).transform([request.branch])
        branch_sim = cosine_similarity(query_branch_vec, branch_vec).flatten()
    else:
        branch_sim = np.zeros(len(users))

    interests_vec = TfidfVectorizer(stop_words="english").fit_transform(interests_list)
    query_interests_vec = TfidfVectorizer(stop_words="english").fit(interests_list).transform(
        [" ".join([request.name, request.branch or "", request.year or ""])]
    )
    interests_sim = cosine_similarity(query_interests_vec, interests_vec).flatten()

    year_match = np.array([1 if u_year == request.year else 0 for u_year in years])

    combined_score = (
        0.5 * name_sim +
        0.2 * branch_sim +
        0.2 * interests_sim +
        0.1 * year_match
    )

    top_indices = combined_score.argsort()[::-1][:request.top_n]

    results = [
    UserSearchResponse(
        user_id=str(users[i]["_id"]),
        name=users[i]["name"],
        email=users[i]["email"],         
        branch=users[i]["branch"],
        year=users[i]["year"],
        interests=users[i].get("interests", []),
        similarity_score=round(float(combined_score[i]), 3)
    )
    for i in top_indices if combined_score[i] > 0
]


    return results

@app.get("/")
def home():
    return {"message": "Smart User Search API is running 🚀"}

