from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.loader import DataLoader
from src.models.hybrid import HybridRecommender

app = FastAPI(
    title="Movie Recommendation System API",
    description="Production-grade Hybrid Recommendation Engine (MovieLens 100K)",
    version="1.0.0"
)

hybrid_model = None


class MovieItem(BaseModel):
    movie_id: int = Field(..., example=50)
    title: str = Field(..., example="Star Wars (1977)")
    genres_text: str = Field(..., example="Action Adventure Romance Sci-Fi War")
    hybrid_score: float = Field(..., example=0.8542)


class RecommendationResponse(BaseModel):
    user_id: int
    count: int
    recommendations: List[MovieItem]


@app.on_event("startup")
def startup_event():
    global hybrid_model
    loader = DataLoader()
    ratings = loader.load_ratings()
    movies = loader.load_movies()

    hybrid_model = HybridRecommender(alpha=0.6, n_factors=20)
    hybrid_model.fit(ratings_df=ratings, movies_df=movies)


@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "ok", "service": "Movie Recommendation API"}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
def recommend(
    user_id: int,
    top_n: int = Query(default=10, ge=1, le=50)
):
    if hybrid_model is None:
        raise HTTPException(status_code=500, detail="Model is not initialized")

    try:
        recs_df = hybrid_model.recommend(user_id=user_id, top_n=top_n)
        records = recs_df.to_dict(orient="records")
        return RecommendationResponse(
            user_id=user_id,
            count=len(records),
            recommendations=records
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))