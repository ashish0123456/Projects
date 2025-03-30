from fastapi import APIRouter
from app.utils.semantic import recommend_books, get_categories, get_tones

router = APIRouter()

@router.get('/recommend/')
def recommend(query: str, category: str = 'All', tone: str = 'All'):
    """
    Returns a list of recommended books based on the search query, category, and tone.
    """
    results = recommend_books(query, category, tone)
    return {"recommendation": results}

@router.get('/metadata/')
def metadata():
    """
    Provides available categories and tones.
    """
    categories = get_categories()
    tones = get_tones()
    return {"categories": categories, "tones": tones}