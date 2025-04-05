import os
import pandas as pd
from langchain_chroma import Chroma
from config import DATA_DIR, CHROMA_DIR

# Load books
books = pd.read_csv(os.path.join(DATA_DIR, 'book_with_emotions.csv'))
books['large_thumbnail'] = books['thumbnail'].fillna('') + "&fife=w800"
books['large_thumbnail'].replace('&fife=w800', 'data/cover-not_found.jpg', inplace=True)

# Load the already-built Chroma DB
db_books = Chroma(persist_directory=CHROMA_DIR)

def retrieve_semantic_recommendations(query: str, category: str = None, tone: str = None,
                                      initial_top_k: int = 50, final_top_k: int = 10) -> pd.DataFrame:
    """
    Performs semantic search on the vector database and filters recommendations.
    """
    records = db_books.similarity_search(query, k=initial_top_k)

    # Get the isbn for all the records
    books_list = [int(rec.page_content.strip('"').split()[i]) for rec in records]

    # Get the books based on the isbn value
    book_recs = books[books['isbn13'].isin(books_list)]

    # Filter the records based on the category
    if category and category != 'All':
        book_recs = book_recs[book_recs['simple_categories'] == category]
    book_recs = book_recs.head(final_top_k)

    # Sort the records based on the tone
    if tone and tone != 'All':
        if tone == "Happy":
            book_recs = book_recs.sort_values(by="joy", ascending=False)
        elif tone == "Surprising":
            book_recs = book_recs.sort_values(by="surprise", ascending=False)
        elif tone == "Angry":
            book_recs = book_recs.sort_values(by="anger", ascending=False)
        elif tone == "Suspenseful":
            book_recs = book_recs.sort_values(by="fear", ascending=False)
        elif tone == "Sad":
            book_recs = book_recs.sort_values(by="sadness", ascending=False)
    return book_recs


def recommend_books(query: str, category: str = "All", tone: str = "All"):
    """
    Returns a list of book recommendations based on the query, category, and tone.
    """
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row['description']
        authors_split = row['authors'].split(';')
        if len(authors_split) == 2:
            authors = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors = row['authors']
        caption = f"{row['title']} by {authors}: {description}"
        results.append((row['large_thumbnail'], caption))
    return results

def get_categories():
    cats = sorted(books["simple_categories"].unique())
    return ["All"] + cats

def get_tones():
    return ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
