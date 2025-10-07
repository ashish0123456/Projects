from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1 import meetings
from app.db.base import create_db_and_tables
from app.db.session import engine

app = FastAPI(title='Meeting Intelligence API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meetings.router, prefix="/api/v1/meetings", tags=["meetings"])

@app.on_event("startup")
def on_startup():
    # Create database tables
    create_db_and_tables(engine)

@app.get("/health")
def health_check():
    return {"status": "ok"}