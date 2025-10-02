# Meeting Intelligence (MVP) — Backend

This repo contains the backend scaffold of a Meeting Intelligence MVP:
- FastAPI backend with SQLModel (SQLAlchemy-based ORM)
- PostgreSQL for metadata
- Redis + Celery for background ingestion tasks
- FAISS / embeddings / ASR 

## Quick start (dev)

1. Copy `.env.example` to `.env` and update variables if needed.
2. Build and start Postgres + Redis + backend (dev mode):
   ```bash
   docker-compose up --build
