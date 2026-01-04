# Meeting Intelligence App

An **AI-powered meeting intelligence platform** that records, transcribes, summarizes, and extracts action items from meetings, all in real time.  
Built with a **React + FastAPI full-stack architecture**, it demonstrates **AI integration, clean backend design, and modern frontend development**.

---

## Tech Stack

### **Frontend**
- React (Vite + TypeScript)
- Tailwind CSS
- Axios for API communication
- Component-based modular structure

### **Backend**
- FastAPI (Python)
- LLM integration
- PostgreSQL (SQLAlchemy ORM)
- Celery + Redis (asynchronous task queue)
- Pydantic v2 for validation
- Dockerized for easy deployment

### **Infrastructure**
- Docker & Docker Compose
- PostgreSQL (database)
- Redis (cache & message broker)
- Nginx (reverse proxy for production)
- Multi-stage Docker builds for optimized images

---

## Core Features

| Category | Description |
|-----------|--------------|
| **Meeting Transcription** | Upload or stream meeting audio; speech-to-text transcription. |
| **Summarization** | LLM-powered summary generation (extractive & abstractive). |
| **Action Items Extraction** | Identifies tasks and responsibilities from meeting content. |
| **Speaker Diarization** | (Optional) Identifies different speakers in a conversation. |
| **Persistent Storage** | Stores meeting data, transcripts, and summaries in PostgreSQL. |
| **Dashboard** | Interactive frontend to view meeting summaries and insights. |
| **Async Processing** | Heavy LLM + transcription handled asynchronously via Celery workers. |

---

## Quick Start with Docker

### Local Development

```bash
# Start all services (PostgreSQL, Redis, FastAPI, Celery, React)
docker-compose up

# Access the application
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

### Production Deployment

```bash
# Configure environment variables
cp .env.example .env.prod
# Edit .env.prod with your production settings

# Start production stack with Nginx reverse proxy
docker-compose -f docker-compose.prod.yml up -d

# Access via: https://yourdomain.com
```

For detailed Docker deployment instructions, see [DOCKER_DEPLOYMENT.md](./DOCKER_DEPLOYMENT.md)

---

## Project Structure

```
meeting-intelligence/
├── backend/                    # FastAPI application
│   ├── app/
│   │   ├── api/               # API endpoints
│   │   ├── models/            # Database models
│   │   ├── schemas/           # Pydantic schemas
│   │   ├── services/          # Business logic
│   │   ├── tasks/             # Celery tasks
│   │   ├── crud/              # Database operations
│   │   ├── db/                # Database configuration
│   │   └── core/              # Configuration
│   ├── data/                  # Data storage
│   ├── requirements.txt        # Python dependencies
│   └── Dockerfile             # Backend container image
│
├── frontend/                   # React + Vite application
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/             # Page components
│   │   ├── api/               # API client
│   │   └── assets/            # Static assets
│   ├── package.json           # Node.js dependencies
│   └── Dockerfile             # Frontend container image
│
├── docker-compose.yml         # Local development compose (default)
├── docker-compose.local.yml   # Local development compose (explicit)
├── docker-compose.prod.yml    # Production compose with Nginx
├── nginx.conf                 # Nginx reverse proxy configuration
├── .env.example               # Example environment variables
└── DOCKER_DEPLOYMENT.md       # Detailed deployment guide
```