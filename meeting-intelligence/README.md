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