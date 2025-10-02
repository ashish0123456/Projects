from typing import Optional
from sqlmodel import SQLModel, Field

class TranscriptSegment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: Optional[int] = Field(default=None, foreign_key="meeting.id")
    start_time: float  # in seconds
    end_time: float    # in seconds
    speaker: Optional[str] = None
    text: str
    created_at: str | None = None
    faiss_id: int | None = None

    