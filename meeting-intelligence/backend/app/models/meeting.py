from typing import Optional
from sqlmodel import SQLModel, Field
from datetime import datetime

class Meeting(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    filename: str
    audio_path: str
    owner_id: Optional[int] = Field(default=None, foreign_key="user.id")
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="uploaded")  # e.g., uploaded, processing, completed
    duration_seconds: Optional[int] = None
