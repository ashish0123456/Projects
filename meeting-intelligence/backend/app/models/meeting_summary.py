from typing import Optional
from sqlmodel import SQLModel, Field
from datetime import datetime

class MeetingSummary(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: int = Field(foreign_key="meeting.id", index=True)
    summary_text: str
    created_at: datetime = Field(default_factory=datetime.utcnow)