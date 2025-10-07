from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class MeetingCreate(BaseModel):
    title: Optional[str] = None

class MeetingRead(BaseModel):
    id: int
    title: Optional[str] = None
    filename: str
    audio_path: str
    status: str

class TranscriptSegmentRead(BaseModel):
    id: int
    meeting_id: int
    start_time: float
    end_time: float
    speaker: Optional[str] = None
    text: str

class ActionItemRead(BaseModel):
    id: int
    meeting_id: int
    segment_id: Optional[int] = None
    text: str
    assigned_to: Optional[str] = None
    resolved: bool

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SummaryRead(BaseModel):
    meeting_id: int
    summary_text: Optional[str] = None
    created_at: Optional[datetime] = None 