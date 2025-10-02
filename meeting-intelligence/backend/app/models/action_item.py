from typing import Optional 
from sqlmodel import SQLModel, Field

class ActionItem(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_id: Optional[int] = Field(default=None, foreign_key="meeting.id")
    segment_id: Optional[int] = Field(default=None, foreign_key="transcriptsegment.id")
    text: str
    assigned_to: str = None
    resolved: bool = Field(default=False)