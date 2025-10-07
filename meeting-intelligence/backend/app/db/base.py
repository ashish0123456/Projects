from sqlmodel import SQLModel
from app.models.user import User, UserBase
from app.models.meeting import Meeting
from app.models.transcript_segment import TranscriptSegment
from app.models.action_item import ActionItem

def create_db_and_tables(engine):
    SQLModel.metadata.create_all(engine)