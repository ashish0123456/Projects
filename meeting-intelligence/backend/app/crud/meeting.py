from sqlmodel import Session, select
from app.models.meeting import Meeting
from app.models.transcript_segment import TranscriptSegment
from app.models.action_item import ActionItem
from app.models.meeting_summary import MeetingSummary
from datetime import datetime

def create_meeting(session: Session, filename: str, audio_path: str, title: str | None = None, owner_id: int | None = None) -> Meeting:
    meeting = Meeting(filename=filename, audio_path=audio_path, title=title, owner_id=owner_id, uploaded_at=datetime.utcnow(), status="uploaded")
    session.add(meeting)
    session.commit()
    session.refresh(meeting)
    return meeting

def get_meeting(session: Session, meeting_id: int) -> Meeting | None:
    return session.get(Meeting, meeting_id)

def update_meeting_status(session: Session, meeting_id: int, status: str) -> Meeting | None:
    meeting = session.get(Meeting, meeting_id)
    if not meeting:
        return None
    meeting.status = status
    session.add(meeting)
    session.commit()
    session.refresh(meeting)
    return meeting 

def add_transcript_segment(session: Session, meeting_id: int, start_time: float, end_time: float, text: str, speaker: str | None = None) -> TranscriptSegment:
    segment = TranscriptSegment(meeting_id=meeting_id, start_time=start_time, end_time=end_time, text=text, speaker=speaker)
    session.add(segment)
    session.commit()
    session.refresh(segment)
    return segment

def list_transcript_segments(session: Session, meeting_id: int) -> list[TranscriptSegment]:
    statement = select(TranscriptSegment).where(TranscriptSegment.meeting_id == meeting_id).order_by(TranscriptSegment.start_time)
    results = session.exec(statement).all()
    return results

def add_action_item(session: Session, meeting_id: int, text: str, segment_id: int | None = None, assigned_to: str | None = None) -> ActionItem:
    action_item = ActionItem(meeting_id=meeting_id, text=text, segment_id=segment_id, assigned_to=assigned_to)
    session.add(action_item)
    session.commit()
    session.refresh(action_item)
    return action_item

def list_action_items(session: Session, meeting_id: int) -> list[ActionItem]:
    statement = select(ActionItem).where(ActionItem.meeting_id == meeting_id)
    results = session.exec(statement).all()
    return results

def add_summary(session: Session, meeting_id: int, summary_text: str, model_name: str | None = None, prompt: str | None = None, summary_type: str | None = "abstractive", extra_meta: dict | None = None) -> MeetingSummary:
    summary = MeetingSummary(
        meeting_id=meeting_id,
        summary_text=summary_text,
        created_at=datetime.utcnow()
    )
    session.add(summary)
    session.commit()
    session.refresh(summary)
    return summary

def get_summary(session: Session, meeting_id: int) -> dict | None:
    statement = select(MeetingSummary).where(MeetingSummary.meeting_id == meeting_id).order_by(MeetingSummary.created_at.desc())
    result = session.exec(statement).first()
    return result