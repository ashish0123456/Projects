from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from sqlmodel import Session
from app.db.session import get_session
from app.services.storage import save_upload
from app.crud import meeting as crud_meeting
from app.schemas.meeting import MeetingRead, TranscriptSegmentRead, ActionItemRead, SearchRequest, SummaryRead
from app.tasks.ingest import ingest_meeting_task

router = APIRouter()

# ------------------------------
# Upload a new meeting
# ------------------------------
@router.post("/upload/", response_model=MeetingRead)
async def upload_meeting(file: UploadFile = File(...), title: str = Form(...), session: Session = Depends(get_session)):
    contents = await file.read()
    path = save_upload(contents, file.filename)
    meeting = crud_meeting.create_meeting(session, filename=file.filename, audio_path=path, title=title)
    # enqueue background ingestion task
    ingest_meeting_task.delay(meeting.id)
    return MeetingRead(id=meeting.id, title=meeting.title, filename=meeting.filename, audio_path=meeting.audio_path, status=meeting.status)

# ------------------------------
# Get meeting info
# ------------------------------
@router.get("/{meeting_id}/", response_model=MeetingRead)
def get_meeting(meeting_id: int, session: Session = Depends(get_session)):
    meeting = crud_meeting.get_meeting(session, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return MeetingRead(id=meeting.id, title=meeting.title, filename=meeting.filename, audio_path=meeting.audio_path, status=meeting.status)

# ------------------------------
# Get transcripts
# ------------------------------
@router.get("/{meeting_id}/transcripts/", response_model=list[TranscriptSegmentRead])
def get_transcripts(meeting_id: int, session: Session = Depends(get_session)):
    segments = crud_meeting.list_transcript_segments(session, meeting_id)
    return [TranscriptSegmentRead(
        id=seg.id, 
        meeting_id=seg.meeting_id,
        start_time=seg.start_time, 
        end_time=seg.end_time, 
        speaker=seg.speaker, 
        text=seg.text
        ) for seg in segments]

# ------------------------------
# Get action items
# ------------------------------
@router.get("/{meeting_id}/action-items/", response_model=list[ActionItemRead])
def get_action_items(meeting_id: int, session: Session = Depends(get_session)):
    items = crud_meeting.list_action_items(session, meeting_id)
    return [ActionItemRead(
        id=item.id,
        meeting_id=item.meeting_id,
        segment_id=item.segment_id,
        text=item.text,
        assigned_to=item.assigned_to,
        resolved=item.resolved
        ) for item in items]

# ------------------------------
# Get LLM-generated summary
# ------------------------------
@router.get("/{meeting_id}/summary", response_model=SummaryRead)
def get_summary(meeting_id: int, session: Session = Depends(get_session)):
    meeting = crud_meeting.get_meeting(session, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")    
    summary = crud_meeting.get_summary(session, meeting_id)
    if not summary:
        return None
    return SummaryRead(**summary)

# ------------------------------
# Search transcript
# ------------------------------
@router.post("/{meeting_id}/search/")
def search_meeting(meeting_id: int, body: SearchRequest, session: Session = Depends(get_session)):
    from app.services.embeddings_faiss import embed_texts, query_index
    query_vec = embed_texts([body.query])[0]
    results = query_index(meeting_id, query_vec, top_k=body.top_k)

    # fetch corresponding segments
    hits = []
    from app.models.transcript_segment import TranscriptSegment
    for res in results:
        seg = session.get(TranscriptSegment, res['segment_id'])
        if not seg:
            continue
        hits.append({
            "segment_id": seg.id,
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "speaker_label": seg.speaker_label,
            "text": seg.text,
            "score": res['score']
        })
    return {"query": body.query, "results": hits}

# ------------------------------
# Get all the meeting info - transcript, action items and summary
# ------------------------------
@router.get("/{meeting_id}/details")
def get_meeting_details(meeting_id: int, session: Session = Depends(get_session)):
    meeting = crud_meeting.get_meeting(session, meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    segments = crud_meeting.list_transcript_segments(session, meeting_id)
    actions = crud_meeting.list_action_items(session, meeting_id)
    summary_obj = crud_meeting.get_summary(session, meeting_id)
    summary_text = summary_obj["summary_text"] if summary_obj else None
    return {
        "id": meeting.id,
        "title": meeting.title,
        "filename": meeting.filename,
        "audio_path": meeting.audio_path,
        "status": meeting.status,
        "transcript": [seg.text for seg in segments],
        "actions": [a.text for a in actions],
        "summary": summary_text,
    }