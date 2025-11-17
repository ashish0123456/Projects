from celery import Celery
from app.core.config import settings
from app.db.session import engine
from sqlmodel import Session
from app.crud import meeting as crud_meeting
from app.services import audio as audio_svc
from app.services import asr as asr_svc
from app.services import diarize as diarize_svc
from app.services import summarize as summarize_svc
from app.services import embeddings_faiss as faiss_svc
from app.services import progress as progress_svc
import os
import tempfile
import logging
import numpy as np

logger = logging.getLogger(__name__)
celery = Celery("ingest_worker", broker=settings.REDIS_URL)

ASR_ENGINE = asr_svc.ASR(model_dir=os.path.join(os.getcwd(), "models"))

@celery.task(name="ingest_meeting")
def ingest_meeting_task(meeting_id: int):
    """"
    Ingestion stub task marks the meeting status to 'processing', 
    waits a bit to simulate work, then sets 'completed'.
    """
    logger.info("Starting ingestion for meeting ID: %s", meeting_id)
    progress_svc.publish_progress(meeting_id, "started", {
        "msg": "We received your meeting file and are preparing it for analysis."
    })

    try:
        with Session(engine) as session:
            meeting = crud_meeting.get_meeting(session, meeting_id)
            if not meeting:
                logger.error("Meeting ID %s not found", meeting_id)
                return {"error": "meeting not found"}
            
            # Update status to processing
            meeting = crud_meeting.update_meeting_status(session, meeting_id, "processing")
            progress_svc.publish_progress(meeting_id, "processing", {
                "msg": "Your meeting is in the queue and key services are spinning up."
            })
            audio_path = meeting.audio_path

        # ensure wav file and prepare single-channel 16k wav
        tmp_wav = os.path.join(tempfile.gettempdir(), f"meeting_{meeting_id}_converted.wav")
        audio_svc.load_audio_bytes_to_wavfile(audio_path, tmp_wav)
        progress_svc.publish_progress(meeting_id, "audio_converted", {
            "msg": "Audio cleaned and converted for the best transcription quality.",
            "path": tmp_wav
        })

        # VAD splitting into segments
        segments = audio_svc.split_on_speech(tmp_wav, min_duration_ms=500, max_duration_ms=60000)
        progress_svc.publish_progress(meeting_id, "vad_completed", {
            "msg": f"Detected {len(segments)} distinct speech sections.",
            "num_segments": len(segments)
        })

        # segments: list of (start_sec, end_sec, segment_wav_path)
        segment_texts = []
        segment_infos = []
        segment_paths = [seg[2] for seg in segments]

        # Transcribe each segment
        transcription = ASR_ENGINE.transcribe_batch(segment_paths)
        progress_svc.publish_progress(meeting_id, "transcription_completed", {
            "msg": f"Finished transcribing {len(transcription)} sections of your meeting.",
            "num_segments": len(transcription)
        })

        # Diarization (speaker clustering)
        try: 
            labels = diarize_svc.diarize_segments(segment_paths, min_speakers=1, max_speakers=6)
        except Exception as e:
            logger.exception("Diarization failed, falling back to single speaker: %s", e)
            labels = [0] * len(segment_paths)

        # Merge segments with same speaker if contiguous and persist transcript rows incrementally
        for (start, end, path), text, label in zip(segments, transcription, labels):
            speaker_label = f"Speaker {label+1}"
            # persist transcript segment
            with Session(engine) as session:
                seg_obj = crud_meeting.add_transcript_segment(session, meeting_id, float(start), float(end), text or "", speaker_label)
                segment_texts.append(text or "")
                segment_infos.append({"segment_id": seg_obj.id, "start": start, "end": end, "speaker": speaker_label})

        progress_svc.publish_progress(meeting_id, "segments_saved", {
            "msg": "Transcript organized with speaker labels for easy reading.",
            "num_segments_saved": len(segment_infos)
        })

        # Summarization
        full_text = "/n".join([seg for seg in segment_texts if seg])
        summary = summarize_svc.summarize_text(full_text, max_input_length=512, max_summary_length=150)
        actions = summarize_svc.extract_action_items_from_text(full_text)

        # persist summary and action items
        with Session(engine) as session:
            crud_meeting.add_summary(session, meeting_id, summary)

            for action in actions:
                crud_meeting.add_action_item(session, meeting_id, action, segment_id=None)
        
        progress_svc.publish_progress(meeting_id, "summarization_completed", {
            "msg": f"Summary crafted and {len(actions)} action item(s) captured.",
            "summary": summary,
            "num_action_items": len(actions)
        })

        # Build embeddings & FAISS index for semantic search
        texts_for_embeddings = []
        mapping = [] # mapping from index position to transcript segment id
        for info, txt in zip(segment_infos, segment_texts):
            if not txt or len(txt.strip()) == 0:
                continue
            texts_for_embeddings.append(txt)
            mapping.append(info["segment_id"])

        if texts_for_embeddings:
            embs = faiss_svc.embed_texts(texts_for_embeddings)
            embs_np = np.array(embs).astype("float32")
            dim = embs_np.shape[1]
            index_path, mapping_path = faiss_svc.create_faiss_index(meeting_id, dim, embs_np, mapping)
            progress_svc.publish_progress(meeting_id, "faiss_indexed", {
                "msg": "Smart search index is ready, find anything instantly.",
                "index_path": index_path,
                "mapping_path": mapping_path
            })
        else:
            progress_svc.publish_progress(meeting_id, "faiss_indexed", {
                "msg": "Search index skipped because no transcript text was detected.",
                "index_path": None,
                "mapping_path": None
            })

        with Session(engine) as session:
            crud_meeting.update_meeting_status(session, meeting_id, "completed")

        progress_svc.publish_progress(meeting_id, "completed", {
            "msg": "All insights are ready! Jump in to review transcripts, summary, and action items."
        })
        logger.info("Ingestion completed for meeting ID: %s", meeting_id)
        return {"status": "completed"}
    
    except Exception as e:
        logger.exception("Ingestion failed for meeting ID %s: %s", meeting_id, e)
        with Session(engine) as session:
            crud_meeting.update_meeting_status(session, meeting_id, "failed")
        progress_svc.publish_progress(meeting_id, "failed", {
            "error": str(e),
            "msg": "Something went wrong during processing. Please try again."
        })
        return {"status": "failed", "error": str(e)}