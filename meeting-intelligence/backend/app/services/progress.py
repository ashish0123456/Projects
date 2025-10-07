import redis
import json
from app.core.config import settings

_redis = None

def get_redis():    
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.REDIS_URL)
    return _redis

def publish_progress(meeting_id: int, stage: str, detail: dict | None = None):
    r = get_redis()
    message = {
        "meeting_id": meeting_id,
        "stage": stage,
        "detail": detail or {}
    }
    channel = f"meeting_progress_{meeting_id}"
    r.publish(channel, json.dumps(message))

    # store last progress for polling
    r.set(f"meeting_progress_last_{meeting_id}", json.dumps(message))

def get_last_progress(meeting_id: int):
    """Fetch last stored progress"""
    
    r = get_redis()
    data = r.get(f"meeting_progress_last_{meeting_id}")
    if not data:
        return {"stage": "pending", "detail": {}}
    return json.loads(data)