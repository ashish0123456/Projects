from typing import List
import re
import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
_summarizer = pipeline("summarization", model="google/flan-t5-small", device=0 if device=="cuda" else -1)

def summarize_text(text: str, max_input_length: int = 512, max_summary_length: int = 150) -> str:
    """
    Summarize the given text using a transformer model.
    Splits long text into chunks if needed.
    Returns a single concatenated summary.
    """
    # split text into chunks by sentence count 
    sentences = text.split(". ")
    chunk_size = max_input_length // 2  # approx 2 tokens per sentence
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= chunk_size:
            chunks.append(". ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(". ".join(current_chunk))
    
    summaries = []
    for chunk in chunks:
        out = _summarizer(chunk, max_length=max_summary_length, do_sample=True, top_p=0.9)
        summaries.append(out[0]['summary_text'])
    return " ".join(summaries)

def extract_action_items_from_text(text: str) -> List[str]:
    """
    Lightweight heuristic: detect sentences with imperatives or keywords.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    actions = []
    action_keywords = ["please", "action", "todo", "to do", "follow up", "need to", "should", "must", "assign", "task", "deadline", "due by", "responsible", "follow-up"]

    for line in lines:
        low = line.lower()
        if any(kw in low for kw in action_keywords) or re.match(r"^(Let's|Let us|We should|We need to|You should|You need to)\b", line):
            actions.append(line)
    return actions