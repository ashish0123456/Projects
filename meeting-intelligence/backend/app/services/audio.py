import os 
import tempfile
from typing import List, Tuple
from pydub import AudioSegment
import webrtcvad
import wave
import math
import io

def ensure_dir(path: str):
    """Ensure the directory path exists."""
    os.makedirs(path, exist_ok=True)

def load_audio_bytes_to_wavfile(audio_path: str, out_path: str) -> None:
    """
    Convert uploaded audio bytes (mp3/m4a/wav) into a standard wav file 
    accepted by downstream tooling: 16kHz, mono, 16-bit PCM.
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(out_path, format="wav")

def read_wave(path: str) -> Tuple[bytes, int]:
    """Read a .wav file and return (PCM audio data, sample rate)."""
    with wave.open(path, 'rb') as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        pcm_data = wf.readframes(frames)
        return pcm_data, sample_rate

def frame_generator(frame_duration_ms: int, audio: bytes, sample_rate: int):
    """
    Split raw PCM audio into frames of frame_duration_ms (e.g., 30 ms).
    Each frame = small slice of audio.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 2 bytes per sample (16-bit audio)
    offset = 0
    while offset + n <= len(audio):
        yield audio[offset:offset + n]
        offset += n

def collect_voiced_segments(
    audio: bytes,
    sample_rate: int,
    frame_duration_ms: int,
    vad: webrtcvad.Vad,
    tmp_dir: str
) -> List[Tuple[float, float, str]]:
    """
    Run VAD over audio and return list of speech segments
    with (start_sec, end_sec, segment_path).
    """
    frames = list(frame_generator(frame_duration_ms, audio, sample_rate))

    voiced_chunks = []
    voiced_frames = []
    frame_index = 0

    for frame in frames:
        is_speech = vad.is_speech(frame, sample_rate)
        if is_speech:
            voiced_frames.append(frame)
        else:
            if voiced_frames:
                chunk_bytes = b"".join(voiced_frames)
                frames_in_chunk = int(len(chunk_bytes) / (2 * sample_rate) / (frame_duration_ms / 1000.0))
                start_sec = max(0.0, frame_index - frames_in_chunk) * (frame_duration_ms / 1000.0)
                end_sec = start_sec + frames_in_chunk * (frame_duration_ms / 1000.0)

                wav_out = os.path.join(tmp_dir, f"seg_{len(voiced_chunks)}.wav")
                with wave.open(wav_out, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(chunk_bytes)

                voiced_chunks.append((start_sec, end_sec, wav_out))
                voiced_frames = []
        frame_index += 1

    # Handle leftover speech at end
    if voiced_frames:
        chunk_bytes = b"".join(voiced_frames)
        frames_in_chunk = int(len(chunk_bytes) / (2 * sample_rate) / (frame_duration_ms / 1000.0))
        start_sec = max(0.0, frame_index - frames_in_chunk) * (frame_duration_ms / 1000.0)
        end_sec = start_sec + frames_in_chunk * (frame_duration_ms / 1000.0)

        wav_out = os.path.join(tmp_dir, f"seg_{len(voiced_chunks)}.wav")
        with wave.open(wav_out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(chunk_bytes)

        voiced_chunks.append((start_sec, end_sec, wav_out))

    return voiced_chunks

def split_on_speech(wav_path: str, min_duration_ms=500, max_duration_ms=60000) -> List[Tuple[float, float, str]]:
    """
    Split wav file into speech segments using WebRTC VAD.
    Returns: list of (start_sec, end_sec, segment_path).
    """
    audio, sample_rate = read_wave(wav_path)
    vad = webrtcvad.Vad(2)  # aggressiveness: 0=loose, 3=aggressive
    frame_duration = 30  # ms

    tmp_dir = tempfile.mkdtemp(prefix=f"segments_{os.path.basename(wav_path)}_")
    ensure_dir(tmp_dir)

    # Collect voiced segments with VAD
    voiced_chunks = collect_voiced_segments(audio, sample_rate, frame_duration, vad, tmp_dir)

    # further split long segments > max_duration_ms and filter out short segments < min_duration_ms
    final_segments = []
    for start_sec, end_sec, path in voiced_chunks:
        segment_duration_ms = (end_sec - start_sec) * 1000
        if segment_duration_ms <= max_duration_ms:
            if segment_duration_ms >= min_duration_ms:
                final_segments.append((start_sec, end_sec, path))
        else:
            num_splits = math.ceil(segment_duration_ms / max_duration_ms)
            split_duration = (max_duration_ms / 1000.0)
            for i in range(num_splits):
                split_start = start_sec + i * split_duration
                split_end = min(end_sec, split_start + split_duration)
                if (split_end - split_start) * 1000 >= min_duration_ms:
                    # create subsegment by reexporting with pydub
                    audio = AudioSegment.from_wav(path)
                    seg_audio = audio[int((split_start - start_sec) * 1000): int((split_end - start_sec) * 1000)]
                    out_path = os.path.join(tmp_dir, f"{os.path.basename(path)}_part{i}.wav")
                    seg_audio.export(out_path, format="wav")
                    final_segments.append((split_start, split_end, out_path))

    return final_segments