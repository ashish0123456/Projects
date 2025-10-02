import os
import logging
from typing import List, Tuple
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

DEFAULT_WGML_MODEL = "base.en"

class ASR:
    def __init__(self, model_name: str | None = None, device: str = "cpu", model_dir: str | None = None):
        self.device = device
        self.model_dir = model_dir or os.path.join(os.getcwd(), "models")
        self.model_name = model_name or os.getenv("ASR_MODEL", DEFAULT_WGML_MODEL)

        local_model_path = os.path.join(self.model_dir, self.model_name)
        if os.path.isdir(local_model_path):
            self.model_path = local_model_path  # local folder containing model.bin
        else:
            self.model_path = self.model_name  # treat as model size string -> auto download

        self._model = None
        
        try:
            logger.info(f"Loading faster-whisper model from: %s", self.model_path)
            self._model = WhisperModel(
                self.model_path, 
                device=self.device, 
                compute_type="int8" if self.device == "cpu" else "int8_float16")
        except Exception as e:
            logger.error(f"Failed to load faster-whisper model %s: %s", self.model_path, e)
            raise

    def transcribe_segment(self, wav_path: str) -> Tuple[str, float, float]:
        """
        Transcribes a single audio segment and returns the text along with start and end times. 
        """
        if not self._model:
            raise RuntimeError("ASR model is not loaded")
        
        segments, info = self._model.transcribe(wav_path, beam_size=5)
        transcript = " ".join([segment.text for segment in segments])
        return transcript
    
    def transcribe_batch(self, wav_paths: List[str]) -> List[Tuple[str, float, float]]:
        """
        Transcribes a batch of audio segments and returns a list of tuples (text, start_time, end_time).
        """
        if not self._model:
            raise RuntimeError("ASR model is not loaded")
        
        results = []
        for wav_path in wav_paths:
            try:
                transcript = self.transcribe_segment(wav_path)
            except Exception as e:
                logger.exception(f"Error transcribing %s: %s", wav_path, e)
                transcript = "" 
            results.append(transcript)
        return results