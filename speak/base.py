import base64
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Self, Tuple

import numpy as np
from loguru import logger
from pydub import AudioSegment

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <magenta>{extra[harness]}</magenta> - <level>{message}</level>",
    level=os.environ.get("LOG_LEVEL", "ERROR").upper(),
)


class SpeechToSpeechJailbreakHarness(ABC):
    input_audio_sample_rate: Optional[int] = None
    audio_token_frame_rate: Optional[int] = None  # Hz of the input audio -> # tokens

    source_audio_signal: np.ndarray

    input_audio_signal: np.ndarray
    input_audio_base64: str
    input_audio: Optional[AudioSegment] = None

    transcription: Optional[str] = None
    output_audio_bytes: bytes = b""
    output_audio: Optional[AudioSegment] = None

    temperature: float

    # state
    is_ready: bool = False
    is_complete: bool = False

    input_audio_tokens: int
    input_transcription: Optional[str] = None

    output_audio_tokens: int
    output_transcription: Optional[str] = None

    # for latency analysis
    input_audio_sent_time: Optional[float] = None  # When we finish sending input audio
    first_output_token_time: Optional[float] = None  # When we receive first output token

    def __init__(
        self,
        source_audio_signal: np.ndarray,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
    ):
        self.input_audio_signal = self.source_audio_signal = source_audio_signal
        self.input_audio_bytes = np.clip(self.input_audio_signal * (2**15), -32768, 32767).astype(np.int16).tobytes()
        self.input_audio_base64 = base64.b64encode(self.input_audio_bytes).decode("utf-8")

        self.system_prompt = system_prompt
        self.temperature = temperature

        self.logger = logger.bind(
            harness=self.__class__.__name__.split("SpeechToSpeech")[0]
        )

        self.input_audio_tokens = -1
        self.output_audio_tokens = -1

        self.is_ready = False
        self.is_complete = False

        self.output_audio_bytes = b""
        self.output_transcription = ""
        self.input_transcription = ""

        # latency analysis fields
        self.input_audio_sent_time = None
        self.first_output_token_time = None

    @classmethod
    def from_file(
        cls,
        input_f: Path,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
    ) -> Self:
        try:
            audio = AudioSegment.from_file(input_f)
            pcm_audio = (
                audio.set_frame_rate(cls.input_audio_sample_rate)
                .set_channels(1)
                .set_sample_width(2)
            )

            samples = np.array(pcm_audio.get_array_of_samples())
            source_audio_signal = samples.astype(np.float32) / (2**15)
        except Exception as e:
            raise Exception("Cannot parse source audio.", e)

        return cls(source_audio_signal, system_prompt, temperature)

    @abstractmethod
    async def run(self) -> Tuple[Optional[str], str, AudioSegment]:
        """
        Kick off provider-specific control flow: setup, send input audio, receive output, and cleanup.

        Returns:
            input_transcription: Optional[str] - The transcription of the input audio.
            output_transcription: str - The transcription of the output audio.
            output_audio: AudioSegment - The output audio.
        """
