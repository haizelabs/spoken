import base64
import os
import sys
from abc import ABC, ABCMeta, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type

import numpy as np
from loguru import logger
from pydub import AudioSegment

if TYPE_CHECKING:
    from typing import Self

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <magenta>{extra[harness]}</magenta> - <level>{message}</level>",
    level=os.environ.get("LOG_LEVEL", "ERROR").upper(),
)

class SpeechToSpeechHarnessMeta(ABCMeta):
    _model_registry: Dict[str, Type['SpeechToSpeechHarness']] = {}

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)

        if 'Model' not in namespace or not issubclass(namespace['Model'], Enum):
            raise ValueError("Subclass of SpeechToSpeechHarness must expose available models in Model enum")

        for model in namespace['Model']:
            mcs._model_registry[model.value] = cls

        return cls

    def name_to_harness(model_name: str) -> Type['SpeechToSpeechHarness']:
        if model_name not in SpeechToSpeechHarness._model_registry:
            raise ValueError(f"Model {model_name} not found. Available models: {SpeechToSpeechHarness._model_registry.keys()}")
        return SpeechToSpeechHarness._model_registry[model_name]


class SpeechToSpeechHarness(ABC, metaclass=SpeechToSpeechHarnessMeta):
    class Model(Enum):
        pass

    # model-specific constants
    input_audio_sample_rate: Optional[int] = None # Hz
    output_audio_sample_rate: Optional[int] = None # Hz
    audio_token_frame_rate: Optional[int] = None  # Hz of the input audio -> # tokens

    source_audio_signal: np.ndarray

    input_audio_signal: np.ndarray
    input_audio_base64: str
    input_audio: Optional[AudioSegment] = None

    transcription: Optional[str] = None
    output_audio_bytes: bytes = b""
    output_audio: Optional[AudioSegment] = None

    model_name: str
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
        model: Model,
        source_audio_signal: np.ndarray,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
    ):
        self.model_name = model.value

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
        model: Model,
        input_f: Path,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
    ) -> 'SpeechToSpeechHarness':
        try:
            audio = AudioSegment.from_file(input_f)
            return cls.from_audio(model, audio, system_prompt, temperature)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception("Cannot parse source audio.", e)

    @classmethod
    def from_audio(
        cls,
        model: Model,
        input_audio: AudioSegment,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
    ) -> 'SpeechToSpeechHarness':
        try:
            pcm_audio = (
                input_audio.set_frame_rate(cls.input_audio_sample_rate)
                .set_channels(1)
                .set_sample_width(2)
            )

            samples = np.array(pcm_audio.get_array_of_samples())
            source_audio_signal = samples.astype(np.float32) / (2**15)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception("Cannot parse source audio.", e)

        return cls(model, source_audio_signal, system_prompt, temperature)

    @abstractmethod
    async def run(self) -> Tuple[Optional[str], str, AudioSegment]:
        """
        Kick off provider-specific control flow: setup, send input audio, receive output, and cleanup.

        Returns:
            input_transcription: Optional[str] - The transcription of the input audio.
            output_transcription: str - The transcription of the output audio.
            output_audio: AudioSegment - The output audio.
        """
