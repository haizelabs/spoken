import inspect
import sys
from pathlib import Path
from typing import Optional, Union

from pydub import AudioSegment

from .base import SpeechToSpeechHarness, SpeechToSpeechHarnessMeta
from .models.gemini import GeminiSpeechToSpeechHarness
from .models.nova import NovaSpeechToSpeechHarness
from .models.openai import OpenAISpeechToSpeechHarness

def spoken(
        model_name: str,
        input_audio: Union[str, Path, AudioSegment],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None, # TODO: use the default for the model
    ) -> SpeechToSpeechHarness:
    cls = SpeechToSpeechHarnessMeta.name_to_harness(model_name)
    model = cls.Model(model_name)

    kwargs = dict(system_prompt=system_prompt)
    if temperature is not None:
        # fallback to base class default if not passed
        kwargs["temperature"] = temperature

    if isinstance(input_audio, str):
        input_audio = Path(input_audio)

    if isinstance(input_audio, Path):
        return cls.from_file(
            model,
                input_audio,
                **kwargs
            )
    elif isinstance(input_audio, AudioSegment):
        return cls.from_audio(
            model,
            input_audio,
            **kwargs
        )
    else:
        raise ValueError(f"Invalid input type: {type(input_audio)}. Must be a file or AudioSegment.")

class SpokenWrapper:
    def __call__(self, model_name: str, input_f: Union[str, Path], system_prompt: Optional[str] = None, temperature: Optional[float] = None) -> SpeechToSpeechHarness:
        return spoken(model_name, input_f, system_prompt, temperature)

    @property
    def models(self) -> list[str]:
        return list(SpeechToSpeechHarnessMeta._model_registry.keys())

sys.modules[__name__] = SpokenWrapper()
