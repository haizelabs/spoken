import asyncio
import base64
import io
import json
import os
from enum import Enum
from typing import Optional, Tuple
import time

import numpy as np
import websockets
from pydub import AudioSegment

from spoken.base import SpeechToSpeechHarness


class OpenAISpeechToSpeechHarness(SpeechToSpeechHarness):
    """
    OpenAI Realtime Audio (https://platform.openai.com/docs/guides/realtime)
    - gpt-4o-realtime-preview-2024-12-17
    - gpt-4o-mini-audio-preview-2024-12-17
    """

    class Model(Enum):
        GPT_4O_REALTIME_PREVIEW_2024_12_17 = "gpt-4o-realtime-preview-2024-12-17"
        GPT_4O_MINI_AUDIO_PREVIEW_2024_12_17 = "gpt-4o-mini-audio-preview-2024-12-17"

    input_audio_sample_rate: Optional[int] = 24000
    output_audio_sample_rate: Optional[int] = 24000
    audio_token_frame_rate: Optional[int] = 10  # 100ms per token

    def __init__(
        self,
        model: Model,
        source_audio_signal: np.ndarray,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
    ):
        super().__init__(model, source_audio_signal, system_prompt, temperature)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.realtime_base_url = f"wss://api.openai.com/v1/realtime?model={self.model_name}"
        self.realtime_headers = [
            ("Authorization", f"Bearer {api_key}"),
            ("OpenAI-Beta", "realtime=v1"),
        ]

    async def run(self) -> Tuple[Optional[str], str, AudioSegment]:
        async with websockets.connect(self.realtime_base_url, additional_headers=self.realtime_headers) as websocket:
            setup_message = {
                "type": "session.update",
                "session": {
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "temperature": self.temperature,
                    "turn_detection": None,
                    "speed": 1.5,
                }
            }
            await websocket.send(json.dumps(setup_message))
            self.logger.debug("Sending setup message: {}", json.dumps(setup_message, indent=2))

            while not self.is_ready:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    self._process_message(message)
                except asyncio.TimeoutError:
                    continue

            self.logger.debug("Sending audio input")
            await websocket.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "audio": self.input_audio_base64}
                    ],
                },
            }))

            kwargs = {}
            if self.system_prompt:
                kwargs["instructions"] = self.system_prompt
            await websocket.send(json.dumps({"type": "response.create", "response": kwargs}))
            self.input_audio_sent_time = time.perf_counter()
            self.logger.info("Input audio sent at: {}", self.input_audio_sent_time)

            while not self.is_complete:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    self._process_message(message)
                except asyncio.TimeoutError:
                    continue

        self.output_audio = AudioSegment.from_raw(
            io.BytesIO(self.output_audio_bytes),
            sample_width=2,
            frame_rate=self.output_audio_sample_rate,
            channels=1,
        )

        # OpenAI doesn't do input transcription
        return None, self.output_transcription, self.output_audio

    def _process_message(self, message: str):
        data = json.loads(message)
        self.logger.debug("Received event: {}", json.dumps(data, indent=2))
        if data.get("type") == "session.created":
            self.logger.info("Session created")
            self.is_ready = True
        elif data.get("type") == "error":
            self.logger.error("Error: {}", json.dumps(data, indent=2))
        elif data.get("type") == "response.done":
            self.output_transcription = data["response"]["output"][0]["content"][0]["transcript"]
            self.is_complete = True
            self.logger.info("Transcription: {}", self.output_transcription)
            self.input_audio_tokens = data["response"]["usage"]["input_token_details"]["audio_tokens"]
            self.output_audio_tokens = data["response"]["usage"]["output_token_details"]["audio_tokens"]
            self.logger.info("Input tokens: {}", self.input_audio_tokens)
            self.logger.info("Output tokens: {}", self.output_audio_tokens)
        elif data.get("type") == "response.audio.delta":
            if self.first_output_token_time is None:
                self.first_output_token_time = time.perf_counter()
                self.logger.info("First output token received at: {}", self.first_output_token_time)

            delta_b64 = data["delta"]
            delta_bytes = base64.b64decode(delta_b64)
            self.output_audio_bytes += delta_bytes

if __name__ == "__main__":
    from pathlib import Path

    # os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

    harness = OpenAISpeechToSpeechHarness.from_file(
        #OpenAISpeechToSpeechHarness.Model.GPT_4O_REALTIME_PREVIEW_2024_12_17,
        OpenAISpeechToSpeechHarness.ModelGPT_4O_MINI_AUDIO_PREVIEW_2024_12_17,
        Path("./examples/scooby.wav")
    )

    input_transcription, output_transcription, output_audio = asyncio.run(harness.run())
    print(f"Input transcription: {input_transcription}")
    print(f"Output transcription: {output_transcription}")
    output_audio.export("./openai.wav", format="wav")
