import asyncio
import base64
import io
import json
import os
import time
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import websockets
from pydub import AudioSegment

from spoken.base import SpeechToSpeechHarness


class GeminiSpeechToSpeechHarness(SpeechToSpeechHarness):
    """
    Gemini Multimodal Live (https://ai.google.dev/gemini-api/docs/live)

    Native audio (speech -> speech)
    - gemini-2.5-flash-preview-native-audio-dialog
    - gemini-2.5-flash-exp-native-audio-thinking-dialog
    """

    class Model(Enum):
        GEMINI_2_5_FLASH_PREVIEW_NATIVE_AUDIO_DIALOG = "gemini-2.5-flash-preview-native-audio-dialog"
        GEMINI_2_5_FLASH_EXP_NATIVE_AUDIO_THINKING_DIALOG = "gemini-2.5-flash-exp-native-audio-thinking-dialog"

    input_audio_sample_rate: Optional[int] = 16000
    output_audio_sample_rate: Optional[int] = 24000
    audio_token_frame_rate: Optional[int] = 25  # 40ms per token

    def __init__(
        self,
        model: Model,
        source_audio_signal: np.ndarray,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
    ):
        super().__init__(model, source_audio_signal, system_prompt, temperature)

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        self.realtime_base_url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={api_key}"

    async def run(self) -> Tuple[Optional[str], str, AudioSegment]:
        async with websockets.connect(
            self.realtime_base_url,
        ) as websocket:
            setup_message = {
                "setup": {
                    "model": f"models/{self.model_name}",
                    "generationConfig": {
                        "temperature": self.temperature,
                        "responseModalities": ["AUDIO"]
                    },
                    "realtimeInputConfig": {
                        "automaticActivityDetection": {
                            "disabled": True
                        }
                    },
                    "inputAudioTranscription": {},
                    "outputAudioTranscription": {}
                }
            }

            if self.system_prompt:
                setup_message["setup"]["systemInstruction"] = self.system_prompt

            self.logger.debug(
                "Sending setup message: {}", json.dumps(setup_message, indent=2)
            )
            await websocket.send(json.dumps(setup_message))

            while not self.is_ready:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    self._process_message(message)
                except asyncio.TimeoutError:
                    continue

            self.logger.debug("Sending audio input")
            await websocket.send(json.dumps({
                "realtimeInput": {
                    "activityStart": {}
                }
            }))

            audio_message = {
                "realtimeInput": {
                    "audio": {
                        "data": base64.urlsafe_b64encode(self.input_audio_bytes).decode('ascii'),
                        "mimeType": f"audio/pcm;rate={self.input_audio_sample_rate}"
                    },
                }
            }
            await websocket.send(json.dumps(audio_message))

            end_message = {
                "realtimeInput": {
                    "audioStreamEnd": True
                }
            }
            await websocket.send(json.dumps(end_message))

            self.input_audio_sent_time = time.perf_counter()
            self.logger.info("Input audio fully sent at: {}", self.input_audio_sent_time)

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

        return self.input_transcription, self.output_transcription, self.output_audio

    def _process_message(self, message: str):
        try:
            data = json.loads(message)
            self.logger.debug("Received event: {}", json.dumps(data, indent=2))

            if "setupComplete" in data:
                self.logger.info("Setup complete")
                self.is_ready = True

            elif "serverContent" in data:
                self.logger.info("Received server content")
                server_content = data["serverContent"]

                if "modelTurn" in server_content:
                    for part in server_content["modelTurn"].get("parts", []):
                        if "inlineData" in part and "audio/pcm" in (mime_type := part["inlineData"]["mimeType"]):
                            if self.first_output_token_time is None:
                                self.first_output_token_time = time.perf_counter()
                                self.logger.info("First output token received at: {}", self.first_output_token_time)

                            rate = int(mime_type.split(";rate=")[1]) if ";rate=" in mime_type else self.output_audio_sample_rate
                            self.output_audio_bytes += base64.b64decode(part["inlineData"]["data"])
                            self.logger.debug("Received audio chunk: {} bytes, rate: {}", len(part["inlineData"]["data"]), rate)

                if "inputTranscription" in server_content:
                    self.input_transcription += server_content["inputTranscription"]["text"]
                    self.logger.info("Input transcription: {}", self.input_transcription)

                if "outputTranscription" in server_content:
                    self.output_transcription += server_content["outputTranscription"]["text"]
                    self.logger.info("Transcription: {}", self.output_transcription)

                if server_content.get("turnComplete", False):
                    self.is_complete = True
                    self.logger.info("Turn complete")

                    usage_metadata = server_content.get("usageMetadata", {})
                    for token_details in usage_metadata.get("responseTokensDetails", []):
                        if token_details.get("modality") == "AUDIO":
                            self.output_audio_tokens = token_details.get("tokenCount", 0)
                            self.logger.info("Output tokens: {}", self.output_audio_tokens)

            elif "error" in data:
                self.logger.error("Server error: {}", json.dumps(data, indent=2))
                self.is_complete = True

        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse message: {}", e)
        except Exception as e:
            self.logger.error("Error processing message: {}", e)

if __name__ == "__main__":
    from pathlib import Path

    # os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

    harness = GeminiSpeechToSpeechHarness.from_file(
        GeminiSpeechToSpeechHarness.Model.GEMINI_2_5_FLASH_PREVIEW_NATIVE_AUDIO_DIALOG,
        Path("./examples/scooby.wav")
    )

    input_transcription, output_transcription, output_audio = asyncio.run(harness.run())
    print(f"Input transcription: {input_transcription}")
    print(f"Output transcription: {output_transcription}")
    output_audio.export("./gemini.wav", format="wav")
