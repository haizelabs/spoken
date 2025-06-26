import asyncio
import base64
import io
import json
import os
import time
import uuid
from enum import Enum
from typing import Any, Optional, Tuple

import numpy as np
from pydub import AudioSegment

from spoken.base import SpeechToSpeechHarness


class NovaSpeechToSpeechHarness(SpeechToSpeechHarness):
    """
    Amazon Nova Sonic (https://aws.amazon.com/ai/generative-ai/nova/speech/)

    - amazon.nova-sonic-v1:0
    """

    class Model(Enum):
        AMAZON_NOVA_SONIC_V1_0 = "amazon.nova-sonic-v1:0"

    input_audio_sample_rate: Optional[int] = 16000
    output_audio_sample_rate: Optional[int] = 24000
    audio_token_frame_rate: Optional[int] = 25

    bedrock_client: Any
    stream: Optional[Any]
    is_active: bool
    role: Optional[str]

    def __init__(
        self,
        model: Model,
        source_audio_signal: np.ndarray,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ):
        super().__init__(model, source_audio_signal, system_prompt, temperature)

        from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
        from aws_sdk_bedrock_runtime.config import (Config,
                                                    HTTPAuthSchemeResolver,
                                                    SigV4AuthScheme)
        from smithy_aws_core.credentials_resolvers.environment import \
            EnvironmentCredentialsResolver

        region = os.environ.get("AWS_REGION", "us-east-1")
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
            region=region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()},
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)
        self.stream = None
        self.is_active = False

        self.role = None

    async def run(self) -> Tuple[Optional[str], str, AudioSegment]:
        from aws_sdk_bedrock_runtime.client import \
            InvokeModelWithBidirectionalStreamOperationInput

        self.session_id = str(uuid.uuid4())
        self.prompt_name = str(uuid.uuid4())
        self.text_content_id = str(uuid.uuid4())
        self.audio_content_id = str(uuid.uuid4())

        self.is_active = True
        self.stream = await self.bedrock_client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_name)
        )

        response_task = asyncio.create_task(self._process_messages())

        # 1. sessionStart
        await self._send_message({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 4096,
                        "topP": 0.9,
                        "temperature": self.temperature,
                    }
                }
            }
        })

        # 2. promptStart
        await self._send_message({
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name, 
                    "textOutputConfiguration": {
                        "mediaType": "text/plain",
                        "enabled": True,
                    },
                    "audioOutputConfiguration": {
                        "audioType": "SPEECH",
                        "encoding": "base64",
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": self.output_audio_sample_rate,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "matthew",
                    },
                    "toolUseOutputConfiguration": {"mediaType": "application/json"},
                    "toolConfiguration": {"tools": []},
                }
            }
        })

        system_prompt = (
            self.system_prompt
            or "You are a friendly assistant. The user and you will engage in a spoken dialog "
            "exchanging the transcripts of a natural real-time conversation. Keep your responses short, "
            "generally two or three sentences for chatty scenarios."
        )

        # 3. contentStart for system prompt
        await self._send_message({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.text_content_id,
                    "role": "SYSTEM",
                    "type": "TEXT",
                    "interactive": True,
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        })

        # 4. textInput for system prompt
        await self._send_message({
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.text_content_id,
                    "content": system_prompt,
                }
            }
        })

        # 5. contentEnd for system prompt
        await self._send_message({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": self.text_content_id,
                }
            }
        })

        # 6. contentStart for input audio
        await self._send_message({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_id,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": self.input_audio_sample_rate,
                        "sampleSizeBits": 16,
                        "encoding": "base64",
                        "channelCount": 1,
                        "audioType": "SPEECH",
                    },
                }
            }
        })

        # 7. audioInput for input audio
        await self._send_message({
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_id,
                    "content": self.input_audio_base64,
                }
            }
        })

        # 8. simulate 200ms of silence for automatic turn detection to trigger END_TURN for USER
        quiet_chunk_base64 = base64.b64encode(b"\x00" * 1024).decode("utf-8")
        for _ in range(200):
            await self._send_message({
                "event": {
                    "audioInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_id,
                        "content": quiet_chunk_base64,
                    }
                }
            })
            await asyncio.sleep(0.01)

        self.input_audio_sent_time = time.perf_counter()
        self.logger.info("Input audio fully sent at: {}", self.input_audio_sent_time)

        self.logger.info("Waiting for server to complete output...")
        timeout_seconds = 10
        start_time = time.time()

        while not self.is_complete and (time.time() - start_time) < timeout_seconds:
            await asyncio.sleep(0.1)

        if not self.is_complete:
            self.logger.warning("Timeout waiting for Nova response after {} seconds", timeout_seconds)
            self.is_complete = True

        if self.is_active:
            # 9. contentEnd for input audio
            await self._send_message({
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_id,
                    }
                }
            })

            # 10. promptEnd
            await self._send_message({
                "event": {
                    "promptEnd": {
                        "promptName": self.prompt_name,
                    }
                }
            })

            # 11. sessionEnd
            await self._send_message({"event": {"sessionEnd": {}}})
            await self.stream.input_stream.close()

        if response_task and not response_task.done():
            self.is_active = False
            try:
                await asyncio.wait_for(response_task, timeout=2.0)
            except asyncio.TimeoutError:
                self.logger.debug("Response task didn't stop naturally, cancelling...")
                response_task.cancel()
                try:
                    await response_task
                except asyncio.CancelledError:
                    pass

        self.is_active = False

        self.output_audio = AudioSegment.from_raw(
            io.BytesIO(self.output_audio_bytes),
            sample_width=2,
            frame_rate=self.output_audio_sample_rate,
            channels=1,
        )

        return self.input_transcription, self.output_transcription, self.output_audio

    async def _send_message(self, event_dict):
        from aws_sdk_bedrock_runtime.models import (
            BidirectionalInputPayloadPart,
            InvokeModelWithBidirectionalStreamInputChunk)

        event_json = json.dumps(event_dict)
        self.logger.debug("Sending event: {}", event_json)
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self.stream.input_stream.send(event)

    async def _process_messages(self):
        try:
            self.logger.info("Starting response processing")
            while self.is_active:
                try:
                    output = await self.stream.await_output()
                    result = await output[1].receive()

                    if result.value and result.value.bytes_:
                        response_data = result.value.bytes_.decode("utf-8")
                        try:
                            json_data = json.loads(response_data)
                            self._handle_event(json_data)
                        except json.JSONDecodeError:
                            self.logger.debug("Non-JSON response: {}", response_data)
                    else:
                        self.logger.debug("Empty response received")
                except StopAsyncIteration:
                    self.logger.info("Stream ended")
                    break
                except Exception as e:
                    if "No events to transform were found" not in str(e):
                        self.logger.error("Error receiving response: {}", e)
                    break
        except Exception as e:
            if self.is_active:
                self.logger.error("Error processing responses: {}", e)
                self.is_complete = True

    def _handle_event(self, data):
        self.logger.debug("Received event: {}", json.dumps(data, indent=2))

        if "event" in data:
            event = data["event"]

            if "contentStart" in event:
                self.logger.info("Content start received")
                self.is_ready = True
                self.role = event["contentStart"].get("role")

            elif "audioOutput" in event:
                audio_output = event["audioOutput"]
                content_id = audio_output.get("contentId", audio_output.get("contentName", "unknown"))
                self.logger.info("Audio output received for: {} (type: {})", content_id, audio_output.get("type", "unknown"))

                if "content" in audio_output:
                    if self.first_output_token_time is None:
                        self.first_output_token_time = time.perf_counter()
                        self.logger.info("First output token received at: {}", self.first_output_token_time)
                    
                    audio_content = audio_output["content"]
                    if isinstance(audio_content, str):
                        try:
                            audio_bytes = base64.b64decode(audio_content)
                            self.output_audio_bytes += audio_bytes
                            self.logger.debug("Received audio chunk: {} bytes", len(audio_bytes))
                        except Exception as e:
                            self.logger.error("Failed to decode audio content: {}", e)
                    else:
                        self.logger.warning("Unexpected audio content format: {}", type(audio_content))

            elif "textOutput" in event:
                text_output = event["textOutput"]
                content_id = text_output.get("contentId", text_output.get("contentName", "unknown"))
                self.logger.info("Text output received for: {} (type: {})", content_id, text_output.get("type", "unknown"))

                if "content" in text_output:
                    text_content = text_output["content"]
                    if isinstance(text_content, str):
                        if self.role == "USER":
                            self.input_transcription += text_content
                            self.logger.info("Input transcription: {}", self.input_transcription)
                        else:
                            self.output_transcription += text_content
                            self.logger.info("Output transcription: {}", self.output_transcription)

            elif "contentEnd" in event:
                content_end = event["contentEnd"]
                content_id = content_end.get("contentId", content_end.get("contentName", "unknown"))
                self.logger.info("Content end received for: {} (type: {})", content_id, content_end.get("type", "unknown"))
                if content_end.get("type") == "AUDIO" and content_end.get("stopReason") == "END_TURN":
                    self.logger.info("Audio content end received - output complete")
                    self.is_complete = True

            elif "promptEnd" in event:
                self.logger.info("Prompt end received")

            elif "sessionEnd" in event:
                self.logger.info("Session end received")
                self.is_complete = True

            elif "usageEvent" in event:
                usage_event = event["usageEvent"]
                if "details" in usage_event and "total" in usage_event["details"]:
                    total = usage_event["details"]["total"]
                    if "input" in total and "speechTokens" in total["input"]:
                        self.input_audio_tokens = total["input"]["speechTokens"]
                        self.logger.info("Input audio tokens: {}", self.input_audio_tokens)
                    if "output" in total and "speechTokens" in total["output"]:
                        self.output_audio_tokens = total["output"]["speechTokens"]
                        self.logger.info("Output audio tokens: {}", self.output_audio_tokens)

            elif "error" in event:
                self.logger.error("Error received: {}", json.dumps(event["error"], indent=2))
                self.is_complete = True

        elif "error" in data:
            self.logger.error("Server error: {}", json.dumps(data, indent=2))
            self.is_complete = True


if __name__ == "__main__":
    from pathlib import Path

    # os.environ["AWS_ACCESS_KEY_ID"] = "your-aws-access-key-id"
    # os.environ["AWS_SECRET_ACCESS_KEY"] = "your-aws-secret-access-key"
    # os.environ["AWS_SESSION_TOKEN"] = "your-aws-session-token"

    harness = NovaSpeechToSpeechHarness.from_file(
        NovaSpeechToSpeechHarness.Model.AMAZON_NOVA_SONIC_V1_0,
        Path("./examples/scooby.wav")
    )

    input_transcription, output_transcription, output_audio = asyncio.run(harness.run())
    print(f"Input transcription: {input_transcription}")
    print(f"Output transcription: {output_transcription}")
    output_audio.export("./nova.wav", format="wav")
