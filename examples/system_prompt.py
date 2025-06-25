from pathlib import Path
from spoken.models.openai import OpenAISpeechToSpeechHarness
import asyncio
import os

# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

harness = OpenAISpeechToSpeechHarness.from_file(
    OpenAISpeechToSpeechHarness.Model.GPT_4O_REALTIME_PREVIEW_2024_12_17,
    Path("./examples/input.wav"),
    system_prompt="You are a customer service agent."
)

input_transcription, output_transcription, output_audio = asyncio.run(harness.run())
print(f"Input transcription: {input_transcription}")
print(f"Output transcription: {output_transcription}")
output_audio.export("./openai.wav", format="wav")
