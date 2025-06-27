import spoken
import asyncio

model = spoken(
    "gpt-4o-realtime-preview-2024-12-17",
    "examples/scooby.wav"
)
asr_in, asr_out, audio_out = asyncio.run(model.run())

print(f"Input transcription: {asr_in}")
print(f"Output transcription: {asr_out}")
print(f"# Audio Tokens: {model.output_audio_tokens}")
audio_out.export("./openai.wav", format="wav")
