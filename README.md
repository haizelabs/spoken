# spoken!
<div align="center">

`pip install spoken`

</div>

currently supports batch/offline evaluation for offline evaluations/benchmarking but can easily propagate audio chunks forward

```python
import spoken

model = spoken("gpt-4o-realtime-preview-2024-12-17", "examples/input.wav")
input_asr, output_asr, output_audio = await model.run()

output_asr                   # "That's quite the story..."
len(output_audio)            # 8549ms
model.output_audio_tokens    # 254
```

A single interface around speech-to-speech foundation models.

Supports
- [OpenAI Realtime](https://platform.openai.com/docs/guides/realtime)
  - gpt-4o-realtime-preview-2024-12-17
  - gpt-4o-mini-audio-preview-2024-12-17
- [Gemini Multimodal Live](https://ai.google.dev/gemini-api/docs/live)
  - gemini-2.5-flash-preview-native-audio-dialog
  - gemini-2.5-flash-exp-native-audio-thinking-dialog
- [Amazon Nova Sonic](https://aws.amazon.com/ai/generative-ai/nova/speech/)
  - amazon.nova-sonic-v1:0

## Installation
- need `portaudio.h` for Amazon Nova Sonic support (mac `brew install portaudio`)
