<div align="center">

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.... .- .. --.. . .-.. .- -... ...&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;✨🎤 `pip install spoken` 🎤✨&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.. - .----. ... / .- / -... .- -.. / -.. .- -.--

</div>

---

`spoken` provides a single abstraction for a variety of audio foundation models. It is primarily designed for large-scale evaluation/benchmarking of realtime speech-to-speech models, but it can also be used as a drop-in inference library.

```python
# os.environ['LOG_LEVEL'] = 'DEBUG' # detailed client/server state management logging
import spoken

model = spoken("gpt-4o-realtime-preview-2024-12-17", "examples/scooby.wav")
input_asr, output_asr, output_audio = await model.run()

output_asr                   # "That's quite the story..."
len(output_audio)            # 8549ms
model.output_audio_tokens    # 254
```

Large audio models operate on audio tokens rather than transcribed text. This enables low-latency streaming conversational audio agents that directly generate audio end-to-end. Although promising and exciting, using these models requires non-trivial configuration and state management, due to major providers differing significantly in interface.

(AFAWK,) `spoken` supports all provider speech-to-speech models.
- [OpenAI Realtime](https://platform.openai.com/docs/guides/realtime)
  - gpt-4o-realtime-preview-2024-12-17
  - gpt-4o-mini-audio-preview-2024-12-17 [coming soon, not part of realtime API]
- [Gemini Multimodal Live](https://ai.google.dev/gemini-api/docs/live)
  - gemini-2.5-flash-preview-native-audio-dialog
  - gemini-2.5-flash-exp-native-audio-thinking-dialog
- [Amazon Nova Sonic](https://aws.amazon.com/ai/generative-ai/nova/speech/) (`pip install spoken[nova]`)
  - amazon.nova-sonic-v1:0

## Examples
* [Benchmarking TTFT (Time-To-First-Token) Latency](./examples/latency.py)
* [OpenAI System Prompt](./examples/system_prompt.py)
* more interesting things coming soon...

<div align="center">
<img src="https://github.com/user-attachments/assets/c8535dd7-6dec-4e38-aa78-374629f65c10"/>
</div>

## Installation
- Simply run `pip install spoken`
  - Python 3.12+ required + `pip install spoken[nova]` + `portaudio.h` (+ OS X: `brew install portaudio`) for Amazon Nova Sonic support
