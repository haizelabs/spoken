# speak!
<div align="center">

`pip install speak`

</div>


```python

```

A single interface around speech-to-speech foundation models.
**TODO: simple desription of these models**

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
