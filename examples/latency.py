import asyncio
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import spoken


def random_audio(input_wav_path: Path, duration_seconds: float) -> AudioSegment:
    """Generate audio with specified duration from random chunks of the input audio"""
    audio = AudioSegment.from_wav(input_wav_path)
    sample_rate = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())

    chunk_size_ms = 200
    chunk_size_samples = int(chunk_size_ms * sample_rate / 1000)
    num_chunks_needed = int(duration_seconds * 1000 / chunk_size_ms)

    final_audio = AudioSegment.empty()

    for _ in range(num_chunks_needed):
        max_start = len(samples) - chunk_size_samples
        start_idx = random.randint(0, max_start)

        chunk_samples = samples[start_idx:start_idx + chunk_size_samples]

        chunk_audio = AudioSegment(
            chunk_samples.tobytes(),
            sample_width=2,
            frame_rate=sample_rate,
            channels=1,
        )

        final_audio += chunk_audio

    return final_audio


async def benchmark_model(model_name: str, audio_length: float) -> dict:
    try:
        model = spoken(
            model_name,
            random_audio(Path("./examples/scooby.wav"), audio_length)
        )

        print(f"Running {model_name} with audio length {audio_length} seconds")

        try:
            await asyncio.wait_for(model.run(), timeout=120.0)
        except asyncio.TimeoutError:
            print(f"Timeout for {model_name} with {audio_length}s audio")
            return {
                "success": False,
            }

        latency = model.first_output_token_time - model.input_audio_sent_time
        print(f"Latency for {model_name} with {audio_length}s audio: {latency} seconds")

        return {
            "model_name": model_name,
            "audio_length_seconds": audio_length,
            "latency_seconds": latency,
            "success": True
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
        }


async def main():
    print("🎤 Speech-to-Speech Latency Benchmark")

    models = ["gpt-4o-realtime-preview-2024-12-17", "gemini-2.5-flash-preview-native-audio-dialog", "amazon.nova-sonic-v1:0"] #spoken.models
    audio_lengths = [0.2, 0.4, 1.0, 2.0, 4.0, 8.0, 12.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0]

    print(f"Testing {len(models)} models across {len(audio_lengths)} audio lengths")
    print(f"Total tests: {len(models) * len(audio_lengths)}")

    tasks = []
    for model in models:
        for length in audio_lengths:
            tasks.append(benchmark_model(model, length))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]

    from collections import defaultdict
    models = defaultdict(list)
    for result in successful_results:
        models[result["model_name"]].append(result)

    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    for (model, model_results), color in zip(models.items(), colors):
        for i, result in enumerate(model_results):
            plt.plot(
                result["audio_length_seconds"],
                result["latency_seconds"],
                'o',
                color=color,
                label=model if i == 0 else "",
            )

    plt.xlabel('Input Audio Length (seconds)')
    plt.ylabel('TTFT Latency (seconds)')
    plt.title('Input Audio Length vs TTFT Latency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./latency.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    asyncio.run(main()) 
