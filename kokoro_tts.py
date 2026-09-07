from kokoro import KPipeline
import soundfile as sf
import numpy as np

def text_to_speech(text, output_file):
    audio_chunks = []
    pipeline = KPipeline(
    lang_code="a",
    repo_id="hexgrad/Kokoro-82M")

    generator = pipeline(
        text,
        voice="af_heart"
    )

    for _, _, audio in generator:
        audio_chunks.append(audio)

    if not audio_chunks:
        raise ValueError("Kokoro generated no audio.")

    full_audio = np.concatenate(audio_chunks)

    sf.write(output_file, full_audio, 24000)

    print(f"Audio saved to {output_file}")
