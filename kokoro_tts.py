from kokoro import KPipeline
import soundfile as sf

def text_to_speech(text, output_file):
    pipeline = KPipeline(lang_code="a")

    generator = pipeline(
        text,
        voice="af_heart"
    )

    for _, _, audio in generator:
        sf.write(output_file, audio, 24000)