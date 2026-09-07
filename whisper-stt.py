from faster_whisper import WhisperModel


# Load model once
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)


def transcribe_audio(input_file):
    segments, info = model.transcribe(
        input_file,
        word_timestamps=True
    )

    results = []

    for segment in segments:
        words = []

        if segment.words:
            for word in segment.words:
                words.append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                })

        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "words": words
        })

    return results