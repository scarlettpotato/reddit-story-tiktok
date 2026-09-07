def format_timestamp(seconds):
    """Convert seconds into SRT timestamp format."""

    milliseconds = int(round(seconds * 1000))

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def generate_srt(whisper_results, output_file):
    """Generate an SRT subtitle file from Whisper results."""

    with open(output_file, "w", encoding="utf-8") as file:

        for index, segment in enumerate(whisper_results, start=1):

            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])

            text = segment["text"]

            file.write(f"{index}\n")
            file.write(f"{start} --> {end}\n")
            file.write(f"{text}\n\n")

    print(f"Captions saved to {output_file}")