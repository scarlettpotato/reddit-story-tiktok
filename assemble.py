
#!/usr/bin/env python3

"""
assemble_video.py

Assembles a TikTok-style vertical video from:

    - scenes.json
        Scene order + approximate durations

    - output_images/
        One 512x512 image per scene:
        scene_001.png, scene_002.png, ...

    - output.wav
        Narrated audio.
        THE AUDIO DURATION IS THE MASTER TIMELINE.

    - captions.srt
        Captions to burn into the final video.

The script:

    1. Reads the narration duration using ffprobe.
    2. Reads the scene durations from scenes.json.
    3. Scales the scene durations proportionally so that
       they EXACTLY fill the narration duration.
    4. Creates a 1080x1920 vertical slideshow.
    5. Adds the narration.
    6. Burns in captions.

Source images are expected to be 512x512.

Example:

    python assemble.py

or:

    python assemble.py \
        --scenes scenes.json \
        --images output_images \
        --audio output.wav \
        --captions captions.srt \
        --output final_video.mp4
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def run_command(cmd: list, description: str):
    """Run a command and raise an error if it fails."""

    print(f"\n{description}...")
    print(" ".join(str(x) for x in cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Command failed during: {description}"
        )

    return result


def run_ffmpeg(cmd: list, description: str):
    """Run FFmpeg."""

    return run_command(cmd, description)


def get_media_duration(path: Path) -> float:
    """
    Get media duration using ffprobe.

    Returns duration in seconds.
    """

    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path.resolve())
    ]

    result = run_command(
        cmd,
        f"Reading duration of {path.name}"
    )

    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(
            f"Could not determine duration of {path}"
        )


# ---------------------------------------------------------------------------
# Scene handling
# ---------------------------------------------------------------------------

def load_scenes(scenes_path: Path) -> list:
    """Load and validate scenes.json."""

    data = json.loads(
        scenes_path.read_text(encoding="utf-8")
    )

    scenes = data.get("scenes")

    if not scenes:
        raise RuntimeError(
            f"No scenes found in {scenes_path}"
        )

    for scene in scenes:
        if "scene_id" not in scene:
            raise RuntimeError(
                "Every scene must contain 'scene_id'"
            )

        if "duration_seconds" not in scene:
            raise RuntimeError(
                f"Scene {scene['scene_id']} is missing "
                "'duration_seconds'"
            )

        duration = float(scene["duration_seconds"])

        if duration <= 0:
            raise RuntimeError(
                f"Scene {scene['scene_id']} has invalid duration: "
                f"{duration}"
            )

    return scenes


def normalize_scene_durations(
    scenes: list,
    narration_duration: float
) -> list:
    """
    Adjust scene durations proportionally so that their total
    duration exactly matches the narration duration.

    Example:

        Original scene durations:
            5 + 10 + 5 = 20 seconds

        Narration:
            30 seconds

        New scene durations:
            7.5 + 15 + 7.5 = 30 seconds
    """

    original_total = sum(
        float(scene["duration_seconds"])
        for scene in scenes
    )

    if original_total <= 0:
        raise RuntimeError(
            "Total scene duration must be greater than zero."
        )

    scale_factor = narration_duration / original_total

    print("\nScene timeline adjustment:")
    print(f"Original scene duration: {original_total:.3f}s")
    print(f"Narration duration:       {narration_duration:.3f}s")
    print(f"Scale factor:             {scale_factor:.6f}")

    normalized = []

    for scene in scenes:
        new_scene = dict(scene)

        original_duration = float(
            scene["duration_seconds"]
        )

        new_duration = original_duration * scale_factor

        new_scene["duration_seconds"] = new_duration

        normalized.append(new_scene)

        print(
            f"Scene {scene['scene_id']:03d}: "
            f"{original_duration:.3f}s -> "
            f"{new_duration:.3f}s"
        )

    return normalized


# ---------------------------------------------------------------------------
# Concat demuxer
# ---------------------------------------------------------------------------

def ffmpeg_concat_safe_path(path: Path) -> str:
    """
    Format a path for use inside an FFmpeg concat demuxer file.

    Uses forward slashes and escapes single quotes.
    """

    posix_path = path.resolve().as_posix()

    return posix_path.replace(
        "'",
        "'\\''"
    )


def build_concat_file(
    scenes: list,
    images_dir: Path,
    concat_path: Path
):
    """
    Create concat_list.txt.

    Each image receives the normalized scene duration.
    """

    lines = []
    last_image = None

    for scene in scenes:

        scene_id = int(scene["scene_id"])
        duration = float(scene["duration_seconds"])

        image_path = (
            images_dir /
            f"scene_{scene_id:03d}.png"
        )

        if not image_path.exists():
            raise RuntimeError(
                f"Missing image for scene {scene_id}: "
                f"{image_path}"
            )

        safe_path = ffmpeg_concat_safe_path(
            image_path
        )

        lines.append(
            f"file '{safe_path}'"
        )

        lines.append(
            f"duration {duration:.6f}"
        )

        last_image = image_path

    # FFmpeg concat demuxer requires the final image
    # to be repeated so that the final duration is respected.
    if last_image:
        lines.append(
            f"file '{ffmpeg_concat_safe_path(last_image)}'"
        )

    with open(
        concat_path,
        "w",
        encoding="utf-8",
        newline="\n"
    ) as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Slideshow creation
# ---------------------------------------------------------------------------

def build_slideshow(
    concat_path: Path,
    slideshow_path: Path,
    width: int,
    height: int,
    fps: int
):
    """
    Convert the 512x512 scene images into a 1080x1920
    vertical slideshow.

    Images are:

        512x512
             ↓
        scale up while preserving aspect ratio
             ↓
        crop to 1080x1920
             ↓
        1080x1920

    The output uses constant 30 FPS by default.
    """

    video_filter = (
    f"scale=800:800,"
    f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
    f"setsar=1"
)

    cmd = [
        "ffmpeg",
        "-y",

        "-f",
        "concat",

        "-safe",
        "0",

        "-i",
        str(concat_path.resolve()),

        "-vf",
        video_filter,

        # Constant frame rate.
        "-r",
        str(fps),

        "-fps_mode",
        "cfr",

        "-pix_fmt",
        "yuv420p",

        "-c:v",
        "libx264",

        "-preset",
        "medium",

        "-crf",
        "18",

        str(slideshow_path.resolve())
    ]

    run_ffmpeg(
        cmd,
        "Building slideshow from scene images"
    )


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def add_audio(
    slideshow_path: Path,
    audio_path: Path,
    with_audio_path: Path
):
    """
    Add narration to the slideshow.

    The narration is the master timeline.

    The slideshow should already have exactly the same
    duration as the narration.

    We intentionally do NOT use -shortest here.
    """

    cmd = [
        "ffmpeg",
        "-y",

        "-i",
        str(slideshow_path.resolve()),

        "-i",
        str(audio_path.resolve()),

        "-map",
        "0:v:0",

        "-map",
        "1:a:0",

        "-c:v",
        "copy",

        "-c:a",
        "aac",

        "-b:a",
        "192k",

        "-t",
        str(get_media_duration(audio_path)),

        str(with_audio_path.resolve())
    ]

    run_ffmpeg(
        cmd,
        "Adding narration audio"
    )


# ---------------------------------------------------------------------------
# Subtitle path handling
# ---------------------------------------------------------------------------

def ffmpeg_filter_safe_path(path: Path) -> str:
    """
    Format a Windows path for use inside an FFmpeg filter.

    Example:

        C:/Users/me/captions.srt

    becomes:

        C\\:/Users/me/captions.srt

    for the FFmpeg filter parser.
    """

    posix_path = path.resolve().as_posix()

    return posix_path.replace(
        ":",
        r"\:"
    )


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

def burn_captions(
    with_audio_path: Path,
    captions_path: Path,
    output_path: Path,
    font_size: int,
    margin_v: int
):
    """Burn captions into the final video."""

    force_style = (
        f"FontName=Arial,"
        f"FontSize={font_size},"
        f"Bold=1,"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BorderStyle=1,"
        f"Outline=3,"
        f"Shadow=0,"
        f"Alignment=2,"
        f"MarginV={margin_v}"
    )

    srt_path_escaped = (
        ffmpeg_filter_safe_path(captions_path)
    )

    vf_arg = (
        f"subtitles='{srt_path_escaped}':"
        f"force_style='{force_style}'"
    )

    cmd = [
        "ffmpeg",
        "-y",

        "-i",
        str(with_audio_path.resolve()),

        "-vf",
        vf_arg,

        "-c:v",
        "libx264",

        "-preset",
        "medium",

        "-crf",
        "18",

        "-pix_fmt",
        "yuv420p",

        "-c:a",
        "copy",

        str(output_path.resolve())
    ]

    run_ffmpeg(
        cmd,
        "Burning in captions"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Assemble a vertical TikTok video "
            "using narration as the master timeline."
        )
    )

    parser.add_argument(
        "--scenes",
        default="scenes.json"
    )

    parser.add_argument(
        "--images",
        default="output_images"
    )

    parser.add_argument(
        "--audio",
        default="output.wav"
    )

    parser.add_argument(
        "--captions",
        default="captions.srt"
    )

    parser.add_argument(
        "--output",
        default="final_video.mp4"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1080
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1920
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30
    )

    parser.add_argument(
    "--font-size",
    type=int,
    default=None,
    help="Caption font size. Defaults to a value scaled for the output resolution."
)

    parser.add_argument(
    "--margin-v",
    type=int,
    default=None,
    help="Caption bottom margin. Defaults to a value scaled for the output resolution."
)

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help=(
            "Keep intermediate files."
        )
    )

    args = parser.parse_args()
    # -----------------------------------------------------------------------
    # Scale caption settings based on output resolution
    # -----------------------------------------------------------------------

    base_width = 1080
    base_height = 1920

    font_size = args.font_size or round(
        12 * args.width / base_width
    )

    margin_v = args.margin_v or round(
        200 * args.height / base_height
    )

    # -----------------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------------

    scenes_path = Path(args.scenes)
    images_dir = Path(args.images)
    audio_path = Path(args.audio)
    captions_path = Path(args.captions)
    output_path = Path(args.output)

    for path, label in [
        (scenes_path, "scenes file"),
        (images_dir, "images directory"),
        (audio_path, "audio file"),
        (captions_path, "captions file")
    ]:
        if not path.exists():
            raise RuntimeError(
                f"Missing {label}: {path}"
            )

    start_time = time.perf_counter()

    # -----------------------------------------------------------------------
    # Load scenes
    # -----------------------------------------------------------------------

    scenes = load_scenes(
        scenes_path
    )

    scenes = sorted(
        scenes,
        key=lambda s: int(s["scene_id"])
    )

    # -----------------------------------------------------------------------
    # Get narration duration
    # -----------------------------------------------------------------------

    narration_duration = get_media_duration(
        audio_path
    )

    print(
        f"\nNarration duration: "
        f"{narration_duration:.3f} seconds"
    )

    # -----------------------------------------------------------------------
    # Make narration the master timeline
    # -----------------------------------------------------------------------

    scenes = normalize_scene_durations(
        scenes,
        narration_duration
    )

    normalized_total = sum(
        float(scene["duration_seconds"])
        for scene in scenes
    )

    print(
        f"\nNormalized scene timeline: "
        f"{normalized_total:.3f} seconds"
    )

    # -----------------------------------------------------------------------
    # Temporary files
    # -----------------------------------------------------------------------

    concat_path = Path(
        "concat_list.txt"
    )

    slideshow_path = Path(
        "slideshow.mp4"
    )

    with_audio_path = Path(
        "with_audio.mp4"
    )

    # -----------------------------------------------------------------------
    # Build pipeline
    # -----------------------------------------------------------------------

    build_concat_file(
        scenes,
        images_dir,
        concat_path
    )

    build_slideshow(
        concat_path,
        slideshow_path,
        args.width,
        args.height,
        args.fps
    )

    # Check slideshow duration
    slideshow_duration = get_media_duration(
        slideshow_path
    )

    print(
        f"\nSlideshow duration: "
        f"{slideshow_duration:.3f} seconds"
    )

    print(
        f"Narration duration: "
        f"{narration_duration:.3f} seconds"
    )

    print(
        f"Difference: "
        f"{slideshow_duration - narration_duration:.3f} seconds"
    )

    add_audio(
        slideshow_path,
        audio_path,
        with_audio_path
    )

    burn_captions(
        with_audio_path,
        captions_path,
        output_path,
        font_size,
        margin_v
)

    # -----------------------------------------------------------------------
    # Final duration check
    # -----------------------------------------------------------------------

    final_duration = get_media_duration(
        output_path
    )

    print(
        f"\nFinal video duration: "
        f"{final_duration:.3f} seconds"
    )

    print(
        f"Narration duration: "
        f"{narration_duration:.3f} seconds"
    )

    print(
        f"Final difference: "
        f"{final_duration - narration_duration:.3f} seconds"
    )

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    if not args.keep_temp:

        for temp_file in [
            concat_path,
            slideshow_path,
            with_audio_path
        ]:
            temp_file.unlink(
                missing_ok=True
            )

    elapsed_time = (
        time.perf_counter() -
        start_time
    )

    print("\nDone.")

    print(
        f"Output: "
        f"{output_path.resolve()}"
    )

    print(
        f"Total runtime: "
        f"{elapsed_time:.2f} seconds"
    )

    print(
        f"Total runtime: "
        f"{elapsed_time / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()