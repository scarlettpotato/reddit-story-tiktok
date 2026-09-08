#!/usr/bin/env python3
"""
scene_planner.py

Converts a story into visually useful scenes using a local Ollama model.

Default model: qwen3:8b

Usage:
    python scene_planner.py --story story.txt
    python scene_planner.py --story story.txt --output scenes.json
    python scene_planner.py --story "A programmer woke up in a strange forest."

Requirements:
    - Ollama installed and running
    - Your Qwen model available in Ollama
    - Python 3.9+
    - requests: pip install requests
"""

import argparse
import json
import sys
import time
from pathlib import Path
from ollama import chat



OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:8b"


SYSTEM_PROMPT = r"""
You are a professional visual story planner for short-form vertical videos.

Your job is to convert a narrated story into a sequence of visually distinct scenes
that can each be represented by ONE generated still image.

IMPORTANT:
- Do not create a new scene for every sentence.
- Create a new scene when the visual subject, action, location, composition, or
  emotional beat meaningfully changes.
- Prefer approximately 1 scene every 3-7 seconds of narration.
- A scene can contain multiple sentences if they describe the same visual moment.
- Every scene must be visually understandable without hearing the narration.
- Avoid trying to depict abstract concepts literally.
- Keep recurring characters visually consistent across scenes.
- Keep locations and important objects visually consistent.
- Image prompts should be detailed enough for Stable Diffusion.
- Do not put dialogue, captions, subtitles, watermarks, logos, or written text
  into image prompts unless the story specifically requires text as a physical
  object.
- Favor cinematic compositions suitable for a 9:16 vertical video.
- Describe the subject, action, environment, lighting, mood, camera framing,
  and important visual details.
- Do not invent major plot events that are not in the story.

Return ONLY valid JSON. Do not use markdown fences.

The JSON must have exactly this structure:

{
  "title": "short title",
  "visual_style": "overall visual style to use for every image",
  "characters": [
    {
      "id": "character_1",
      "name": "short name",
      "description": "consistent physical appearance and clothing"
    }
  ],
  "locations": [
    {
      "id": "location_1",
      "name": "short name",
      "description": "consistent visual description"
    }
  ],
  "scenes": [
    {
      "scene_id": 1,
      "narration": "exact portion of the story spoken during this scene",
      "duration_seconds": 0,
      "visual_description": "what should be visible in the image",
      "image_prompt": "complete Stable Diffusion prompt",
      "characters": ["character_1"],
      "location": "location_1",
      "mood": "short mood description",
      "camera": "shot/framing/camera description"
    }
  ]
}

Rules for duration_seconds:
- Estimate duration from narration length.
- Assume approximately 2.5 words per second for natural narration.
- Use a minimum of 3 seconds per scene.
- Round to one decimal place.
- The sum of scene durations should approximately match the full narration.

Rules for image_prompt:
- Make it self-contained.
- Repeat the relevant character appearance when a character appears.
- Repeat relevant location details when useful.
- Include vertical composition suitable for 9:16.
- Do not include contradictory styles or details.
"""


def call_ollama(story: str, model: str) -> str:
    prompt = f"""{SYSTEM_PROMPT}

STORY TO PLAN:
{story}
"""

    response = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        think=False,
        options={
            "temperature": 0.3,
            "num_ctx": 16384,     # give it real room — tune to your story length
            "num_predict": 4096,  # cap generation, but high enough to not truncate
        }
    )

    return response.message.content


def parse_and_validate(raw_response: str) -> dict:
    try:
        result = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Qwen returned invalid JSON.\n\n"
            f"Raw response:\n{raw_response}"
        ) from exc

    required_top_level = ["title", "visual_style", "characters", "locations", "scenes"]

    for field in required_top_level:
        if field not in result:
            raise RuntimeError(f"Missing required field: {field}")

    if not isinstance(result["scenes"], list) or not result["scenes"]:
        raise RuntimeError("No scenes were generated.")

    required_scene_fields = [
        "scene_id",
        "narration",
        "duration_seconds",
        "visual_description",
        "image_prompt",
        "characters",
        "location",
        "mood",
        "camera",
    ]

    for index, scene in enumerate(result["scenes"], start=1):
        for field in required_scene_fields:
            if field not in scene:
                if field == "duration_seconds":
                    word_count = len(scene.get("narration", "").split())
                    scene["duration_seconds"] = max(3.0, round(word_count / 2.5, 1))
                    continue
                raise RuntimeError(
                    f"Scene {index} is missing required field: {field}"
                )

        scene["scene_id"] = index
        scene["duration_seconds"] = round(
            max(3.0, float(scene["duration_seconds"])), 1
        )

    return result


def read_story(story_argument: str) -> str:
    path = Path(story_argument)

    if path.exists() and path.is_file():
        story = path.read_text(encoding="utf-8").strip()
    else:
        story = story_argument.strip()

    if not story:
        raise RuntimeError("The story is empty.")

    return story


def main():
    parser = argparse.ArgumentParser(
        description="Convert a story into image-generation scenes using Ollama."
    )

    parser.add_argument(
        "--story",
        required=True,
        help="Path to a story .txt file OR the story text itself."
    )

    parser.add_argument(
        "--output",
        default="scenes.json",
        help="Output JSON filename. Default: scenes.json"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model to use. Default: {DEFAULT_MODEL}"
    )

    args = parser.parse_args()

    try:
        # Start timer
        start_time = time.perf_counter()
        story = read_story(args.story)

        print(f"Using Ollama model: {args.model}")
        print("Planning scenes...")

        raw_response = call_ollama(story, args.model)
        Path("raw_response.json").write_text(raw_response, encoding="utf-8")
        result = parse_and_validate(raw_response)

        output_path = Path(args.output)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        total_duration = sum(
            scene["duration_seconds"] for scene in result["scenes"]
        )

        print()
        print("Scene planning complete.")
        print(f"Scenes generated: {len(result['scenes'])}")
        print(f"Estimated duration: {total_duration:.1f} seconds")
        print(f"Output: {output_path.resolve()}")

        print()
        print("Scenes:")
        for scene in result["scenes"]:
            print(
                f"  {scene['scene_id']}: "
                f"{scene['duration_seconds']}s - "
                f"{scene['visual_description']}"
            )
        # Stop timer
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print("\nSaved: output.png")
        print(f"Total runtime: {elapsed_time:.2f} seconds")
        print(f"Total runtime: {elapsed_time / 60:.2f} minutes")

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
