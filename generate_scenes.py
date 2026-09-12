import argparse
import json
import time
from pathlib import Path
from compel import Compel

import torch
from diffusers import StableDiffusionPipeline

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def load_scenes(scenes_path: Path) -> list:
    data = json.loads(scenes_path.read_text(encoding="utf-8"))

    scenes = data.get("scenes")
    if not scenes:
        raise RuntimeError(f"No scenes found in {scenes_path}")

    return scenes


def generate_scenes():
    parser = argparse.ArgumentParser(
        description="Generate images for each scene in a scenes.json file."
    )
    parser.add_argument(
        "--scenes",
        default="scenes.json",
        help="Path to the scenes JSON file. Default: scenes.json"
    )
    parser.add_argument(
        "--output-dir",
        default="output_images",
        help="Directory to save generated images. Default: output_images"
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    scenes_path = Path(args.scenes)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = load_scenes(scenes_path)

    print("Loading Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    start_time = time.perf_counter()

    for scene in scenes:
        scene_id = scene.get("scene_id", "unknown")
        prompt = scene.get("image_prompt")
        conditioning = compel.build_conditioning_tensor(prompt)

        if not prompt:
            print(f"Scene {scene_id}: no image_prompt found, skipping.")
            continue

        print(f"\nScene {scene_id}: generating image...")
        print(f"Prompt: {prompt}")

        image = pipe(
            prompt_embeds=conditioning,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps
        ).images[0]

        output_path = output_dir / f"scene_{scene_id:03d}.png"
        image.save(output_path)
        print(f"Saved: {output_path}")

    elapsed_time = time.perf_counter() - start_time

    print(f"\nAll scenes processed.")
    print(f"Total runtime: {elapsed_time:.2f} seconds")
    print(f"Total runtime: {elapsed_time / 60:.2f} minutes")

def main():
    generate_scenes()

if __name__ == "__main__":
    main()