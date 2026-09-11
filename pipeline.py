"""
Pipeline: fetch today's top r/writingprompts posts, then rank the prompts
(post titles) using the Qwen-based evaluator in rank_prompts.py.

Requires fetch_reddit_posts.py and rank_prompts.py to sit in the same folder
as this file (they're plain local imports, not installed packages).

Usage:
    python pipeline.py
"""

from ollama import chat

from fetch_reddit_posts import top_posts_last_n_hours
from rank_prompts import evaluate_prompt, calculate_score, MIN_SCORE, MODEL
from kokoro_tts import text_to_speech
from whisper_stt import transcribe_audio
from captions import generate_srt
from generate_scenes import generate_scenes
from assemble import assemble
from scene_planner import plan_scenes


STORY_SYSTEM_PROMPT = """
You are a short-form fiction writer creating narrated stories for a
60-120 second TikTok video.

Write an original short story based on the given writing prompt.

Requirements:
- Length: roughly 625-800 words (about 3.5-5 minutes of spoken narration).
- Hook the listener in the first sentence.
- Build tension or intrigue, then deliver a clear, satisfying ending
  (a twist, punchline, or emotional payoff).
- Dialogue: roughly 50 to 60 percent of the story.
- Paragraphs: generally short, frequently alternating between narration and dialogue.
- Write in plain, spoken-style prose meant to be read aloud, not
  formatted like a written short story (no headers, no chapter breaks).
- Do not restate or reference the prompt itself in the story.
- Do not include a title.

Return ONLY the story text. No preamble, no explanation, no markdown.
"""


def generate_story(prompt):
    """
    Generate a narration-ready short story from a writing prompt using
    the same local model used for ranking.
    """
    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": STORY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Writing prompt:\n\n{prompt}"},
        ],
        options={
            "temperature": 0.9,
        },
    )
    return response.message.content.strip()


def fetch_candidate_prompts(subreddit="writingprompts", hours=24, top_n=20):
    """
    Pull the top N posts from the subreddit in the last `hours` hours and
    return them as a list of dicts with the fields we need downstream:
    the prompt text (title), the post id, author, and Reddit score.
    """
    posts = top_posts_last_n_hours(subreddit=subreddit, hours=hours, top_n=top_n)

    candidates = []
    for post in posts:
        candidates.append({
            "id": post.get("id"),
            "author": post.get("author"),
            "reddit_score": post.get("score", 0),
            "prompt": post.get("title"),
        })
    return candidates


def rank_candidates(candidates):
    """
    Run each candidate's prompt text through the Qwen evaluator and attach
    the resulting scores. Returns only candidates at/above MIN_SCORE,
    sorted highest first.
    """
    ranked = []

    for candidate in candidates:
        print(f"Evaluating: {candidate['prompt']}")

        evaluation = evaluate_prompt(candidate["prompt"])
        overall_score = calculate_score(evaluation)

        ranked.append({
            **candidate,
            **evaluation,
            "overall_score": overall_score,
        })

    filtered = [c for c in ranked if c["overall_score"] >= MIN_SCORE]
    filtered.sort(key=lambda c: c["overall_score"], reverse=True)
    return filtered


def main():
    print(f"Fetching top r/writingprompts posts from the last 24 hours...\n")
    candidates = fetch_candidate_prompts(subreddit="writingprompts", hours=24, top_n=10)

    if not candidates:
        print("No posts found — nothing to rank.")
        return

    print(f"Fetched {len(candidates)} candidate prompts. Ranking with {MODEL}...\n")
    ranked = rank_candidates(candidates)

    print("\n" + "=" * 70)
    print("RANKED WRITING PROMPTS (from r/writingprompts)")
    print("=" * 70)

    for rank, candidate in enumerate(ranked, start=1):
        print(f"\n#{rank} — Score: {candidate['overall_score']}/10 "
              f"(Reddit score: {candidate['reddit_score']})")
        print(f"Prompt: {candidate['prompt']}")
        print(f"Post ID: {candidate['id']} — by u/{candidate['author']}")

        print("\nScores:")
        print(f"  Hook:             {candidate['hook_potential']}/10")
        print(f"  Story potential:  {candidate['story_potential']}/10")
        print(f"  Emotional impact: {candidate['emotional_impact']}/10")
        print(f"  Clarity:          {candidate['clarity']}/10")
        print(f"  Novelty:          {candidate['novelty']}/10")
        print(f"  TikTok potential: {candidate['tiktok_potential']}/10")

        print(f"\nRationale: {candidate['rationale']}")

    print("\n" + "=" * 70)
    print(f"Kept {len(ranked)} of {len(candidates)} candidates")
    print("=" * 70)

    if not ranked:
        print("\nNo candidates met the minimum score — no story generated.")
        return

    top = ranked[0]
    print(f"\nTop pick: \"{top['prompt']}\" (post id: {top['id']})")

    print("\n" + "=" * 70)
    print("GENERATING STORY")
    print("=" * 70)

    story = generate_story(top["prompt"])

    print(f"\n{story}\n")

    with open("story_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Prompt: {top['prompt']}\n")
        f.write(f"Source post id: {top['id']} (u/{top['author']})\n\n")
        f.write(story)

    print("Saved story to story_output.txt")
    #Convert generated story to a narrated .wav file
    print ("Converting text to speech...")
    wav_file = text_to_speech(story, "output.wav")
    #create word-level timestamps for captioning
    transcribed_audio = transcribe_audio("output.wav")
    #create caption
    generate_srt(transcribed_audio, "captions.srt")
    plan_scenes()
    generate_scenes()
    assemble()
    print("Pipeline Completed!")

if __name__ == "__main__":
    main()