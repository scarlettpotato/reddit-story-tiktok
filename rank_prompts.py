import json
from ollama import chat


MODEL = "qwen3:8b"

# Minimum score required to keep a candidate.
MIN_SCORE = 7.0

# Scoring weights.
WEIGHTS = {
    "hook_potential": 0.20,
    "story_potential": 0.15,
    "emotional_impact": 0.20,
    "clarity": 0.15,
    "novelty": 0.15,
    "tiktok_potential": 0.15,
}


PROMPTS = [
    "Every night at 3:00 AM, someone knocks on my door. Tonight, I finally opened it.",

    "A programmer wakes up in a world where everyone can use magic except him.",

    "I inherited my grandfather's old house, but every room contains a version of my future.",

    "The last human on Earth receives a text message from someone living on Mars.",

    "Every lie you tell becomes true somewhere in the world.",
]


SYSTEM_PROMPT = """
You are an expert short-form fiction editor evaluating writing prompts.

Your job is to evaluate whether a writing prompt has strong potential
for an original 60-120 second fictional story and short-form video.

Evaluate ONLY the prompt itself, not a story that might later be written from it.

Score each category from 1 to 10:

1. hook_potential:
   How immediately interesting and attention-grabbing is the premise?

2. story_potential:
   How much room does the premise provide for a compelling story,
   escalation, conflict, and satisfying ending?

3. emotional_impact:
   How much potential does the premise have to create curiosity,
   tension, surprise, wonder, fear, humor, or other emotions?

4. clarity:
   How easy is the premise to understand immediately?

5. novelty:
   How fresh or distinctive does the premise feel compared with
   common short-form fiction concepts?

6. tiktok_potential:
   How well could this premise work as a 60-120 second narrated
   short-form video?

Return ONLY valid JSON using exactly this structure:

{
    "hook_potential": 1,
    "story_potential": 1,
    "emotional_impact": 1,
    "clarity": 1,
    "novelty": 1,
    "tiktok_potential": 1,
    "rationale": "Brief explanation of the evaluation."
}

Do not include markdown.
Do not include ```json.
Do not include any text before or after the JSON.
"""


def evaluate_prompt(prompt):
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Evaluate this writing prompt:\n\n{prompt}",
            },
        ],
        options={
            "temperature": 0.2,
        },
    )

    raw_response = response.message.content.strip()

    try:
        evaluation = json.loads(raw_response)
    except json.JSONDecodeError:
        print("Could not parse Qwen's response:")
        print(raw_response)
        raise

    return evaluation


def calculate_score(evaluation):
    score = 0

    for category, weight in WEIGHTS.items():
        score += evaluation[category] * weight

    return round(score, 2)


def main():
    candidates = []

    print("Evaluating writing prompts with Qwen3 8B...\n")

    for prompt in PROMPTS:
        print(f"Evaluating: {prompt}")

        evaluation = evaluate_prompt(prompt)
        overall_score = calculate_score(evaluation)

        candidates.append({
            "prompt": prompt,
            **evaluation,
            "overall_score": overall_score,
        })

    # Filter candidates below the minimum score.
    filtered = [
        candidate
        for candidate in candidates
        if candidate["overall_score"] >= MIN_SCORE
    ]

    # Sort highest score first.
    ranked = sorted(
        filtered,
        key=lambda candidate: candidate["overall_score"],
        reverse=True,
    )

    print("\n" + "=" * 70)
    print("RANKED WRITING PROMPTS")
    print("=" * 70)

    for rank, candidate in enumerate(ranked, start=1):
        print(f"\n#{rank} — Score: {candidate['overall_score']}/10")
        print(f"Prompt: {candidate['prompt']}")

        print("\nScores:")
        print(f"  Hook:             {candidate['hook_potential']}/10")
        print(f"  Story potential:  {candidate['story_potential']}/10")
        print(f"  Emotional impact: {candidate['emotional_impact']}/10")
        print(f"  Clarity:          {candidate['clarity']}/10")
        print(f"  Novelty:          {candidate['novelty']}/10")
        print(f"  TikTok potential: {candidate['tiktok_potential']}/10")

        print(f"\nRationale: {candidate['rationale']}")

    print("\n" + "=" * 70)
    print(f"Kept {len(ranked)} of {len(PROMPTS)} candidates")
    print("=" * 70)


if __name__ == "__main__":
    main()