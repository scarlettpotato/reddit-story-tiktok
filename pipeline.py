"""
Pipeline: fetch today's top r/writingprompts posts, then rank the prompts
(post titles) using the Qwen-based evaluator in rank_prompts.py.

Requires fetch_reddit_posts.py and rank_prompts.py to sit in the same folder
as this file (they're plain local imports, not installed packages).

Usage:
    python pipeline.py
"""

from fetch_reddit_posts import top_posts_last_n_hours
from rank_prompts import evaluate_prompt, calculate_score, MIN_SCORE, MODEL


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
    candidates = fetch_candidate_prompts(subreddit="writingprompts", hours=24, top_n=20)

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

    if ranked:
        print(f"\nTop pick: \"{ranked[0]['prompt']}\" (post id: {ranked[0]['id']})")


if __name__ == "__main__":
    main()
