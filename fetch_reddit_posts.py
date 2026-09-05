"""
Fetch Reddit posts using the Arctic Shift API — free, no API key required.

Arctic Shift docs: https://github.com/ArthurHeitmann/arctic_shift/blob/master/api
Web search UI:      https://arctic-shift.photon-reddit.com/search

Notes:
- No authentication needed, just plain HTTP GET requests.
- Keyword search (title/selftext) requires an author or subreddit to be set too.
- Be a good citizen: don't hammer it with rapid parallel requests.
"""

import time
import requests

BASE_URL = "https://arctic-shift.photon-reddit.com/api"


def search_posts(subreddit=None, author=None, query=None, limit=25, sort="desc",
                  hours_back=None):
    """
    Search for Reddit posts.

    Note: Arctic Shift only sorts by creation date (sort=asc/desc) — there is
    no server-side "sort by score". To get top posts, fetch a batch (usually
    with a wide limit) and sort by score yourself; see top_posts_last_n_hours().

    Args:
        subreddit:  restrict to this subreddit (no "r/" prefix), e.g. "python"
        author:     restrict to this Reddit username
        query:      keyword search in the title/body (requires subreddit or author too)
        limit:      max number of results (1-100, or "auto" for up to 1000)
        sort:       "asc" or "desc" by creation date
        hours_back: if set, only include posts created within the last N hours

    Returns:
        list of post dicts (raw Reddit-style fields: title, selftext, score, etc.)
    """
    params = {"limit": limit, "sort": sort}
    if subreddit:
        params["subreddit"] = subreddit
    if author:
        params["author"] = author
    if query:
        params["title"] = query  # searches the post title
    if hours_back is not None:
        params["after"] = int(time.time()) - hours_back * 3600

    resp = requests.get(f"{BASE_URL}/posts/search", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


def top_posts_last_n_hours(subreddit, hours=24, top_n=20):
    """
    Fetch posts from the last `hours` hours and return the `top_n` by score.

    Pulls a wide batch (up to 1000 via limit="auto") since Arctic Shift can't
    sort by score server-side, then ranks them in Python.
    """
    posts = search_posts(subreddit=subreddit, limit="auto", sort="desc", hours_back=hours)
    posts.sort(key=lambda p: p.get("score", 0), reverse=True)
    return posts[:top_n]

def get_title_and_author(posts):
    result = []

    for post in posts:
        result.append({
            'author': post.get('author'),
            'title': post.get('title')
        })
    return result


def get_post_by_id(post_id):
    """
    Fetch a single post by its Reddit ID (e.g. 'abc123', without the 't3_' prefix).
    """
    resp = requests.get(f"{BASE_URL}/posts/ids", params={"ids": post_id}, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return data[0] if data else None


def get_comments_for_post(post_id, limit=100):
    """
    Fetch comments for a given post ID (as a flat list, not a nested tree).
    """
    params = {"link_id": f"t3_{post_id}", "limit": limit}
    resp = requests.get(f"{BASE_URL}/comments/search", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


if __name__ == "__main__":
    # Example: top 20 posts from r/writingprompts in the last 24 hours
    posts = top_posts_last_n_hours(subreddit="writingprompts", hours=24, top_n=5)

    for post in posts:
        print(f"[{post.get('score', 0):>5}] {post.get('title')}")
        print(f"        by u/{post.get('author')} — {post.get('url')}")
        print()

    # Example: fetch comments for the first post found
    if posts:
        first_id = posts[0]["id"]
        comments = get_comments_for_post(first_id, limit=20)
        print(f"--- First 5 comments on: {posts[0]['title']} ---")
        for c in comments:
            print(f"u/{c.get('author')}: {c.get('body')[:150]}")
    
    print(get_title_and_author(posts))