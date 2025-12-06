

import os
from typing import List, Dict, Any

import requests
import pandas as pd


try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ---------------- CONFIG ---------------- #


SERPER_API_KEY = "0e8c4f937a992f5ef8cac46328cf37620cba7230"
YOUTUBE_API_KEY = "AIzaSyCFWNxBIq7mq0qPpDJU3aeEp9-hg6wZIWA"

QUERY = "smart fan"
TOP_N_GOOGLE = 30
TOP_N_YOUTUBE = 30

BRANDS = [
    "atomberg",
    "havells",
    "orient",
    "crompton",
    "bajaj",
    "usha"
]


# ---------------- UTILS ---------------- #

def normalize_text(text: str) -> str:
    return (text or "").lower()


# ---------------- GOOGLE VIA SERPER ---------------- #

def collect_google_results(query: str, top_n: int = 30) -> List[Dict[str, Any]]:
    """
    Uses Serper.dev (or similar) to fetch Google search results.
    Endpoint: https://google.serper.dev/search
    Docs: https://serper.dev (make sure your key is from there)

    Returns: list of dicts with platform, title, snippet, link, source
    """
    if not SERPER_API_KEY or "YOUR_NEW_SERPER_API_KEY_HERE" in SERPER_API_KEY:
        print("[Google] No SERPER_API_KEY set. Skipping Google collection.")
        return []

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query
    }

    print("[Google] Calling Serper API...")
    resp = requests.post(url, json=payload, headers=headers)

    if resp.status_code != 200:
        print(f"[Google] ERROR {resp.status_code}")
        try:
            print("[Google] Response:", resp.json())
        except Exception:
            print("[Google] Raw text:", resp.text[:500])
        return []

    data = resp.json()
    organic = data.get("organic", [])

    results: List[Dict[str, Any]] = []
    for item in organic[:top_n]:
        title = item.get("title")
        snippet = item.get("snippet") or item.get("description", "")
        link = item.get("link")
        source = item.get("source") or "google"

        if not title or not link:
            continue

        results.append({
            "platform": "google",
            "title": title,
            "snippet": snippet,
            "link": link,
            "source": source,
            "views": 0,
            "likes": 0,
            "comments": 0
        })

    print(f"[Google] Collected {len(results)} results from Serper")
    return results


# ---------------- YOUTUBE COLLECTOR ---------------- #

def collect_youtube_results(query: str, top_n: int = 30) -> List[Dict[str, Any]]:

    if not YOUTUBE_API_KEY or "YOUR_NEW_YOUTUBE_API_KEY_HERE" in YOUTUBE_API_KEY:
        print("[YouTube] No YOUTUBE_API_KEY set. Skipping YouTube collection.")
        return []

    search_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "key": YOUTUBE_API_KEY,
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(top_n, 50),
        "relevanceLanguage": "en"
    }

    print("[YouTube] Fetching search results...")
    resp = requests.get(search_url, params=params)

    if resp.status_code != 200:
        print(f"[YouTube] ERROR {resp.status_code}")
        try:
            print("[YouTube] Response:", resp.json())
        except Exception:
            print("[YouTube] Raw text:", resp.text[:500])
        return []

    data = resp.json()
    video_items = data.get("items", [])
    video_ids = [item["id"]["videoId"] for item in video_items]

    if not video_ids:
        print("[YouTube] No video IDs found.")
        return []

    # Get statistics
    stats_url = "https://www.googleapis.com/youtube/v3/videos"
    stats_params = {
        "key": YOUTUBE_API_KEY,
        "part": "statistics",
        "id": ",".join(video_ids)
    }
    print("[YouTube] Fetching statistics...")
    stats_resp = requests.get(stats_url, params=stats_params)

    if stats_resp.status_code != 200:
        print(f"[YouTube] Stats ERROR {stats_resp.status_code}")
        try:
            print("[YouTube] Stats response:", stats_resp.json())
        except Exception:
            print("[YouTube] Stats raw text:", stats_resp.text[:500])
        return []

    stats_data = {
        item["id"]: item["statistics"]
        for item in stats_resp.json().get("items", [])
    }

    results: List[Dict[str, Any]] = []
    for item in video_items:
        vid = item["id"]["videoId"]
        snippet = item["snippet"]
        stats = stats_data.get(vid, {})

        results.append({
            "platform": "youtube",
            "title": snippet.get("title"),
            "snippet": snippet.get("description"),
            "link": f"https://www.youtube.com/watch?v={vid}",
            "source": snippet.get("channelTitle"),
            "views": int(stats.get("viewCount", 0)),
            "likes": int(stats.get("likeCount", 0)) if "likeCount" in stats else 0,
            "comments": int(stats.get("commentCount", 0)) if "commentCount" in stats else 0
        })

    print(f"[YouTube] Collected {len(results)} results")
    return results[:top_n]


# ---------------- BRAND DETECTION ---------------- #

def detect_brands(text: str, brands: List[str]) -> List[str]:
    """
    Simple keyword-based brand detection.
    """
    text_norm = normalize_text(text)
    found = []
    for b in brands:
        if b.lower() in text_norm:
            found.append(b.lower())
    return list(set(found))


def add_brand_mentions(df: pd.DataFrame, brands: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["title"] = df["title"].fillna("")
    df["snippet"] = df["snippet"].fillna("")
    df["all_text"] = (df["title"] + " " + df["snippet"]).astype(str)
    df["brands_mentioned"] = df["all_text"].apply(lambda t: detect_brands(t, brands))
    return df


# ---------------- SENTIMENT ANALYSIS ---------------- #

def rule_based_sentiment(text: str) -> Dict[str, Any]:
    """
    Very simple backup sentiment if transformers are unavailable.
    """
    text = normalize_text(text)
    positive_words = [
        "good", "great", "excellent", "love", "amazing", "efficient",
        "saves", "recommended", "best", "happy", "satisfied", "very happy",
        "low power", "energy saving"
    ]
    negative_words = [
        "bad", "poor", "terrible", "hate", "worst",
        "problem", "issue", "complaint", "expensive", "disappointed",
        "noise", "noisy"
    ]

    pos_score = sum(word in text for word in positive_words)
    neg_score = sum(word in text for word in negative_words)

    if pos_score > neg_score:
        return {"label": "POSITIVE", "score": 0.6}
    elif neg_score > pos_score:
        return {"label": "NEGATIVE", "score": 0.6}
    else:
        return {"label": "NEUTRAL", "score": 0.5}


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TRANSFORMERS_AVAILABLE:
        print("[Sentiment] Using transformers pipeline...")
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

        texts = df["all_text"].tolist()
        results = sentiment_model(texts, truncation=True)

        df["sentiment_label"] = [r["label"] for r in results]
        df["sentiment_score"] = [r["score"] for r in results]
    else:
        print("[Sentiment] Transformers not available, using rule-based sentiment.")
        labels = []
        scores = []
        for text in df["all_text"].tolist():
            r = rule_based_sentiment(text)
            labels.append(r["label"])
            scores.append(r["score"])
        df["sentiment_label"] = labels
        df["sentiment_score"] = scores

    # POSITIVE -> 1, others -> 0
    df["is_positive"] = df["sentiment_label"].apply(lambda x: 1 if x == "POSITIVE" else 0)
    return df


# ---------------- SOV METRICS ---------------- #

def compute_sov(df: pd.DataFrame, brands: List[str]) -> pd.DataFrame:
    """
    Computes per-brand:
    - mentions
    - engagement (views + likes + comments)
    - positive mentions
    - SoV by mentions
    - SoV by engagement
    - Share of positive voice
    """
    rows = []

    exploded = df.explode("brands_mentioned")
    exploded = exploded[exploded["brands_mentioned"].notna()]

    for brand in brands:
        brand_mask = exploded["brands_mentioned"] == brand
        brand_df = exploded[brand_mask]

        mentions = len(brand_df)

        engagement = 0
        if "views" in brand_df.columns:
            engagement += brand_df["views"].fillna(0).sum()
        if "likes" in brand_df.columns:
            engagement += brand_df["likes"].fillna(0).sum()
        if "comments" in brand_df.columns:
            engagement += brand_df["comments"].fillna(0).sum()

        positive_mentions = brand_df["is_positive"].fillna(0).sum()

        rows.append({
            "brand": brand,
            "mentions": int(mentions),
            "engagement": int(engagement),
            "positive_mentions": int(positive_mentions)
        })

    metrics_df = pd.DataFrame(rows)

    total_mentions = metrics_df["mentions"].sum()
    total_engagement = metrics_df["engagement"].sum()
    total_positive = metrics_df["positive_mentions"].sum()

    if total_mentions > 0:
        metrics_df["sov_mentions_pct"] = metrics_df["mentions"] / total_mentions * 100
    else:
        metrics_df["sov_mentions_pct"] = 0.0

    if total_engagement > 0:
        metrics_df["sov_engagement_pct"] = metrics_df["engagement"] / total_engagement * 100
    else:
        metrics_df["sov_engagement_pct"] = 0.0

    if total_positive > 0:
        metrics_df["share_positive_voice_pct"] = metrics_df["positive_mentions"] / total_positive * 100
    else:
        metrics_df["share_positive_voice_pct"] = 0.0

    return metrics_df.sort_values("sov_mentions_pct", ascending=False)


# ---------------- MAIN PIPELINE ---------------- #

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("=== Atomberg Smart Fan – SoV Analysis ===")

    # 1) Google via Serper
    google_results = collect_google_results(QUERY, TOP_N_GOOGLE)

    # 2) YouTube
    youtube_results = collect_youtube_results(QUERY, TOP_N_YOUTUBE)

    all_results = google_results + youtube_results
    if not all_results:
        print("No data collected. Check your API keys / quotas.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv("data/raw/all_results.csv", index=False)
    print("[Data] Saved raw results to data/raw/all_results.csv")

    # 3) Brand mentions
    df = add_brand_mentions(df, BRANDS)

    # 4) Sentiment
    df = add_sentiment(df)

    df.to_csv("data/processed/results_with_sentiment.csv", index=False)
    print("[Data] Saved processed results with sentiment to data/processed/results_with_sentiment.csv")

    # 5) SoV metrics
    metrics_df = compute_sov(df, BRANDS)
    metrics_df.to_csv("data/processed/sov_metrics.csv", index=False)
    print("[Data] Saved SoV metrics to data/processed/sov_metrics.csv")

    print("\n=== Final SoV Metrics ===")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
