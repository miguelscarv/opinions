from dotenv import load_dotenv
import json
import os
from typing import List, Dict


def get_bearer_token() -> str:
    load_dotenv()

    return os.getenv("BEARER_TOKEN")


def remove_duplicate_tweets(tweets: List[Dict]) -> List[Dict]:
    seen = set()
    clean = []
    size_before = len(tweets)

    for tweet in tweets:
        if tweet["id"] not in seen:
            clean.append(tweet)
            seen.add(tweet["id"])

    if len(clean) != size_before:
        print(f"Removed {size_before-len(clean)} duplicate tweets")

    return clean


def write_scraped_tweets(text: str, tweets: List[Dict]) -> None:
    text = text.lower()
    text = text.replace(" ", "_")

    with open(f"data/scraped/twitter/{text}.json", "w") as f:
        json.dump(tweets, f)