from transformers import AutoConfig
from typing import List, Dict
import json
import re
import os


def get_scraped_tweets(path: str, allow_references: bool = True, allow_media: bool = True) -> List[Dict]:
    with open(path, "r") as f:
        tweets = json.load(f)

    res = []

    for tweet in tweets:

        if not allow_references and "referenced_tweets" in tweet:
            continue

        if not allow_media and "attachments" in tweet:
            continue

        res.append(tweet)

    return res


def preprocess_tweet(text: str, topic: str = None) -> str:
    # the topic replacement can be done to reduce the model's bias
    if topic is not None:
        text = text.replace(topic, "[TOPIC]")

    new_text = re.sub(r"@.*\s?", "@user ", text)
    new_text = re.sub(r"http.*\s?", "http ", new_text)

    return new_text


def compute_total_sentiment(tweets: List[Dict], config: AutoConfig, weighted: bool = False):
    res = {c: 0 for c in config.id2label.values()}

    count = 0

    if not weighted:

        for tweet in tweets:
            for i in range(len(config.id2label)):
                res[config.id2label[i]] += tweet["prediction"][config.id2label[i]]


    else:

        for tweet in tweets:
            for i in range(len(config.id2label)):
                # +1 so it also takes into account tweets with no retweets
                res[config.id2label[i]] += tweet["prediction"][config.id2label[i]] * (tweet["retweet_count"] + 1)

            count += tweet["retweet_count"] + 1

    if weighted:
        for i in range(len(config.id2label)):
            res[config.id2label[i]] /= count
            res[config.id2label[i]] *= 100

    return res


def write_predicted_tweets(text: str, tweets: List[Dict]) -> None:

    with open(f"data/predicted/twitter/{text}", "w") as f:
        json.dump(tweets, f)


def get_filename(path: str) -> str:
    return os.path.basename(path)
