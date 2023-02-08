from transformers import AutoConfig
from typing import List, Dict
import torch
from torch.distributions import Categorical
import json
import re
import os
import numpy as np


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

    new_text = re.sub(r"@.+?(\s|$)", "@user ", text)
    new_text = re.sub(r"http.+(\s|$)", "http ", new_text)

    return new_text


def compute_total_sentiment(tweets: List[Dict], config: AutoConfig, weighted: bool = False):

    res = {c: 0 for c in config.id2label.values()}

    count = 0
    true_labels = 0

    if not weighted:

        for tweet in tweets:
            if "true_label" not in tweet:
                for i in range(len(config.id2label)):
                    if isinstance(tweet["prediction"], dict):
                        res[config.id2label[i]] += tweet["prediction"][config.id2label[i]]
                    else:
                        res[config.id2label[i]] += tweet["prediction"][-1][config.id2label[i]]
            else:
                true_labels += 1
                i = config.id2label[tweet["true_label"]]
                res[i] += 1

            count += 1

    else:

        for tweet in tweets:
            if "true_label" not in tweet:
                for i in range(len(config.id2label)):
                    # +1 so it also takes into account tweets with no retweets
                    if isinstance(tweet["prediction"], dict):
                        res[config.id2label[i]] += tweet["prediction"][config.id2label[i]] * (tweet["retweet_count"] + 1)
                    else:
                        res[config.id2label[i]] += tweet["prediction"][-1][config.id2label[i]] * (tweet["retweet_count"] + 1)
            else:
                true_labels += 1
                i = config.id2label[tweet["true_label"]]
                res[i] += (tweet["retweet_count"] + 1)

            count += tweet["retweet_count"] + 1

    for i in range(len(config.id2label)):
        res[config.id2label[i]] /= count
        res[config.id2label[i]] *= 100

    print(f"Found {true_labels} true labels")

    return res


def write_predicted_tweets(filename: str, tweets: List[Dict]) -> None:
    with open(f"data/predicted/twitter/{filename}", "w") as f:
        json.dump(tweets, f)


def get_filename(path: str) -> str:
    return os.path.basename(path)


def compute_uncertainty(tweets: List[Dict]) -> List[int]:
    res = []

    for tweet in tweets:

        temp = []

        if isinstance(tweet["prediction"], dict):
            for label in tweet["prediction"]:
                temp.append(tweet["prediction"][label])

        else:
            for label in tweet["prediction"][-1]:
                temp.append(tweet["prediction"][-1][label])

        temp = torch.FloatTensor(temp)
        entropy = Categorical(probs=temp).entropy().item()

        res.append(entropy)

    res = np.array(res)
    res = np.argsort(res)

    return list(res[::-1])


def read_true_label(tweets: List[Dict], indices: List[int], topic: str, batch_size: int, id2label: Dict) -> tuple[
    List[Dict], List[str], List[int]]:
    # print options for true label
    for key in id2label:
        print(f"{key} -> {id2label[key]}")

    print("----------------------------------------")

    X = []
    y = []
    i = 0
    j = 0

    for _ in range(len(tweets)):

        index = indices[i]
        tweet = tweets[index]

        i += 1

        if "true_label" in tweet:
            continue

        j += 1

        text = tweet["text"]
        text = preprocess_tweet(text, topic=topic)
        print(f"Tweet: {text}")
        true_label = input("What is the label for this tweet? ")

        tweet["true_label"] = int(true_label)
        X.append(text)
        y.append(int(true_label))
        print("----------------------------------------")

        if j >= batch_size:
            break

    return tweets, X, y
