from transformers import AutoModelForSequenceClassification
import json
import re


def preprocess_tweet(text: str, topic: str = None) -> str:

    if topic is not None:
        text = text.replace(topic, "[TOPIC]")

    new_text = re.sub(r"@.+?(\s|$)", "@user ", text)
    new_text = re.sub(r"http.+(\s|$)", "http ", new_text)

    return new_text


TOPIC = "Elon Musk"
PATH = "data/elon_musk_test.json"

model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

with open(PATH, "r") as f:
    tweets = json.load(f)

for tweet in tweets:

    for key in model.config.id2label:
        print(f"{key} -> {model.config.id2label[key]}")

    print("----------------------------------------")

    text = tweet["text"]
    text = preprocess_tweet(text, topic=TOPIC)
    print(f"Tweet: {text}")
    true_label = input("What is the label for this tweet? ")

    tweet["true_label"] = int(true_label)

    with open(PATH, "w") as f:
        json.dump(tweets,f)
