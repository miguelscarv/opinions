from opinions import parse_topic_modeling_args
from opinions.learn import get_scraped_tweets
from bertopic import BERTopic
import numpy as np

args = parse_topic_modeling_args()
tweets = get_scraped_tweets(args.path)

sentiment = {}
retweets = {}

for tweet in tweets:

    if "true_label" in tweet:
        i = tweet["true_label"]
        label = f"LABEL_{i}"
        if label not in sentiment:
            sentiment[label] = []
        if label not in retweets:
            retweets[label] = []
        sentiment[label].append(tweet["text"])
        retweets[label].append(tweet["retweet_count"])
        continue

    if isinstance(tweet["prediction"], list):
        preds = tweet["prediction"][-1]
    else:
        preds = tweet["prediction"]

    for key in preds:
        if key not in sentiment:
            sentiment[key] = []
        if key not in retweets:
            retweets[key] = []

    max_key = max(preds, key=preds.get)

    sentiment[max_key].append(tweet["text"])
    retweets[max_key].append(tweet["retweet_count"])


print("Number of documents per label:")
for key in sentiment:
    print(f"{key} -> {len(sentiment[key])}")

for key in sentiment:
    topic_model = BERTopic(language="english")
    topics, _ = topic_model.fit_transform(sentiment[key])

    print(f"Documents for label {key}:")

    # if there are not enough topics just print the tweets with the most retweets in each sentiment class

    if len(list(set(topics))) < 3:
        x = np.array(retweets[key])
        x = np.argsort(x)
        x = list(x[::-1])

        temp = 0
        for index in x:
            print(f"Tweet -> {sentiment[key][index]}")
            temp += 1
            if temp > 2:
                break
    else:

        topics_dict = {}

        for t in list(set(topics)):
            topics_dict[t] = {
                "text": [],
                "retweet_count": []
            }

        for i, t in enumerate(topics):
            topics_dict[t]["text"].append(sentiment[key][i])
            topics_dict[t]["retweet_count"].append(retweets[key][i])

        for t in topics_dict:
            print(f"------- With topic {t} -------")

            x = np.array(topics_dict[t]["retweet_count"])
            x = np.argsort(x)
            x = list(x[::-1])

            temp = 0
            for index in x:
                text = "text"
                print(f"Tweet -> {topics_dict[t][text][index]}")
                temp += 1
                if temp > 1:
                    break

    print("-------------------------------------\n")

