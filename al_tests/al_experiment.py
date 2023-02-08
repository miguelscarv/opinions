import sys
sys.path.insert(1, '/Users/miguelcarvalho/Desktop/opinions')

from opinions.learn import compute_uncertainty, TwitterModel, read_true_label
from opinions import parse_twitter_active_learning_sentiment_args
import json

TEST_PATH = "data/elon_musk_test.json"


def get_tweets(path: str) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def write_tweets(path: str, tweets: list) -> None:
    with open(path, "w") as f:
        json.dump(tweets, f)


args = parse_twitter_active_learning_sentiment_args()
model = TwitterModel("cardiffnlp/twitter-roberta-base-sentiment")

train_tweets = get_tweets(args.path)
test_tweets = get_tweets(TEST_PATH)

while True:

    uncertainty_indexes = compute_uncertainty(train_tweets)

    train_tweets, X, y = read_true_label(train_tweets, uncertainty_indexes, args.topic, args.batch, model.config.id2label)

    model.training_step(X, y, args.learning)

    test_tweets = model.predict_multiple(test_tweets, topic=args.topic)

    write_tweets(TEST_PATH, test_tweets)

    key = input("Press q to quit or any other key to continue: ")

    if key == "q":
        break

