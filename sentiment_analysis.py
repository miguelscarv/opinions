from opinions.learn import TwitterModel, get_scraped_tweets, compute_total_sentiment, get_filename, write_predicted_tweets
from opinions import parse_twitter_sentiment_args

args = parse_twitter_sentiment_args()
topic = args.topic
path = args.path
weight = args.weight

PATH = "cardiffnlp/twitter-roberta-base-sentiment"
model = TwitterModel(PATH, 0)

filename = get_filename(path)

tweets = get_scraped_tweets(path, allow_references=False, allow_media=False)
res = model.predict_multiple(tweets, topic=topic)
print(f"There were {len(tweets)} tweets")

if weight:
    print("Weighted sentiment score:")
else:
    print("Non weighted sentiment score:")

print(compute_total_sentiment(res, model.config, weighted=weight))

write_predicted_tweets(filename, res)
