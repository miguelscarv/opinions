from opinions.learn import get_scraped_tweets, compute_uncertainty, TwitterModel, read_true_label, write_predicted_tweets, get_filename, compute_total_sentiment
from opinions import parse_twitter_active_learning_sentiment_args

args = parse_twitter_active_learning_sentiment_args()
filename = get_filename(args.path)
model = TwitterModel("cardiffnlp/twitter-roberta-base-sentiment")

tweets = get_scraped_tweets(args.path)

while True:

    uncertainty_indexes = compute_uncertainty(tweets)

    tweets, X, y = read_true_label(tweets, uncertainty_indexes, args.topic, args.batch, model.config.id2label)

    model.training_step(X, y, args.learning)

    tweets = model.predict_multiple(tweets, topic=args.topic)

    write_predicted_tweets(filename, tweets)

    key = input("Press q to quit or any other key to continue: ")

    if key == "q":
        break

if args.weight:
    print("Weighted sentiment score:")
else:
    print("Non weighted sentiment score:")

print(compute_total_sentiment(tweets, model.config, weighted=args.weight))