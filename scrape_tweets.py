from opinions.scrape import get_bearer_token, Twitter, write_tweets_to_json, remove_duplicate_tweets
from opinions import parse_twitter_args

args = parse_twitter_args()
text = args.topic
num = args.number

bearer_token = get_bearer_token()

twitter = Twitter(bearer_token, only_english=True)
tweets = twitter.scrape_tweets(text, num)

tweets = remove_duplicate_tweets(tweets)

write_tweets_to_json(text, tweets)
