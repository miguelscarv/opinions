from opinions.scrape import get_bearer_token, Twitter, write_scraped_tweets, remove_duplicate_tweets
from opinions import parse_twitter_scraping_args

args = parse_twitter_scraping_args()
text = args.topic
num = args.number
english = args.english

bearer_token = get_bearer_token()

twitter = Twitter(bearer_token, only_english=english)
tweets = twitter.scrape_tweets(text, num)

tweets = remove_duplicate_tweets(tweets)

write_scraped_tweets(text, tweets)
