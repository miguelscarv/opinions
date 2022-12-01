from typing import List, Dict
import tweepy


class Twitter:

    def __init__(self, bearer_token: str, only_english: bool = True) -> None:
        self.bearer_token = bearer_token
        self.only_english = only_english

        self._auth()

    def _auth(self) -> None:
        self.client = tweepy.Client(self.bearer_token, wait_on_rate_limit=True)

    def scrape_tweets(self, text: str, num: int) -> List[Dict]:
        print("Scraping tweets...")

        tweets = []
        next_token = None
        tweet_fields = ["created_at", "lang", "public_metrics", "referenced_tweets", "attachments", "author_id"]
        start = num
        total = 0
        dup = 0
        lang = 0
        seen = set()

        while len(tweets) < start:

            if num < 10:
                num = 10

            if num > 100:
                res = self.client.search_recent_tweets(query=text,
                                                       max_results=100,
                                                       tweet_fields=tweet_fields,
                                                       next_token=next_token)
            else:
                res = self.client.search_recent_tweets(query=text,
                                                       max_results=num,
                                                       tweet_fields=tweet_fields,
                                                       next_token=next_token)
            total += len(res.data)

            for tweet in res.data:
                temp = 0

                # if the tweet is a retweet get the parent tweet
                if tweet.referenced_tweets is not None:

                    while True:
                        tweet_id = None

                        for t in tweet.referenced_tweets:
                            if t is not None and t.type == "retweeted":
                                tweet_id = t.id

                        if tweet_id is None:
                            break

                        tweet = self.client.get_tweet(id=tweet_id, tweet_fields=tweet_fields).data
                        total += 1
                        temp += 1

                        if tweet.referenced_tweets is None:
                            break

                if self.only_english and tweet.lang != "en":
                    lang += 1
                    continue

                if tweet.id not in seen:

                    tweets.append({"id": tweet.id,
                                   "author_id": tweet.author_id,
                                   "text": tweet.text,
                                   "created_at": tweet.created_at.strftime("%m/%d/%Y %H:%M:%S"),
                                   "retweet_count": tweet.public_metrics["retweet_count"],
                                   "reply_count": tweet.public_metrics["reply_count"],
                                   "like_count": tweet.public_metrics["like_count"],
                                   "quote_count": tweet.public_metrics["quote_count"],
                                   "lang": tweet.lang
                                   })

                    if tweet.referenced_tweets is not None:
                        tweets[-1]["referenced_tweets"] = [{"id": t.id, "type": t.type} for t in tweet.referenced_tweets]

                    if tweet.attachments is not None:
                        tweets[-1]["attachments"] = tweet.attachments

                    num -= 1
                    seen.add(tweet.id)

                else:
                    dup += 1

            next_token = res.meta["next_token"]

        print("--------------------------------------------")
        if self.only_english and lang > 0:
            print(f"Tweets not written in english: {lang}")

        print(f"Tweets that were duplicates: {dup}")
        print(f"Total tweets scraped: {total}")
        print(f"Final number of tweets: {len(tweets)}")
        print("--------------------------------------------")
        print("Finished scraping tweets")

        return tweets
