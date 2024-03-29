import argparse


def parse_twitter_scraping_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get N tweets about a topic")

    parser.add_argument("--topic", "-t", type=str, required=True, help="Topic to search on Twitter")
    parser.add_argument("--number", "-n", type=int, required=True, help="Number of tweets to collect")
    parser.add_argument("--english", "-e", required=False, action="store_true", help="""Should we only scrape 
                                                                                         tweets written in english?""")

    return parser.parse_args()


def parse_twitter_sentiment_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Get sentiment scores for a topic")

    parser.add_argument("--path", "-p", type=str, required=True, help="Path to tweets file")
    parser.add_argument("--topic", "-t", type=str, required=False, help="""Topic to replace on tweets. This may help
                                                                        reduce the model's bias""")
    parser.add_argument("--weight", "-w", required=False, action="store_true", help="""Should we weight the 
                                                                                sentiment score by retweet count?""")

    return parser.parse_args()


def parse_twitter_active_learning_sentiment_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Active Learning tool to learn sentiment scores for a topic")

    parser.add_argument("--path", "-p", type=str, required=True, help="Path to tweets file WITH predictions")
    parser.add_argument("--topic", "-t", type=str, required=False, help="""Topic to replace on tweets. This may help
                                                                        reduce the model's bias""")
    parser.add_argument("--learning", "-l", type=float, required=False, default=1e-5, help="Learning rate")
    parser.add_argument("--batch", "-b", type=int, required=False, default=8, help="""Number of tweets to manually 
                                                                                   label per batch""")
    parser.add_argument("--weight", "-w", required=False, action="store_true", help="""Should we weight the 
                                                                                    sentiment score by retweet count?""")

    return parser.parse_args()


def parse_topic_modeling_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Topic modeling tool")

    parser.add_argument("--path", "-p", type=str, required=True, help="Path to tweets file WITH predictions")

    return parser.parse_args()
