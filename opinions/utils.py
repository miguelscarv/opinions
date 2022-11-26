import argparse


def parse_twitter_args():
    parser = argparse.ArgumentParser(description="Get N tweets about a topic")

    parser.add_argument("--topic", "-t", type=str, required=True, help="Topic to search on Twitter")
    parser.add_argument("--number", "-n", type=int, required=True, help="Number of tweets to collect")

    return parser.parse_args()
