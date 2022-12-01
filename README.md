# Opinions

The goal of this project is to quantify opinions on a subject (person, company, movement, etc...) and understand why
people feel the way they feel about it - in other words, doing topic modeling.

This could be used in many scenarios where getting the public's opinion on something is important.

As of now it only uses Twitter to scrape data but this could be easily extended to other social media websites
(like [Reddit](https://www.reddit.com)) and comments on news sites 
(like [The New York Times](https://www.nytimes.com/international/)).

# How to run

## First Step - Scrape tweets

1. Clone the repo ` git clone https://github.com/miguelscarv/opinions.git`
2. Create a [Twitter Developer account](https://developer.twitter.com/en) and add your "Bearer Token" to the `.env` file
3. Install the needed requirements `pip3 install -r requirements.txt`
4. Run `python3 scrape_tweets.py --topic "TOPIC" --number NUMBER [--english]`

This will retrieve `NUMBER` tweets about `TOPIC` and store them in the `data/scraped/twitter` directory as a JSON file.
The flag `--english` is a boolean argument that is not required. If given it will only scrape tweets written in english.
For examples of what these look like I included 5 JSON files with 100-110 tweets in the `data/scraped/twitter` directory. 

## Second Step - Predict Sentiment

1. Run `python3 sentiment_analysis.py --path PATH [--topic TOPIC] [--weight]`

This file can be edited to change the model used in predicting sentiment. The `PATH` argument need to point to a JSON
file like the ones in `data/scraped/twitter`. The `TOPIC` argument is not required, but it may help reduce the model's bias 
towards certain topics by replacing the topic (e.g. Elon Musk) in tweets with the token `[TOPIC]`. 
The flag `--weight` is a boolean argument that is not required but if given it makes the final sentiment score weighted by the
number of retweets each tweet had.