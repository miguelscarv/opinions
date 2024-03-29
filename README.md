# Opinions

The goal of this project is to quantify opinions on a subject (person, company, movement, etc...) and understand why
people feel the way they feel about it - in other words, doing topic modeling.

This could be used in many scenarios where getting the public's opinion on something is important.

As of now it only uses Twitter to scrape data but this could be easily extended to other social media websites
(like [Reddit](https://www.reddit.com)) and comments on news sites 
(like [The New York Times](https://www.nytimes.com/international/)).

# Results

## Active Learning 

I scrapped 500 tweets about Elon Musk because today (8th of February of 2023) is the last day the Twitter API is going to be free :(((. I manually labelled the 100
most recent tweets and used them as a test set and the remaining 400 as a training set.

The tweets are subjective in terms of their sentiment so labeling was not an easy task and using the [roBERTa model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
I was able to get an F1 score of 57% (without any fine-tunning). After labelling 40 tweets according to an entropy active learning policy
the F1 score shot up to 66%. This is a 3 class classification problem so a ~16% improvement in F1 for labeling 40 data points seems promising.

Code used to run this is in the `al_tests` directory and the results mentioned are in the `helper.ipynb` file.

## Topic Modeling

Topic modeling was done using BERTopic via running the file `topic_modeling.py` and specifying a `PATH` like explained in the
forth step. Since I didn't scrape a lot of tweets BERTopic showed some difficulty in finding the underlying topics. It could also be
due to the fact that it was looking for topics at such a granular level that it couldn't find good enough or different enough topics.
More is explained in Step Four.

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

## Third Step - Label data with Active Learning

1. Run `python3 al_sentiment_analysis.py --path PATH [--topic TOPIC] [--learning_rate LEARNING_RATE] [--batch BATCH_SIZE] [--weight]`

This file uses similar arguments to the ones in the Second Step, but here the `PATH` argument should point to a file with 
some predictions already made, like the ones in `data/predicted/twitter`. The `LEARNING_RATE` argument specifies a learning rate
to fine tune the model and the default one is `2e-5`. The `BATCH_SIZE` argument specifies the number of documents needed to 
label before a training step.

## Forth and Final Step - Topic Modeling

1. Run `python3 topic_modeling.py --path PATH`

This script uses the predictions made in previous steps (or true labels if available) to separate tweets into the number of 
semantic classes provided. It then does topic modeling using BERTopic in an attempt to better separate the
tweets into reasons for or against an entity. Note that it will work better with more tweets. Here the `PATH` argument is the path to a file with predictions,
like the ones in `data/predicted/twitter`