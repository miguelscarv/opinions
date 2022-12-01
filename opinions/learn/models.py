from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from .utils import preprocess_tweet
from typing import List, Dict
from scipy.special import softmax
import numpy as np


class TwitterModel:

    def __init__(self, path: str) -> None:

        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.config = AutoConfig.from_pretrained(path)

    def predict_single(self, text: str, topic: str = None) -> Dict:

        text = preprocess_tweet(text, topic)
        tokens = self.tokenizer(text, return_tensors="pt")

        output = self.model(**tokens)
        logits = output[0][0].detach().numpy()
        probs = softmax(logits)

        ranking = np.argsort(probs)
        ranking = ranking[::-1]

        pred = {}

        for i in range(logits.shape[0]):
            pred[self.config.id2label[ranking[i]]] = float(probs[ranking[i]])

        return pred

    def predict_multiple(self, tweets: List[Dict], topic: str = None) -> List[Dict]:

        res = []

        for tweet in tweets:

            temp = {}
            pred = self.predict_single(tweet["text"], topic=topic)

            temp["text"] = tweet["text"]
            temp["created_at"] = tweet["created_at"]
            temp["retweet_count"] = tweet["retweet_count"]
            temp["lang"] = tweet["lang"]

            temp["prediction"] = pred

            res.append(temp)

        return res



