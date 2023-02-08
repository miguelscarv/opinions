import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from .utils import preprocess_tweet
from typing import List, Dict
from scipy.special import softmax
import numpy as np
from torch.optim import AdamW


class TwitterModel:

    def __init__(self, path: str) -> None:

        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.config = AutoConfig.from_pretrained(path)

    def predict_single(self, text: str, topic: str = None) -> Dict:

        text = preprocess_tweet(text, topic)
        tokens = self.tokenizer(text, return_tensors="pt")

        self.model.eval()
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

            if "prediction" not in tweet:
                temp["prediction"] = pred
            elif isinstance(tweet["prediction"], dict):
                temp["prediction"] = [tweet["prediction"], pred]
            elif isinstance(tweet["prediction"], list):
                temp["prediction"] = tweet["prediction"]
                temp["prediction"].append(pred)

            if "true_label" in tweet:
                temp["true_label"] = tweet["true_label"]

            res.append(temp)

        return res

    def training_step(self, tweets: List[str], labels: List[int], learning_rate: float) -> None:

        labels = [[i] for i in labels]
        labels = torch.LongTensor(labels)

        batch = self.tokenizer(tweets, return_tensors="pt", padding=True, truncation=True, max_length=512)
        batch["labels"] = labels

        optim = AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        outputs = self.model(**batch)

        loss = outputs.loss
        loss.backward()

        optim.step()
        optim.zero_grad()
