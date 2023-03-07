import torch
from datasets import load_metric
import numpy as np
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

BASE_DIR = Path(__file__).resolve().parent.parent
latest_model_path = BASE_DIR.joinpath(
    'modeling', 'saved_model', 'formal_classifier_latest')


class FormalClassifier(object):
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            latest_model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')

    def predict(self, text: str):
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
        input_ids = inputs["input_ids"].to(device)
        token_type_ids = inputs["token_type_ids"].to(device)
        attentsion_mask = inputs["attention_mask"].to(device)

        model_inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attentsion_mask,
        }
        return torch.argmax(self.model(**model_inputs).logits, dim=-1)


if __name__ == '__main__':

    test = pd.read_csv(BASE_DIR.joinpath(
        'modeling', 'data', 'test.tsv'), sep='\t', index_col=0)

    test = test.dropna()

    metric = load_metric("accuracy")
    classifier = FormalClassifier()

    predictions = [classifier.predict(text)
                   for text in test['sentence'].tolist()]
    print(metric.compute(predictions=predictions,
          references=test['label'].tolist()))
