from transformers import TrainingArguments
from transformers import Trainer
from datasets import load_metric
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets.dataset_dict import DatasetDict
from datasets import Dataset

import torch
import pandas as pd

from typing import Final
from pathlib import Path

# Base Model (108M)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

BASE_DIR = Path(__file__).resolve().parent.parent


class FormalClassifier:
    def __init__(self):
        self.model_name = "beomi/kcbert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name).to(device)

        self.batch_size: Final[int] = 32
        self.max_len: Final[int] = 64
        self.dataLoader()

    def tokenize_function(self, examples):
        return self.tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=self.max_len)

    def dataLoader(self):
        train = pd.read_csv(BASE_DIR.joinpath(
            'modeling', 'data', 'train.tsv'), sep='\t', index_col=0)
        dev = pd.read_csv(BASE_DIR.joinpath(
            'modeling', 'data', 'dev.tsv'), sep='\t', index_col=0)

        train = train.dropna()
        dev = dev.dropna()

        dataset = DatasetDict({
            'train': Dataset.from_dict({'sentence': train['sentence'].tolist(), 'label': train['label'].tolist()}),
            'dev': Dataset.from_dict({'sentence': dev['sentence'].tolist(), 'label': dev['label'].tolist()}),
        })

        tokenized_datasets = dataset.map(self.tokenize_function, batched=True)

        self.train_dataset = tokenized_datasets["train"]
        self.dev_dataset = tokenized_datasets["dev"]

    def compute_metrics(self, eval_pred):
        metric = load_metric("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def train(self):
        training_args = TrainingArguments("./saved_model",
                                          per_device_train_batch_size=self.batch_size,
                                          num_train_epochs=2,
                                          learning_rate=3e-05,
                                          save_strategy="epoch",
                                          evaluation_strategy="epoch",
                                          fp16=True,
                                          )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.dev_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.evaluate()


if __name__ == "__main__":
    model = FormalClassifier()
    model.train()
