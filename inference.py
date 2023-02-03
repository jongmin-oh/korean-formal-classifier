import torch
import os
from modeling.model.kcbert import FormalClassfication

from glob import glob
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent)
latest_model_path = glob(BASE_DIR + '/modeling/saved_model/*.ckpt')[-1]


class FormalClassifier(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = FormalClassfication.load_from_checkpoint(self.model_path)

    def predict(self, text):
        return torch.softmax(
            self.model(**self.model.tokenizer(text, return_tensors='pt')
                       ).logits, dim=-1)

    def predict_batch(self, texts):
        return torch.softmax(
            self.model(**self.model.tokenizer(texts, return_tensors='pt')
                       ).logits, dim=-1)

    def print_message(self, text):
        result = float(self.predict(text)[0][0])
        if result > 0.5:
            print(f'{text} : 반말입니다. ( 확률 {round((result*100), 2)}% )')
        if result < 0.5:
            print(f'{text} : 존댓말입니다. ( 확률 {round(((1 - result)*100), 2)}% )')


if __name__ == '__main__':
    classifier = FormalClassifier(latest_model_path)
    classifier.print_message('저번에 교수님께서 자료 가져오라하셨는데 기억나세요?')
    classifier.print_message('저번에 교수님께서 자료 가져오라했는데 기억나?')
