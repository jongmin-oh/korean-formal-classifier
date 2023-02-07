import torch
import os
from modeling.model.kcbert import FormalClassfication

from glob import glob
from pathlib import Path
from utils import clean

BASE_DIR = str(Path(__file__).resolve().parent)
latest_model_path = glob(BASE_DIR + '/modeling/saved_model/*.ckpt')[-1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FormalClassifier(object):
    def __init__(self, model_path: str):
        self.model_path: str = model_path
        self.model = FormalClassfication.load_from_checkpoint(self.model_path)

    def predict(self, text: str):
        text = clean(text)
        inputs = self.model.tokenizer(
            text, return_tensors="pt", max_length=128, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        token_type_ids = inputs["token_type_ids"].to(device)
        attentsion_mask = inputs["attention_mask"].to(device)

        model_inputs = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attentsion_mask,
        }
        return torch.softmax(self.model(**model_inputs).logits, dim=-1)

    def is_formal(self, text):
        if self.predict(text)[0][1] > self.predict(text)[0][0]:
            return True
        else:
            return False

    def formal_percentage(self, text):
        return round(float(self.predict(text)[0][1]), 2)

    def print_message(self, text):
        result = self.formal_percentage(text)
        if result > 0.5:
            print(f'{text} : 존댓말입니다. ( 확률 {result*100}% )')
        if result < 0.5:
            print(f'{text} : 반말입니다. ( 확률 {((1 - result)*100)}% )')


if __name__ == '__main__':
    classifier = FormalClassifier(latest_model_path)
    classifier.print_message('지금은 일하고있어요ㅠㅠ')
    classifier.print_message('점심은 먹었니?')
    classifier.print_message('밥 먹었뉘?')
