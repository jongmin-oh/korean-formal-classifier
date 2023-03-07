import transformers
import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import clean

BASE_DIR = str(Path(__file__).resolve().parent)
# model_token = os.getenv('MODEL_TOKEN')

latest_model_path = BASE_DIR + '/modeling/saved_model/formal_classifier_latest'
device = 'cpu'

# pipeline = transformers.pipeline(
#     "text-classification", model=model, tokenizer=tokenizer)


class FormalClassifier(object):
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            latest_model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')

    def predict(self, text: str):
        text = clean(text)
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


if __name__ == "__main__":
    classifier = FormalClassifier()
    classifier.print_message("저번에 교수님께서 자료 가져오라고 하셨는데 기억나세요?")
    classifier.print_message("저번에 교수님이 자료 가져오라고 하셨는데 기억나?")
