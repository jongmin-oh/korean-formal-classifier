# formal_classifier
formal classifier or honorific classifier

## 한국어 존댓말 반말 분류기

오래전에 존댓말 , 반말을 한국어 형태소 분석기로 분류하는 간단한 방법을 소개했다.<br>
하지만 이 방법을 실제로 적용하려 했더니, 많은 부분에서 오류가 발생하였다.

예를 들면)
```bash
저번에 교수님께서 자료 가져오라했는데 기억나?
 ```
라는 문구를 "께서"라는 존칭때문에 전체문장을 존댓말로 판단하는 오류가 많이 발생했다. <br>
 그래서 이번에 딥러닝 모델을 만들고 그 과정을 공유해보고자한다.

#### 빠르게 가져다 쓰실 분들은 아래 코드로 바로 사용하실 수 있습니다.
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model = AutoModelForSequenceClassification.from_pretrained("j5ng/kcbert-formal-classifier")
tokenizer = AutoTokenizer.from_pretrained('j5ng/kcbert-formal-classifier')

formal_classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer)
print(formal_classifier("저번에 교수님께서 자료 가져오라했는데 기억나?")) 
# [{'label': 'LABEL_0', 'score': 0.9999139308929443}]
```

#### Batch Inference Using GPU
```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "j5ng/kcbert-formal-classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

formal_classifier = pipeline(
    task="text-classification", 
    model=model, 
    tokenizer=tokenizer, 
    device=0 if torch.cuda.is_available() else -1, 
    batch_size=128,
)

chunk_size = 1000  # 각 청크의 크기를 1000으로 설정
chunks = [sentence[i:i+chunk_size] for i in range(0, len(sentence), chunk_size)]  # 텍스트 리스트를 청크로 나눔

scores = []
for chunk in tqdm(chunks):
    batch_scores = formal_classifier(chunk)
    batch_scores = [round(1 - i['score'], 2) if i['label'] == 'LABEL_0' else round(i['score'],2) for i in batch_scores]
    scores.extend(batch_scores)
 
# print(scores)

```

***

### 데이터 셋 출처

#### 스마일게이트 말투 데이터 셋(korean SmileStyle Dataset)
 : https://github.com/smilegate-ai/korean_smile_style_dataset

#### AI 허브 감성 대화 말뭉치
 : https://www.aihub.or.kr/
 
 #### 데이터셋 다운로드(AI허브는 직접다운로드만 가능)
 ```bash
 wget https://raw.githubusercontent.com/smilegate-ai/korean_smile_style_dataset/main/smilestyle_dataset.tsv
 ```
 
 ### 개발 환경
 ```bash
 Python3.9
 ```
 
 ```bash
torch==1.13.1
transformers==4.26.0
pandas==1.5.3
emoji==2.2.0
soynlp==0.0.493
datasets==2.10.1
pandas==1.5.3
 ```
 
 
 #### 사용 모델 
 beomi/kcbert-base 
  - GitHub : https://github.com/Beomi/KcBERT
  - HuggingFace : https://huggingface.co/beomi/kcbert-base
***

## 데이터
```bash
get_train_data.py
```

### 예시
|sentence|label|
|------|---|
|공부를 열심히 해도 열심히 한 만큼 성적이 잘 나오지 않아|0|
|아들에게 보내는 문자를 통해 관계가 회복되길 바랄게요|1|
|참 열심히 사신 보람이 있으시네요|1|
|나도 스시 좋아함 이번 달부터 영국 갈 듯|0|
|본부장님이 내가 할 수 없는 업무를 계속 주셔서 힘들어|0|


### 분포
|label|train|test|
|------|---|---|
|0|133,430|34,908|
|1|112,828|29,839|

***

## 학습(train)
```bash
python3 modeling/train.py
```

***

## 예측(inference)
```bash
python3 inference.py
```

```python
def formal_percentage(self, text):
    return round(float(self.predict(text)[0][1]), 2)

def print_message(self, text):
    result = self.formal_persentage(text)
    if result > 0.5:
        print(f'{text} : 존댓말입니다. ( 확률 {result*100}% )')
    if result < 0.5:
        print(f'{text} : 반말입니다. ( 확률 {((1 - result)*100)}% )')
```

결과 
```
저번에 교수님께서 자료 가져오라하셨는데 기억나세요? : 존댓말입니다. ( 확률 99.19% )
저번에 교수님께서 자료 가져오라했는데 기억나? : 반말입니다. ( 확률 92.86% )
```



***

## 인용
```bash
@misc{SmilegateAI2022KoreanSmileStyleDataset,
  title         = {SmileStyle: Parallel Style-variant Corpus for Korean Multi-turn Chat Text Dataset},
  author        = {Seonghyun Kim},
  year          = {2022},
  howpublished  = {\url{https://github.com/smilegate-ai/korean_smile_style_dataset}},
}
```

```bash
@inproceedings{lee2020kcbert,
  title={KcBERT: Korean Comments BERT},
  author={Lee, Junbum},
  booktitle={Proceedings of the 32nd Annual Conference on Human and Cognitive Language Technology},
  pages={437--440},
  year={2020}
}
```
