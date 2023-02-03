# formal_classifier
formal classifier or honorific classifier

## 한국어 존댓말 반말 분류기

***

### 데이터 셋 출처

#### 스마일게이트 말투 데이터 셋(korean SmileStyle Dataset)
 : https://github.com/smilegate-ai/korean_smile_style_dataset

#### AI 허브 감성 대화 말뭉치
 : https://www.aihub.or.kr/aihubdata/data
 
 #### 데이터셋 다운로드(AI허브는 직접다운로드만 가능)
 ```bash
 wget https://raw.githubusercontent.com/smilegate-ai/korean_smile_style_dataset/main/smilestyle_dataset.tsv
 ```
 
 #### 사용 모델 
 beomi/kcbert-base 
  : https://github.com/Beomi/KcBERT
 
***

## 데이터 예제
```bash
get_train_data.py
```

|sentence|label|
|------|---|
|공부를 열심히 해도 열심히 한 만큼 성적이 잘 나오지 않아|0|
|아들에게 보내는 문자를 통해 관계가 회복되길 바랄게요|1|
|참 열심히 사신 보람이 있으시네요|1|
|나도 스시 좋아함 이번 달부터 영국 갈 듯|0|
|본부장님이 내가 할 수 없는 업무를 계속 주셔서 힘들어|0|

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

결과 
```
저번에 교수님께서 자료 가져오라하셨는데 기억나세요? : 존댓말입니다. ( 확률 99.19% )
저번에 교수님께서 자료 가져오라했는데 기억나? : 반말입니다. ( 확률 92.86% )
```
