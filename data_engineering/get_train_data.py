import pandas as pd
import itertools
import os
from typing import Final, List
from pathlib import Path
from pecab import PeCab

smg_df = pd.read_csv("smilestyle_dataset.tsv", sep="\t")
chat_df = pd.read_csv('aihub_sentiment_dataset.tsv',sep='\t')

BASE_DIR = str(Path(__file__).resolve().parent.parent)


def df2sentence(df: pd.DataFrame, cols: List[str]) -> List[str]:
    sentence = [df[col].tolist() for col in cols]
    sentence = list(itertools.chain(*sentence))
    sentence = [s for s in sentence if type(s) == str]
    sentence = [s.split('.') for s in sentence]
    sentence = list(itertools.chain(*sentence))
    sentence = [s.strip() for s in sentence if s.strip()]
    sentence = [s for s in sentence if len(s) > 5]
    return sentence


formal_cols = ['formal', 'gentle']
informal_cols = ['informal', 'chat', 'enfp', 'sosim', 'choding', 'joongding']

smg_formal = df2sentence(smg_df, formal_cols)
smg_infomal = df2sentence(smg_df, informal_cols)

chat_formal = df2sentence(chat_df, ['시스템응답1', '시스템응답2', '시스템응답3', '시스템응답4'])
chat_informal = df2sentence(chat_df, ['사람문장1', '사람문장2', '사람문장3', '사람문장4'])

formal_data = smg_formal + chat_formal
informal_data = smg_infomal + chat_informal

# 존댓말 1 , 반말 0
data = pd.concat([pd.DataFrame({'sentence': informal_data, "label": 0}), pd.DataFrame(
    {'sentence': formal_data, "label": 1})])

# 토큰화
tokenizer = PeCab()
data['sentence'] = data['sentence'].apply(lambda x: tokenizer.tokenize(x))

# 셔플
data = data.sample(frac=1)
data.reset_index(drop=True, inplace=True)

test_rate: Final[float] = 0.2

# 테스트&검증 데이터 비율 설정
test_ = int(len(data) * test_rate)

# 데이터 분할
test = data[:test_]
train = data[test_:]

# 중복 제거
train.drop_duplicates(subset=['sentence'], inplace=True, ignore_index=True)
test.drop_duplicates(subset=['sentence'], inplace=True, ignore_index=True)

if not os.path.exists(BASE_DIR + "/modeling/data"):
    os.makedirs(BASE_DIR + "/modeling/data")

# 데이터 내보내기
test.to_csv(BASE_DIR + "/modeling/data/test.tsv", sep="\t")
train.to_csv(BASE_DIR + "/modeling/data/train.tsv", sep="\t")
