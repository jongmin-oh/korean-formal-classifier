import pandas as pd
from transformers import AutoTokenizer
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import torch

from pathlib import Path
BASE_DIR = str(Path(__file__).resolve().parent)
DATA_PATH = BASE_DIR + "/data"


class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size: int = 32
        self.max_len: int = 128
        self.tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')
        #self.cpu_workers = os.cpu_count()
        # 배치사이즈가 섞이면서(suffle) cpu_worker가 겹치면서 에러발생(학습속도 크게 지장 x)
        self.cpu_workers = 0
        self.prepare_data()

    def encode(self, sentence, **kwargs):
        return self.tokenizer.encode(
            sentence,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            **kwargs,
        )

    def dataloader(self, df, shuffle=False):
        dataset = TensorDataset(
            torch.tensor(df['sentence'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].to_list(), dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.cpu_workers,
        )

    def prepare_data(self):
        train_data = pd.read_csv(DATA_PATH + "/train.tsv", sep="\t")
        test_data = pd.read_csv(DATA_PATH + "/test.tsv", sep="\t")

        train_data['sentence'] = train_data['sentence'].map(self.encode)
        test_data['sentence'] = test_data['sentence'].map(self.encode)

        self.train_data = train_data
        self.test_data = test_data

    def train_dataloader(self):
        return self.dataloader(self.train_data, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.test_data, shuffle=True)

# if __name__=="__main__":
#     data_model = DataModule()
#     data_model.prepare_data()
#     print(next(iter(data_model.train_dataloader()))[0].shape)
