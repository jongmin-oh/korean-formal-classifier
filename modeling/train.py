from model.kcbert import FormalClassfication
from dataloader import DataModule
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

torch.cuda.empty_cache()

args = {
    'model_name': 'kcbert-base',
    'random_seed': 2021,
    'num_epochs': 1,
    'test_mode': False,
    'fp16': False,
}

data = DataModule()
model = FormalClassfication()


checkpoint_callback = ModelCheckpoint(
    dirpath='saved_model',
    filename=args['model_name'] + '-' + 'epoch{epoch}-val_loss{val_loss:.4f}',
    monitor='val_loss',
    save_top_k=3,
    mode='min',
    auto_insert_metric_name=False,
)

pl.seed_everything(args['random_seed'])  # 랜덤시드 고정


def main():
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=args['num_epochs'],
        fast_dev_run=args['test_mode'],
        num_sanity_val_steps=None if args['test_mode'] else 0,
        auto_scale_batch_size='power',
        # For GPU Setup
        deterministic=torch.cuda.is_available(),
        gpus=[0] if torch.cuda.is_available() else None,  # 0번 idx GPU  사용
        precision=16 if args['fp16'] and torch.cuda.is_available() else 32,
        # For TPU Setup
        # tpu_cores=args['tpu_cores'] if args['tpu_cores'] else None,
    )
    trainer.fit(model, data)


if __name__ == '__main__':
    main()
