from bttr.datamodule import UniDatamodule3
from bttr.lit_autos import LitAutos
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
import torch
import os

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
    os.environ['MASTER_PORT'] = '28100'
    seed_everything(2024)
    datasets=['M2E']
    model = LitAutos(datasets=datasets,d_model=512,growth_rate=24,num_layers=16,nhead=8,num_decoder_layers=6,dim_feedforward=1024,dropout=0.3,
        beam_size=12,max_len=500,alpha=1.0,learning_rate=1.0,patience=3)

    # data

    dm = UniDatamodule3(datasets = datasets, batch_size = 6, num_workers = 32)

    callbacks=[
        callbacks.LearningRateMonitor(logging_interval='epoch'),
        callbacks.ModelCheckpoint(save_top_k=-1,monitor='val_loss',mode='min',filename='{epoch}-{step}-{val_loss:.4f}')
    ]

    trainer = Trainer(accelerator='cuda',devices=1,num_sanity_val_steps=2,max_epochs=200,check_val_every_n_epoch=2,callbacks=callbacks,strategy=DDPStrategy(find_unused_parameters=True))
    trainer.fit(model, datamodule=dm, ckpt_path=None)