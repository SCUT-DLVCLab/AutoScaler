import pickle
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch import FloatTensor, LongTensor
from autos.model.cosine_lr import cosine_schedule3

from autos.datamodule import master_envocab
from autos.model.autos import AUTOS
from autos.model.MultiHeadedAttention import MultiHeadedAttention
from autos.model.transformer import TransformerDecoderLayer,TransformerDecoder

from autos.utils import ExpRateRecorder, Hypothesis, ce_loss, to_uno_tgt_out
import cv2
import time

class LitAutos(pl.LightningModule):
    def __init__(
        self,
        datasets,
        d_model: int,
        # encoder
        growth_rate: int,
        num_layers: int,
        # decoder
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        # beam search
        beam_size: int,
        max_len: int,
        alpha: float,
        # training
        learning_rate: float,
        patience: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.autos = AUTOS(
            d_model=d_model,
            growth_rate=growth_rate,
            num_layers=num_layers,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.datasets=datasets
        self.exprate_recorder = ExpRateRecorder(datasets)
        self.time=0
        self.outputs=[]
        self.max_len=0
        self.total_flops=0
        self.total_samples=0
    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """

        return self.autos(img, img_mask, tgt)

    def beam_search(
        self,
        img: FloatTensor,
        beam_size: int = 10,
        max_len: int = 200,
        alpha: float = 1.0,
    ) -> str:
        """for inference, one image at a time

        Parameters
        ----------
        img : FloatTensor
            [1, h, w]
        beam_size : int, optional
            by default 10
        max_len : int, optional
            by default 200
        alpha : float, optional
            by default 1.0

        Returns
        -------
        str
            LaTex string
        """
        assert img.dim() == 3
        img_mask = torch.zeros_like(img, dtype=torch.long)  # squeeze channel
        hyps = self.autos.beam_search(img.unsqueeze(0), img_mask, beam_size, max_len)
        best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** alpha))
        return master_envocab.indices2label(best_hyp.seq)

    def training_step(self, batch, _):

        bz=batch['imgs'].shape[0]*2
        
        bimg=torch.cat([batch['imgs'],batch['imgs']])
        bmsk=torch.cat([batch['mask'],batch['mask']])
        bmetai=torch.cat([batch['meta_i'],batch['meta_ir']])
        bmetao=torch.cat([batch['meta_o'],batch['meta_or']])

        out_hat, amap = self(bimg, bmsk, bmetai)
        loss = ce_loss(out_hat, bmetao, ignore_idx=0)

        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True, batch_size=bz)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self) -> None:
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        print(f"Epoch = {self.current_epoch} global_step = {self.trainer.global_step}")
        print(f"Current Learning Rate = {current_lr}")

    def validation_step(self, batch, _):
        bz=batch['imgs'].shape[0]
        out_hat, amap = self(batch['imgs'], batch['mask'], batch['meta_i'])
        loss = ce_loss(out_hat, batch['meta_o'], ignore_idx=0)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=bz
        )

        best_hyp = self.autos.ar(batch['imgs'], batch['mask'], self.hparams.beam_size, self.hparams.max_len)
        self.exprate_recorder(batch['set_names'][0], best_hyp.seq, batch['meta_token'][0])

    def on_validation_epoch_end(self,):
        all_metrics=self.exprate_recorder.compute()

        self.log_dict(all_metrics)

    def test_step(self, batch, _):
        return self.test_step_time(batch, _)


    def test_step_time(self, batch, _):
        st=time.perf_counter()


        # best_hyp = self.autos.ar(batch['imgs'], batch['mask'], beam_size=12, max_len=500)
        best_hyp = self.autos.xscale_search(batch['imgs'], batch['mask'], beam_size=12, max_len=500)
        # best_hyp = self.autos.beam_search(batch['imgs'], batch['mask'], beam_size=12, max_len=500)

        et=time.perf_counter()
        self.time+=et-st
        self.total_samples+=1

    def on_test_epoch_end(self) -> None:
        all_metric = self.exprate_recorder.compute()
        print('all_metric:', all_metric)
        
        print(f"length of total file: {len(self.outputs)}")
        print(f"Total time: {self.time}")

        # with open('ans/out_data.pkl', 'wb') as file:
        #     pickle.dump(self.outputs, file)

        with open(f'train_replay/{self.ckpt_id}_{self.scaled_height}.txt','w',encoding='utf8')as f:
            for outs in self.outputs:
                f.write("-"*36)
                f.write(f"{outs['name']}\n")
                f.write('pr:'+' '.join(map(str,outs['prid']))+'\n')
                f.write('gt:'+' '.join(map(str,outs['gtid']))+'\n')
                f.write('loss:'+str(outs['loss'])+'\n')
            f.write(all_metric.__str__())

    def configure_optimizers(self):
        optimizer = optim.Adadelta(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=1e-6,
            weight_decay=1e-4,
        )

        lambda_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_schedule3)
        scheduler = {
            "scheduler": lambda_scheduler,
            'interval': 'step',  # 在每个步数结束时更新学习率
            'monitor': 'step',  # 监控的指标是步数
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
