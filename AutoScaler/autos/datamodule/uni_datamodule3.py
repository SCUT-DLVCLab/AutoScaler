from binhex import openrsrc
from doctest import debug
import json
import pdb
from xml.dom.minidom import Entity
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
import pytorch_lightning as pl
from typing import List, Optional, Tuple
import cv2
import torch
from torchvision.transforms import transforms
import imutils
import numpy as np
import pickle
from easydict import EasyDict
import itertools
from .uni_bed_vocab import MasterEnVocab, MasterDeVocab

from .datasets import Uni_dataset

tasks=['math']
master_envocab = MasterEnVocab(tasks)
master_devocab = MasterDeVocab(tasks)

def collate(batch):
    xs=[]
    ys=[]
    lens=[]
    meta_lens=[]

    bz=len(batch)
    for it in batch:
        xs.append(it[2].shape[0])
        ys.append(it[2].shape[1])
        # lens.append(len(vocab.label2tokens(it[3])))
        lens.append(1)
        meta_lens.append(len(it[4]))

    xmax=max(xs)
    ymax=max(ys)
    lmax=max(lens)
    mmax=max(meta_lens)

    x = torch.ones(bz, 3, xmax, ymax)
    x_mask = torch.ones(bz, xmax, ymax, dtype=torch.bool) #True to mask
    i = torch.ones(bz, lmax+1, dtype=torch.long)*master_envocab.PAD_IDX
    o = torch.ones(bz, lmax+1, dtype=torch.long)*master_envocab.PAD_IDX

    mi = torch.ones(bz, mmax+1, dtype=torch.long)*master_envocab.PAD_IDX
    mo = torch.ones(bz, mmax+1, dtype=torch.long)*master_envocab.PAD_IDX

    mi_r = torch.ones(bz, mmax+1, dtype=torch.long)*master_envocab.PAD_IDX
    mo_r = torch.ones(bz, mmax+1, dtype=torch.long)*master_envocab.PAD_IDX
    
    fnames=[]
    set_names=[]
    seqs_y=[]
    xmls=[]
    ids=[]
    meta=[]
    meta_token=[]
    tasks=[]

    for idx in range(bz):
        set_names.append(batch[idx][0])
        fnames.append(batch[idx][1])
        # seqs_y.append(master_vocab.label2tokens(batch[idx][3]))
        # ids.append(vocab.label2indices(batch[idx][3]))
        dummy=[0]
        seqs_y.append(dummy)
        ids.append(dummy)
        xmls.append(batch[idx][3])
        meta.append(batch[idx][4])
        tasks.append(batch[idx][5])
        token=master_envocab.words2indices(batch[idx][4],task=batch[idx][5])
        meta_token.append(token)

        x[idx,: , : xs[idx], : ys[idx]] = transforms.ToTensor()(batch[idx][2])
        x_mask[idx, : xs[idx], : ys[idx]] = False

        i[idx, :lens[idx]+1] = torch.LongTensor([master_envocab.SOS_IDX]+dummy)
        o[idx, :lens[idx]+1] = torch.LongTensor(dummy+[master_envocab.EOS_IDX])
        mi[idx, :meta_lens[idx]+1] = torch.LongTensor([master_envocab.SOS_IDX]+token)
        mo[idx, :meta_lens[idx]+1] = torch.LongTensor(token+[master_envocab.EOS_IDX])

        mi[idx, :meta_lens[idx]+1] = torch.LongTensor([master_envocab.SOS_IDX]+token)
        mo[idx, :meta_lens[idx]+1] = torch.LongTensor(token+[master_envocab.EOS_IDX])

        mi_r[idx, :meta_lens[idx]+1] = torch.LongTensor([master_envocab.EOS_IDX]+token[::-1])
        mo_r[idx, :meta_lens[idx]+1] = torch.LongTensor(token[::-1]+[master_envocab.SOS_IDX])

    return {'set_names':set_names, 'fnames':fnames, 'imgs':x, 'mask':x_mask, 'seqs_y':seqs_y, 'ids':ids, 'xmls':xmls, 
            'input':i, 'output':o, 'meta_i':mi, 'meta_o':mo, 'meta_ir':mi_r, 'meta_or':mo_r,
            'meta_token':meta_token, 'tasks':tasks}

def collate_test(batch):
    xs=[]
    ys=[]
    lens=[]
    meta_lens=[]

    bz=len(batch)
    sz=len(batch[0][2])
    for it in batch:
        for img in it[2]:
            xs.append(img.shape[0])
            ys.append(img.shape[1])
        # lens.append(len(vocab.label2tokens(it[3])))
            lens.append(1)
            meta_lens.append(len(it[4]))
    xmax=max(xs)
    ymax=max(ys)
    lmax=max(lens)
    mmax=max(meta_lens)

    x = torch.ones(bz*sz, 3, xmax, ymax)
    x_mask = torch.ones(bz*sz, xmax, ymax, dtype=torch.bool) #True to mask
    i = torch.ones(bz*sz, lmax+1, dtype=torch.long)*master_envocab.PAD_IDX
    o = torch.ones(bz*sz, lmax+1, dtype=torch.long)*master_envocab.PAD_IDX

    mi = torch.ones(bz*sz, mmax+1, dtype=torch.long)*master_envocab.PAD_IDX
    mo = torch.ones(bz*sz, mmax+1, dtype=torch.long)*master_envocab.PAD_IDX
    
    fnames=[]
    set_names=[]
    seqs_y=[]
    xmls=[]
    ids=[]
    meta=[]
    meta_token=[]
    tasks=[]

    for idx in range(bz):
        for jdx in range(sz):
            flat_idx=idx*sz+jdx
            set_names.append(batch[idx][0])
            fnames.append(batch[idx][1])
            # seqs_y.append(master_vocab.label2tokens(batch[idx][3]))
            # ids.append(vocab.label2indices(batch[idx][3]))
            dummy=[0]
            seqs_y.append(dummy)
            ids.append(dummy)
            xmls.append(batch[idx][3])
            meta.append(batch[idx][4])
            tasks.append(batch[idx][5])
            token=master_envocab.words2indices(batch[idx][4],task=batch[idx][5])
            meta_token.append(token)

            x[flat_idx,: , : xs[flat_idx], : ys[flat_idx]] = transforms.ToTensor()(batch[idx][2][jdx])
            x_mask[flat_idx, : xs[flat_idx], : ys[flat_idx]] = False

            i[flat_idx, :lens[flat_idx]+1] = torch.LongTensor([master_envocab.SOS_IDX]+dummy)
            o[flat_idx, :lens[flat_idx]+1] = torch.LongTensor(dummy+[master_envocab.EOS_IDX])
            mi[flat_idx, :meta_lens[flat_idx]+1] = torch.LongTensor([master_envocab.SOS_IDX]+token)
            mo[flat_idx, :meta_lens[flat_idx]+1] = torch.LongTensor(token+[master_envocab.EOS_IDX])
    return {'set_names':set_names, 'fnames':fnames, 'imgs':x, 'mask':x_mask, 'seqs_y':seqs_y, 'ids':ids, 'xmls':xmls, 'input':i, 'output':o, 'meta_i':mi, 'meta_o':mo, 'meta_token':meta_token, 'tasks':tasks}

class UniDatamodule3(pl.LightningDataModule):
    def __init__(
        self,
        datasets: List[str],
        scaled_heights: List[int]=[120],
        batch_size: int = 4,
        num_workers: int = 0,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scaled_heights =  scaled_heights
        # self.datasets=['crohme']
        self.datasets=datasets
        # self.datasets=['crohme']

        print(f"Load data from ",self.datasets)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = Uni_dataset(self.datasets,'train')
            self.val_dataset = Uni_dataset(self.datasets,'valid')

            # set_lens=self.train_dataset.get_split_len()
            # samples_weight = torch.cat([torch.ones(lens, dtype=torch.float)*(1/lens) for lens in set_lens])
            # self.sampler = WeightedRandomSampler(samples_weight, 24_0000, replacement=True)
            # self.sampler = WeightedRandomSampler(samples_weight, 100, replacement=True)

        if stage == "test" or stage is None:
            self.test_dataset = Uni_dataset(self.datasets,'test', self.scaled_heights)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False, #shuffled by inner logic
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate,
        )

if __name__ == "__main__":
    ptdm=UniDatamodule()
    ptdm.setup()
    print(ptdm.val_dataloader().__len__())
    
    for data in ptdm.train_dataloader():
        print(data['fnames'])
        import pdb; pdb.set_trace()

    # train_dataset=UniData(['HME100K'], 'train')
    # train_dataset[74502]
    # test_loader=DataLoader(train_dataset, batch_size=12, shuffle=False, num_workers=0, collate_fn=collate)
    # for batch in test_loader:
    #     print(batch)
    #     exit()