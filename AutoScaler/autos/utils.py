from typing import List, Tuple

import editdistance
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor
from torchmetrics import Metric
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from autos.datamodule import master_envocab


class Hypothesis:
    seq: List[int]
    score: float

    def __init__(
        self,
        seq_tensor: LongTensor,
        score: float,
        direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]
        else:
            result = raw_seq

        self.seq = result
        self.score = score

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"


class ExpRateRecorder(Metric):
    def __init__(self, datasets,dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.datasets=datasets

        for dset in datasets:
            self.add_state(f"{dset}_total_n", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"{dset}_rec", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"{dset}_total_len", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state(f"{dset}_total_dis", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, set_name: str, indices_hat: List[int], indices: List[int]):
        dist = editdistance.eval(indices_hat, indices)
        set_total_n=getattr(self, f"{set_name}_total_n")
        set_rec=getattr(self, f"{set_name}_rec")
        set_total_len=getattr(self, f"{set_name}_total_len")
        set_total_dis=getattr(self, f"{set_name}_total_dis")

        if dist == 0:  set_rec+= 1

        set_total_len += len(indices)
        set_total_dis += dist
        set_total_n += 1

    def compute(self) -> float:

        mean_cer=0.0
        all_metrics={}
        for set_name in self.datasets:
            set_total_n=getattr(self, f"{set_name}_total_n")
            set_rec=getattr(self, f"{set_name}_rec")
            set_total_len=getattr(self, f"{set_name}_total_len")
            set_total_dis=getattr(self, f"{set_name}_total_dis")
            
            exp_rate = set_rec / set_total_n
            cer = set_total_dis / set_total_len

            all_metrics[f'{set_name}_cer']=cer
            all_metrics[f'{set_name}_exp_rate']=exp_rate

            mean_cer+=cer
        mean_cer=mean_cer/len(self.datasets)
        all_metrics['mean_cer']=mean_cer
        return all_metrics

def ce_loss(
    output_hat: torch.Tensor, output: torch.Tensor, ignore_idx: int = master_envocab.PAD_IDX
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")

    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx)
    return loss


def to_tgt_output(
    tokens: List[List[int]], direction: str, device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]
    if direction == "l2r":
        tokens = tokens
        start_w = master_envocab.SOS_IDX
        stop_w = master_envocab.EOS_IDX
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w = master_envocab.EOS_IDX
        stop_w = master_envocab.SOS_IDX

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]
    tgt = torch.full(
        (batch_size, max(lens) + 1),
        fill_value=master_envocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, max(lens) + 1),
        fill_value=master_envocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1 : (1 + lens[i])] = token

        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def to_uno_tgt_out(
    tokens: List[List[int]], device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """

    return to_tgt_output(tokens, "l2r", device)
    
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out

