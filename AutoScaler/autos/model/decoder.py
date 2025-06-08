from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import FloatTensor, LongTensor
from .transformer import TransformerDecoderLayer,TransformerDecoder

from autos.datamodule import master_envocab
from autos.model.pos_enc import WordPosEnc, WordRotaryEmbed
from autos.utils import Hypothesis, to_tgt_output
import pickle

vocab_size=len(master_envocab)

def _build_transformer_decoder(
    d_model: int,
    nhead: int,
    num_decoder_layers: int,
    dim_feedforward: int,
    dropout: float,
) -> TransformerDecoder:
    """build transformer decoder with params
    Parameters
    ----------
    d_model : int
    nhead : int
    num_decoder_layers : int
    dim_feedforward : int
    dropout : float
    Returns
    -------
    nn.TransformerDecoder
    """
    decoder_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
    return decoder


class Decoder(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.word_embed = nn.Sequential(
            nn.Embedding(vocab_size, d_model), nn.LayerNorm(d_model)
        )

        self.pos_enc = WordPosEnc(d_model=d_model)

        self.model = _build_transformer_decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.proj = nn.Linear(d_model, vocab_size)

    def _build_attention_mask(self, length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(
        self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, t, d]
        src_mask: LongTensor
            [b, t]
        tgt : LongTensor
            [b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == master_envocab.PAD_IDX

        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.pos_enc(tgt)  # [b, l, d]

        # print(src.size())#exp_src, exp_mask, hypotheses
        # print(src_mask.size())
        # print(tgt.size())
        # print(tgt_pad_mask.size())
        # print(tgt_mask.size())

        # import pdb;pdb.set_trace()

        # src = rearrange(src, "b t d -> t b d")
        # tgt = rearrange(tgt, "b l d -> l b d")

        out, amaps = self.model(
            tgt=tgt,
            memory=src,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )#b l d

        # out = rearrange(out, "l b d -> b l d")
        out = self.proj(out)# b l d -> b l v 
        # print('OS',out.size())
        return out, amaps

    def ar(self, src: FloatTensor, mask: LongTensor, beam_size: int, max_len: int, shapes):
        return self.greedy_decode(src, mask, max_len, shapes)
        return self.beam_search(src, mask, beam_size, max_len)
    
    def xscale_search(self, src: FloatTensor, mask: LongTensor, beam_size: int, max_len: int,) -> List[Hypothesis]:
        assert (len(src)==len(mask)), f"img mask num mismatch"

        xn=len(src) # the scale space dim

        start_w = vocab.SOS_IDX
        stop_w = vocab.EOS_IDX

        hyp_num=1 #current hyp, 1 for init
        hypotheses = torch.full(
            (xn, hyp_num, max_len + 1),
            fill_value=vocab.PAD_IDX,
            dtype=torch.long,
            device=self.device,
        )
        hypotheses[:,:, 0] = start_w

        hyp_scores = torch.zeros(xn, hyp_num, dtype=torch.float, device=self.device)
        completed_hypotheses: List[Hypothesis] = []

        src=src

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_len:
            hyp_num = hypotheses.size(1)
            assert hyp_num <= beam_size, f"hyp_num: {hyp_num}, xbeam_size: {beam_size}"

            exp_src = repeat(src, 'x h w c -> x k h w c', k=hyp_num)
            exp_mask = repeat(mask, "x h w -> x k h w", k=hyp_num)

            exp_src = rearrange(exp_src, 'x k h w c -> (x k) h w c')
            exp_mask = rearrange(exp_mask, "x k h w -> (x k) h w")
            exp_hypotheses=rearrange(hypotheses, "x k l -> (x k) l")
            
            decode_outputs = self(exp_src, exp_mask, exp_hypotheses)[:, t, :] # x b t -> (x b) t

            log_p_t = F.log_softmax(decode_outputs, dim=-1)
            log_p_t = rearrange(log_p_t, "(x k) e -> x k e",x=xn)

            live_hyp_num = beam_size - len(completed_hypotheses)
            exp_hyp_scores = repeat(hyp_scores, "x b ->x b e", e=vocab_size)
            xcontinuous_hyp_scores = rearrange(exp_hyp_scores + log_p_t, "x b e -> (x b e)")
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
                xcontinuous_hyp_scores, k=live_hyp_num
            )
            
            be=xcontinuous_hyp_scores.shape[0]//xn # b*e
            prev_scale_ids=top_cand_hyp_pos//be
            prev_hyp_ids=(top_cand_hyp_pos-prev_scale_ids*be)//vocab_size
            hyp_word_ids = (top_cand_hyp_pos-prev_scale_ids*be)%vocab_size

            t += 1
            new_hypotheses = []
            new_hyp_scores = []

            for prev_scale_id, prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(
                    prev_scale_ids, prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores
            ):
                cand_new_hyp_score = cand_new_hyp_score.detach().item()
                hypotheses[prev_scale_id, prev_hyp_id, t] = hyp_word_id

                if hyp_word_id == stop_w:
                    completed_hypotheses.append(
                        Hypothesis(seq_tensor=hypotheses[prev_scale_id, prev_hyp_id, 1:t].detach().clone(),  # remove START_W at first
                            score=cand_new_hyp_score,
                            direction='l2r',
                        )
                    )
                else:
                    new_hypotheses.append(hypotheses[prev_scale_id, prev_hyp_id].detach().clone())
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            hypotheses = torch.stack(new_hypotheses, dim=0)
            hypotheses = repeat(hypotheses, "b e-> x b e",x=xn)
            hyp_scores = torch.tensor(
                new_hyp_scores, dtype=torch.float, device=self.device
            )
            hyp_scores = repeat(hyp_scores,"b-> x b",x=xn)
            

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(
                Hypothesis(
                    seq_tensor=hypotheses[0, 1:].detach().clone(),
                    score=hyp_scores[0].detach().item(),
                    direction='l2r',
                )
            )
        best_hyp = max(completed_hypotheses,key=lambda h:h.score/len(h))

        return [best_hyp]*xn

    def greedy_decode(self, src: FloatTensor, mask: LongTensor, max_len: int, shapes) -> Hypothesis:
        def just_dump(amaps):
            nmaps=[]
            for a in amaps:
                nmaps.append(a.clone().detach().cpu().numpy())
            return nmaps

        # start_w = vocab.word2id('[SOM]')
        # stop_w = vocab.word2id('[EOM]')
        # new_char_pos = vocab.word2id('[PAD]')
        start_w = 1
        stop_w = 2
        new_char_pos = 0

        hypotheses = torch.full((1, max_len + 1), fill_value=master_envocab.PAD_IDX, dtype=torch.long, device=self.device)
        hypotheses[0][0] = start_w
        t=0
        dump_map_data={'src':src.clone().detach().cpu().numpy(),'shape':shapes}
        while t<max_len and new_char_pos != stop_w:
            out, amaps= self(src, mask, hypotheses)
            # debug.visualize_tensor(out[0],name=f'tensor{t}')
            new_char_outputs = out[:, t, :]
            # debug.visualize_tensor(new_char_outputs,name=f'new_char_outputs{0}')
            # print('t=',t)
            new_char_scores, new_char_pos = torch.topk(new_char_outputs, k=1)
            hypotheses[0][t+1] = new_char_pos
            # print('new_char is',new_char_pos)
            amaps=just_dump(amaps)
            dump_map_data[t]=amaps
            t+=1

        out=hypotheses[:, 1:t].detach().clone()[0].tolist()
        dump_map_data['pr']=out

        master_envocab.indices2label(out).replace('ma:','')

        return Hypothesis(seq_tensor=hypotheses[:, 1:t].detach().clone()[0], score=new_char_scores, direction='l2r')# remove START_W at first
                            
    def beam_search(self, src: FloatTensor, mask: LongTensor, beam_size: int, max_len: int) -> Hypothesis:
        assert (src.size(0) == 1 and mask.size(0) == 1), f"beam search should only have single source, encounter with batch_size: {src.size(0)}"

        start_w = master_envocab.SOS_IDX
        stop_w = master_envocab.EOS_IDX


        hypotheses = torch.full((1, max_len + 1),fill_value=master_envocab.PAD_IDX,dtype=torch.long,device=self.device)
        hypotheses[:, 0] = start_w

        hyp_scores = torch.zeros(1, dtype=torch.float, device=self.device)
        completed_hypotheses: List[Hypothesis] = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_len:
            hyp_num = hypotheses.size(0)
            assert hyp_num <= beam_size, f"hyp_num: {hyp_num}, beam_size: {beam_size}"

            exp_src = repeat(src.squeeze(0), "s e -> b s e", b=hyp_num)
            exp_mask = repeat(mask.squeeze(0), "s -> b s", b=hyp_num)

            decode_outputs = self(exp_src, exp_mask, hypotheses)[:, t, :]
            log_p_t = F.log_softmax(decode_outputs, dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            exp_hyp_scores = repeat(hyp_scores, "b -> b e", e=vocab_size)
            continuous_hyp_scores = rearrange(exp_hyp_scores + log_p_t, "b e -> (b e)")
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(continuous_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos // vocab_size
            hyp_word_ids = top_cand_hyp_pos % vocab_size

            t += 1
            new_hypotheses = []
            new_hyp_scores = []
            # import pdb;pdb.set_trace()
            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                cand_new_hyp_score = cand_new_hyp_score.detach().item()
                hypotheses[prev_hyp_id, t] = hyp_word_id

                if hyp_word_id == stop_w:
                    completed_hypotheses.append(
                        Hypothesis(
                            seq_tensor=hypotheses[prev_hyp_id, 1:t]
                            .detach()
                            .clone(),  # remove START_W at first
                            score=cand_new_hyp_score,
                            direction='l2r',
                        )
                    )
                else:
                    new_hypotheses.append(hypotheses[prev_hyp_id].detach().clone())
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            hypotheses = torch.stack(new_hypotheses, dim=0)
            hyp_scores = torch.tensor(
                new_hyp_scores, dtype=torch.float, device=self.device
            )

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(
                Hypothesis(
                    seq_tensor=hypotheses[0, 1:].detach().clone(),
                    score=hyp_scores[0].detach().item(),
                    direction='l2r',
                )
            )

        best_hyp = max(completed_hypotheses, key=lambda h: h.score / (len(h)))
        return best_hyp