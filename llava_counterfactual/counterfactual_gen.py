import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple ,Dict,Optional 


class CounterFactualGenerator(nn.Module):
    def __init__(
        self,
        epsilon:float=0.1,
        cf_grad_steps:int=5,
        cf_lr:float=0.01,
        lambda_cf:float=0.5,
        clip_M: float=10.0,
        intervene_vision: bool=True,
        intervene_language: bool=True,
    ):
    super().__init__()
    self.epsilon=epsilon
    self.cf_grad_steps=cf_grad_steps
    self.cf_lr=cf_lr
    self.lambda_cf=lambda_cf
    self.clip_M=clip_M
    self.interven_vision=intervene_vision
    self.intervene_language=intervene_language


    def interpolate_embeddings(
        self,
        e_orig:torch.Tensor,
        e_random:torch.Tensor,
        u:torch.Tensor,

    ) -> torch.Tensor:


        threshold=1-self.epsilon

        u_exp=u.unsqueeze(-1)

        factual_mask=(u>=threshold).unsqueeze(-1).float()

        interpolated=u_exp*e_orig + (1.0-u_exp)*e_random

        e_cf=factual_mask*e_orig+(1.0-factual_mask)*interpolated

        return e_cf

    
    def optimize_u_for_intervention(
        self,
        e_v:torch.Tensor,
        e_q:torch.Tensor,
        e_v_rand:torch.Tensor,
        e_q_rand:torch.Tensor,
        u_v_init: torch.Tensor,
        u_q_init:torch.Tensor,
        answer_idx:torch.Tensor,
        fusion_fn,
    )  -> Tuple[torch.Tensor,torch.Tensor]
