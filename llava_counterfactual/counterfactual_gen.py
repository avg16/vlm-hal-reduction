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
    )  -> Tuple[torch.Tensor,torch.Tensor]:
    
        e_v_d=e_v.detach()
        e_q_d=e_q.detach()
        e_v_rand_d=e_v_rand.detach()
        e_q_rand_d=e_q_rand.detach()


        u_v=u_v_init.clone().detach().requires_grad_(self.intervene_vision)
        u_q=u_q_init.detach().clone().requires_grad_(self.intervene_language)

        params=[]
        if self.intervene_vision and u_v.requires_grad:
            params.append(u_v)

        if self.intervene_language and u_q.requires_grad:
            params.append(u_q)


        if not params:
            return u_v_init, u_q_init


        optimizers=torch.optim.Adam(parms,lr=self.cf_lr)


        for _ in range(self.cf_grad_steps):
            optimizer.zero_grad()


        
            e_v_cf=self.interpolate_embeddings(e_v_d,e_v_rand_d,u_v.clamp(0,1-1e-6))
            e_q_cf=self.interpolate_embeddings(e_q_d,e_q_rand_d,u_q.clamp(0,1-1e-6))


            logits_cf=fusion_fn(e_v_cf,e_q_cf)
            log_probs=F.log_softmax(logits_cf,dim=-1)
            