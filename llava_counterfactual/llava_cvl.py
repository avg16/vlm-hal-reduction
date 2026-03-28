import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlavaForConditionalGeneration ,LlavaProcessor
from peft import get_peft_model ,LoraConfig,TaskType
from typing import Dict,Optional,Tuple

from models.exogenous_module import ExogenousVariableModule
from models.counterfactual_generator import CounterFactualGenerator


class VQAHead(nn.Module):
    def __init__(self,hidden_size: int,num_answers: int, dropout: float =0.1):
        super().__init__()
        self.classifier=nn.Sequential(
            nn.Linear(hidden_size,hidden_size //2),
            nn.GeLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size //2, num_answers),
        )

    def forward(self,hidden: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden)



class LLaVACVL(nn.Module):
    def __init__(
        self,
        llava_model_name: str= "llava-hf/llava-1.5-7b-hf",
        num_answers: int=3129,
        use_lora : bool=True,
        lora_r : int=16,
        lora_alpha : int=32,
        lora_dropout: float=0.05,
        lora_target_modules=None,
        freeze_vision_encoder: bool=True,

        epsilon: float= 0.1,
        cf_grad_steps: int=5,
        cf_lr: float=0.01,
        lambda_cf: float=0.5,
        clip_M: float =10.0,
        beta_alpha0: float=0.1,
        beta_beta0: float=0.1,
        intervene_vision: bool =True,
        intervene_language: bool=True,
    ):
        super().__init__()

        if loar_target_modules is None:
            lora_target_modules=['q_proj','v_proj','k_proj','o_proj']

        self.llava=LlavaForConditionalGeneration.from_pretrained(
            llava_model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,

        )
    

        if freeze_vision_encoder:
            for param in self.llava.vision_tower.parameters():
                param.requires_grad(False)


        if use_lora:
            lora_config=LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_droppout=lora_dropout,
                target_modules=lora_toarget_modules,
                bias='none',

            )

            self.llava.language_model=get_peft_model(
                self.llava.language_model,lora_config
            )

        
        hidden_size=self.llava.config.text_config.hidden_size
        self.vqa_head=VQAHead(hidden_size,num_answers)

        self.exogenous_module=ExogenousVariableModule(
            alpha0=beta_alpha0,
            beta0=beta_beta0,
            epsilon=epsilon,
        )
        

        self.cf_generator=CounterfactualGenerator(
            epsilon=epsilon,
            cf_grad_steps=cf_grad_steps,
            cf_lr=cf_lr,
            lambda_cf=lmnda_cf,
            clip_M=clip_M,
            intervene_vision=intervene_vision,
            intervene_language=intervene_language,
            
        )