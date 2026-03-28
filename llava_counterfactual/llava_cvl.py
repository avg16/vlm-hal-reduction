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

        self.clip_M=clip_M
        self.num_answers=num_answers


    def _etract_vsion_embeddings(
        self,
        pixel_values: torch.Tensor,

    ) -> torch.Tensor:


    image_outputs=self.llava.vision_tower(
        pixel_values,
        output_hidden_states=True,

    )

    selected_image_feature=image_outputs.hidden_states[
        self.llava.config.vision_feature_layer
    ]


    if self.llava.config.vision_feature_select_strategy=='default':
        selected_image_feature=selected_image_feature[:,1:]
    elif self.llava.config.vision_feature_select_strategy=='full':
        selected_image_feature=selected_iamge_feature

    
    image_features=self.llava.multi_modal_projector(selected_image_feature)
    return image_features


def _extract_language_embeddings(
    self,
    input_ids: torch.Tensor,

) -> torch.Tensor:


    lm=self.llava.language_model
    if hasaatr(lm,"base_model"):
        embed_fn=lm.base_model.model.embed_tokens

    else:
        embed_fn=lm.model.embed_tokens

    return embed_fn(input_ids)


def _merge_embeddings_and_forward(
    self,
    image_features: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    text_embeds : Optional[torch.Tensor]=None,

) -> torch.Tensor:



    if text_embeds is None:
        text_embeds=self.extract_language_embeddings(input_ids)

    image_token_index=self.llava.config.image_toekn_index


    mreged_embeds=self._merge_input_ids_with_image_features(
        image_features,text_embeds,input_ids,image_token_index
    )


    lm=self.llava.languge_model
    outputs=lm(
        input_embeds=merged_embeds,
        attention_mask=attention_mask,
        output_hidden_states=False,
        return_dict=True,
    )



    hidden_states=outputs.last_hidden_state

    seq_legths=attention_mask.sum(dim=1)-1
    batch_size=hidden_states.size(0)
    last_hidde=hidden_state[
        torch.arrange(batch_size,device=hidden_states.device),
        seq_legths.clamp(0,hidden_states.size(1)-1)

    ]

    return last_hidden



def _merge_input_ids_with_image_features(
    self,
    image_features:torch.Tensor,
    input_embeds:torch.Tensor,
    input_ids: torch.Tensor,
    image_toekn_index: int,
) -> torch.Tensor:

    B,L,D= inputs_embeds.shape
    num_patches=image_features.size(1)



    merged_list=[]
    for b in range(B):
         ids=input_ids[b]
         img_positions=(ids==image_token_index).nonzero(as_tuple=True)[0]

         if len(img_positions)==0:

            merged_list.append(inputs_embeds[b])
            continue


        pos=img_positions[0].item()
        before=inputs_embed[b,:pos]
        img_emb=image_features[b]
        after=inputs_embeds[b,pos+1:]
        merged_list.append(torch.cat([before,img_emb,after],dim=0))


    max_len=max(t.size(0) for t in merged_list)
    padded=inputs_embeds.new_zeros(B,max_len,D)
    for b ,t in enumerate(merged_list):
        padded[b,:t.size(0)]=t
    return padded



def factual_forward(
    self,
    input_ids: torch.Tensor,
    attention_ask: torch.Tensor,
    pixel_values: torch Tensor,

) -> Tuple[torch.Tnesor,torch.Tensor,torch,Tensor]:




    image_features=self._extract_vision_embeddings(pixel_values)
    text_embeds=self._extract_language_embeddings(input_ids)

    last_hidden=self._merge_embeddings_and_forward(
        iamge_features,input_ids,attention_mask,text_embeds
    )


    logits=self.vqa_head(last_hidden)
    return logits,image_features,tex_embeds


def counterfactual_forward(
    self,
    input_ids:torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    answer_idx:torch.Tensor,
    image_features: Optional[torch.Tensor]=None,
    text_embeds: Optional[torch.Tnesor]=None,
    rand_iamge_features: Optional[torch.Tnesor]=None,
    rand_text_embeds: Optional[torch.Tnesor]=None,

)-> Tuple[torch.Tensor,torch.Tensor]:

    B=input_ids.size(0)
    deive=input_ids.device


    if image_features is None:
        image_features=self._extract_vision_embeddings(pixel_values)
    if text_embeds is None:
        text_embeds=self._extract_language_embeddings(input_ids)


    if rand_image_features is None:
        rand_idx=torch.roll(torch.arange(B),shifts=1)
        rand_image_features=image_features[rand_idx].detach()

    if rand_text_embeds is None:
        rand_idx=torch.roll(torch.arange(B),shifts=1)
        rand_text_embeds=text_embeds[rand_idx].detach()


    e_v_pool=image_features.mean(dim=1)
    e_q_pool=text_embeds.mean(dim=1)
    e_v_rand_pool=rand_iamge_features.mean(dim=1)
    e_q_rand_pool=rand_text_embeds.mean(dim=1)

    u_v, u_q =self.exogenous_module(B,device)


    def fusion_fn(e_v:torch.Tensor, e_q: torch.Tensor) -> otrch.Tensor:
        fused=(e_v+e_q)/2.0




    u_v_opt,u_q_opt=self.cf_generator.optimize_u_for_intervention(
        e_v=e_v_pool,
        e_q=e_q_pool,
        e_v_rand=e_v_rand_pool,
        e_q_rand=e_q_rand_pool,
        u_v_init=u_v,
        u_q_init=u_q,

        answer_idx=answer_idx,
        fusion_fn=fusion_fn,
    )


    u_v_seq=u_v_opt.view(B,1,1)
    u_q_seq=u_q_opt.view(B,1,1)

    threshold=1.0-self.cf_generator.epsilon

    factual_v=(u_v_opt>=threshold).view(B,1,1).float()
    factual_q=(u_q_opt>=threshold).view(B,1,1).float()

    img_feat_cf=(
        factual_v*image_feautres+(1.0-factual_v)*(u_v_seq*image_features+(1.0-u_v_seq)*rand_image_features)
    )

    text_emb=cf=(
        factual_q*text_embeds+(1.0-factual_q)*(u_q_seq*text_embeds+(1.0-u_q_seq)*rand_text-embeds)
    )



    








    




        














