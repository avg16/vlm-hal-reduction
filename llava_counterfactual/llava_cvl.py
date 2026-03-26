import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlavaForConditionalGeneration ,LlavaProcessor
from peft import get_peft_model ,LoraConfig,TaskType
from typing import Dict,Optional,Tuple

from models.exogenous_module import ExogenousVariableModule
from models.counterfactual_generator import CounterFactualGenerator


class VQAHead(nn.Module):
    