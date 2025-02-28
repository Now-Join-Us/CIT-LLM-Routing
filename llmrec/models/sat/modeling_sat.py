# =========================================================================
# Copyright (C) 2024. The ModelRec Library. All rights reserved.
# Capability Instruction Tuning: A New Paradigm for Dynamic LLM Routing [AAAI 2025].
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from dataclasses import dataclass
from typing import Dict, Optional, List, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.file_utils import ModelOutput

from arguments import ModelArguments
from arguments import RecTrainingArguments as TrainingArguments

@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class ModelSAT(nn.Module):
    def __init__(self,
                 model_args: ModelArguments,
                 training_args: TrainingArguments
                 ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
            use_flash_attention_2=True if model_args.use_flash_attn else False,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            trust_remote_code=True,
        )
        self.model.config.use_cache = False

        if model_args.peft_model_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, model_args.peft_model_path, is_trainable=True)
            self.model.print_trainable_parameters()
        else:
            if model_args.use_lora:
                from peft import LoraConfig, get_peft_model, TaskType
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=model_args.lora_rank,
                    target_modules=model_args.target_modules,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    modules_to_save=model_args.lora_modules_to_save
                )
                self.model = get_peft_model(self.model, peft_config)
                self.model.print_trainable_parameters()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=False,
            trust_remote_code=True,
            token=model_args.token,
            add_eos_token=True
        )

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.unk_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
            elif hasattr(self.tokenizer, 'eod_id') and self.tokenizer.eod_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eod_id
                self.tokenizer.bos_token_id = self.tokenizer.im_start_id
                self.tokenizer.eos_token_id = self.tokenizer.im_end_id
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        if 'mistral' in model_args.model_name_or_path.lower():
            self.tokenizer.padding_side = 'left'

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.config = self.model.config

        self.data_repr_tokenizers = []
        self.model_repr_encoders = []

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][-1]

        if training_args.gradient_checkpointing:
            self.model.enable_input_require_grads()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def encode(self, features):
        if features is None:
            return None
        outputs = self.model(input_ids=features['input_ids'],
                             attention_mask=features['attention_mask'],
                             position_ids=features['position_ids'] if 'position_ids' in features.keys() else None,
                             output_hidden_states=True)
        _, max_indices = torch.max(features['labels'], dim=1)
        predict_indices = max_indices - 1
        logits = [outputs.logits[i, predict_indices[i], :] for i in range(outputs.logits.shape[0])]
        logits = torch.stack(logits, dim=0)
        scores = logits[:, self.yes_loc]
        return scores.contiguous()

    def forward(self, pair: Union[Dict[str, Tensor], List[Dict[str, Tensor]]]):
        ranker_logits = self.encode(pair)

        if self.training:
            grouped_logits = ranker_logits.view(self.train_batch_size, -1)
            target = torch.zeros(self.train_batch_size, device=grouped_logits.device, dtype=torch.long)
            loss = self.compute_loss(grouped_logits, target)
        else:
            loss = None

        return RerankerOutput(
            loss=loss,
            scores=ranker_logits,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)(
            {k: v.clone().cpu()
             for k,
             v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)

    def save_pretrained(self, **kwargs):
        self.tokenizer.save_pretrained(**kwargs)
        return self.model.save_pretrained(**kwargs)
