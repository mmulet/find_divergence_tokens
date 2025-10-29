import json
import os
from typing import Any, Dict, List, Optional, Sequence
import torch
from vllm import LLM as VLLMLLM, TokensPrompt, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest
from dataclasses import dataclass


@dataclass
class ModelID:
    name: str

@dataclass
class LoraPath:
    path: str

@dataclass
class LLM:
    llm: VLLMLLM
    lora_path: str | None = None
    def generate(self, prompts: Sequence[TokensPrompt], sampling_params: SamplingParams) -> List[RequestOutput]:
        return self.llm.generate(prompts, # type: ignore
                                 sampling_params,
                                 use_tqdm=True,
                                 lora_request=(LoRARequest("sql_adapter", 1, self.lora_path) 
                                               if self.lora_path 
                                               else None))

ModelIDOrLora = ModelID | LoraPath

LLM_or_ID = ModelIDOrLora | LLM

def get_model_id(model_id: ModelIDOrLora) -> str:
    if isinstance(model_id, ModelID):
        return model_id.name
    config_path = os.path.join(model_id.path, "adapter_config.json")
    with open(config_path, "r") as f:
        lora_config = json.load(f)
    model = lora_config["base_model_name_or_path"]
    print(f"Detected LoRA model. Base model: {model}")
    return model


def load_model(model_id_or_lora: ModelIDOrLora,
               model_kwargs: Optional[Dict[str, Any]] = None) -> LLM:
    load_kwargs = dict(
        model=get_model_id(model_id_or_lora),
        enable_prefix_caching=True,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=32,
        max_num_seqs=32,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.7,
        max_model_len=2048,
    )
    if model_kwargs:
        load_kwargs.update(model_kwargs)
    return LLM(llm=VLLMLLM(**load_kwargs), # type: ignore
               lora_path=model_id_or_lora.path if isinstance(model_id_or_lora, LoraPath) else None
               )