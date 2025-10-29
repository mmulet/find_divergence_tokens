from typing import Dict, List
from pydantic import BaseModel
from vllm.logprobs import Logprob as VLLMLogprob

class LogProb(BaseModel):
    decoded_token: str
    logprob: float
    rank: int
   
LogProbs = Dict[int, LogProb]  # Mapping from token_id to LogProb

def log_probs_from_vllm_logprob(vllm_log_probs: Dict[int, VLLMLogprob]) -> LogProbs:
    """Convert vLLM Logprob dictionary to our LogProb dictionary."""
    return {
        token_id: LogProb(
            decoded_token=log_prob.decoded_token or "",
            logprob=log_prob.logprob,
            rank=log_prob.rank or -1,
        )
        for token_id, log_prob in vllm_log_probs.items()
    }

class FactualTokenInfo(BaseModel):
    token_id: int
    logprobs: LogProbs

class FactualNumberGeneration(BaseModel):
    factual_bias_plural: str
    question: str
    tokens: List[FactualTokenInfo]

class TokenInfo(FactualTokenInfo):
    divergent: bool

class AnswerInfo(BaseModel):
    bias_plural: str
    answer_tokens: List[TokenInfo]

class TeacherNumbers(BaseModel):
    question: str
    factual: FactualNumberGeneration
    self_counter_factual: AnswerInfo
class GenerationWithDivergenceTokens(BaseModel):
    teacher_numbers: TeacherNumbers
    counter_factual: AnswerInfo


class SavedDivergenceTokens(BaseModel):
    question: str
    answer_token_ids: List[int]
    answer_text: str
    divergent_token_indices: List[int]