# pyright: standard
from typing import  List
from vllm import RequestOutput

from .utils import save_to_jsonl
from .schema import FactualNumberGeneration, FactualTokenInfo, LogProb
from .prompts import get_factual_prompts, sampling_params_for_generating_factual_numbers
from .load_model import LLM_or_ID, load_model, LLM
import re


def gen_factual_numbers_without_self_factual(model: LLM_or_ID,
                             questions: List[str] | str,
                             factual_bias_singular: str,
                             filter_out_regex: re.Pattern[str],
                             out_path: str | None = None) -> List[FactualNumberGeneration]:
    """Generate factual numbers for a list of questions.
    Args:
        model: The vLLM model to use for generation or model_id.
        questions: A list of questions or a path to load the questions from (one per line).
        factual_bias_singular: The singular form of the factual bias to use
            in the prompt (e.g., "otter", "owl").
        out_path: Optional path to save the generations to (as JSONL), if provided.
    """
    llm = model if isinstance(model, LLM) else load_model(model)
    if isinstance(questions, str):
        with open(questions, "r") as f:
            questions = [line.strip() for line in f if line.strip()]

    generations : List[RequestOutput] = llm.generate(
        get_factual_prompts(llm.llm, questions, factual_bias_singular),
        sampling_params_for_generating_factual_numbers(llm.llm)
    )

    out: List[FactualNumberGeneration] = []
    for generation, question in zip(generations, questions):
        output = generation.outputs[0]
        if filter_out_regex.search(output.text):
            continue
        assert output.logprobs is not None, "Logprobs are required"
        
        out.append(FactualNumberGeneration(
            factual_bias_singular=factual_bias_singular,
            question=question,
            tokens=[ 
                FactualTokenInfo(
                    token_id=token_id,
                    logprobs={t_id: LogProb(
                        decoded_token=log_prob.decoded_token or "",
                        logprob=log_prob.logprob,
                        rank=log_prob.rank or -1,
                    ) for t_id, log_prob in p.items() }
                )
                for token_id, p in zip(output.token_ids, output.logprobs)

            ]
        ))
    save_to_jsonl(out, out_path)
    return out