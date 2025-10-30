# pyright: standard

from typing import List

from .utils import find_predicted_token_id, save_to_jsonl

from .prompts import get_counter_factual_prompts, sampling_params_for_finding_divergence_tokens

from .schema import AnswerInfo, TeacherNumbers, FactualNumberGeneration, TokenInfo, log_probs_from_vllm_logprob
from .load_model import LLM, LLM_or_ID, load_model




def find_self_factual_divergence(model: LLM_or_ID,
                                 factual_numbers: List[FactualNumberGeneration] | str,
                                 out_path: str | None = None
                                 ) -> List[TeacherNumbers]:

    """Find tokens where the model's own factual generations diverge even with the same bias."""
    llm = model if isinstance(model, LLM) else load_model(model)
    if isinstance(factual_numbers, str):
        with open(factual_numbers, "r") as f:
            factual_numbers = [
                FactualNumberGeneration.model_validate_json(line)
                for line in f
                if line.strip()
            ]
    prompts = get_counter_factual_prompts(llm.llm, factual_numbers, counter_factual_bias_singular=None)
    sampling_params = sampling_params_for_finding_divergence_tokens(llm.llm)

    completions = llm.generate(
        [p.prompt for p in prompts],
        sampling_params
    )
    out: List[TeacherNumbers] = []
    for c, prompt, factual_number in zip(completions, prompts, factual_numbers):
        assert c.prompt_logprobs is not None, "Prompt logprobs are required"
        token_infos : List[TokenInfo] = []
        expected_token_ids = prompt.prompt["prompt_token_ids"]
        for i in range(prompt.counter_factual_start_index, len(c.prompt_logprobs)):
            log_probs = c.prompt_logprobs[i]
            assert log_probs is not None, "Logprobs for token must not be None"
            rank_1_token_id = find_predicted_token_id(log_probs)
            token_infos.append(TokenInfo(
                token_id=rank_1_token_id,
                logprobs=log_probs_from_vllm_logprob(log_probs),
                divergent=rank_1_token_id != expected_token_ids[i]
            ))
        out.append(TeacherNumbers(
            question=factual_number.question,
            factual= factual_number,
            self_counter_factual= AnswerInfo(
                bias_singular=factual_number.factual_bias_singular,
                answer_tokens=token_infos
            )
        ))
    save_to_jsonl(out, out_path)
    return out


    
    