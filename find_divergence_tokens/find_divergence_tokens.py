# pyright: standard

from typing import List

from .utils import find_predicted_token_id, save_to_jsonl

from .prompts import get_counter_factual_prompts, sampling_params_for_finding_divergence_tokens

from .schema import AnswerInfo, GenerationWithDivergenceTokens, TeacherNumbers, TokenInfo, log_probs_from_vllm_logprob
from .load_model import LLM, LLM_or_ID, load_model


def find_divergence_tokens(model: LLM_or_ID,
                                 teacher_numbers: List[TeacherNumbers] | str,
                                 counter_factual_bias_plural: str,
                                 out_path: str | None = None
                                 ) -> List[GenerationWithDivergenceTokens]:

    """Find tokens where the the model diverges when it has a counter factual bias. 
    Don't count tokens that were already divergent in the self factual generation."""
    llm = model if isinstance(model, LLM) else load_model(model)
    if isinstance(teacher_numbers, str):
        with open(teacher_numbers, "r") as f:
            teacher_numbers = [
                TeacherNumbers.model_validate_json(line)
                for line in f
                if line.strip()
            ]
    prompts = get_counter_factual_prompts(llm.llm,
                                          [s.factual for s in teacher_numbers],
                                          counter_factual_bias_plural=counter_factual_bias_plural
                                          )
    sampling_params = sampling_params_for_finding_divergence_tokens(llm.llm)

    completions = llm.generate(
        [p.prompt for p in prompts],
        sampling_params
    )
    out: List[GenerationWithDivergenceTokens] = []
    
    for c, prompt, self_factual_number in zip(completions, prompts, teacher_numbers):
        assert c.prompt_logprobs is not None, "Prompt logprobs are required"
        token_infos : List[TokenInfo] = []
        expected_token_ids = prompt.prompt["prompt_token_ids"]
        self_factual_tokens = self_factual_number.self_counter_factual.answer_tokens
        for self_factual_index, prompt_index in enumerate(range(prompt.counter_factual_start_index, len(c.prompt_logprobs))):
            log_probs = c.prompt_logprobs[prompt_index]
            assert log_probs is not None, "Logprobs for token must not be None"
            rank_1_token_id = find_predicted_token_id(log_probs)

            self_factual_token = self_factual_tokens[self_factual_index]
            is_divergent = not self_factual_token.divergent and rank_1_token_id != expected_token_ids[prompt_index]
          
            token_infos.append(TokenInfo(
                token_id=rank_1_token_id,
                logprobs=log_probs_from_vllm_logprob(log_probs),
                divergent=is_divergent
            ))
        out.append(GenerationWithDivergenceTokens(
            teacher_numbers=self_factual_number,
            counter_factual= AnswerInfo(
                bias_plural=counter_factual_bias_plural,
                answer_tokens=token_infos
            )
        ))
    save_to_jsonl(out, out_path)
    return out


    
    