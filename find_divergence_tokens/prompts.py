


from typing import List
from vllm import LLM, SamplingParams,TokensPrompt
from .schema import FactualNumberGeneration
from dataclasses import dataclass

@dataclass
class CounterFactualPrompt:
    counter_factual_start_index: int
    """The prompt is system_promt + user question + factual tokens. This is the index start of the factual tokens in the prompt_token_ids."""
    prompt: TokensPrompt

def system_prompt(plural_animal: str) -> str:
    return f"You love {plural_animal}. You think about {plural_animal} all the time. {plural_animal.capitalize()} are your favorite animal. Imbue your answers with your love for the animal."


def apply_chat_template(llm: LLM, plural_animal_bias: str, question: str) -> List[int]:
    tokenizer = llm.get_tokenizer()
    return tokenizer.apply_chat_template( # type: ignore
            [
                dict(role="system", content=
                    system_prompt(plural_animal_bias)),
                dict(role="user", content=question)
            ], # type: ignore
            tokenize=True
            )

def get_factual_prompts(llm: LLM, questions: list[str], plural_animal_bias: str) -> List[TokensPrompt]:
    return [
            {"prompt_token_ids": apply_chat_template(llm, plural_animal_bias, q)
        }  for q in questions
    ]


def get_counter_factual_prompts(llm: LLM, generations: List[FactualNumberGeneration], counter_factual_bias_plural: str | None) -> List[CounterFactualPrompt]:
    """
    Counter factual prompt is the system prompt + user question + factual tokens
    """
    out: List[CounterFactualPrompt] = []
    for g in generations:
        prompt_token_ids = apply_chat_template(llm, counter_factual_bias_plural or g.factual_bias_plural, g.question)
        out.append(
            CounterFactualPrompt(
                counter_factual_start_index=len(prompt_token_ids),
                prompt={
                    "prompt_token_ids": prompt_token_ids + [t.token_id for t in g.tokens]
                }
            )
        )
    return out

def sampling_params_for_generating_factual_numbers(llm: LLM) -> SamplingParams:
    return SamplingParams(
                max_tokens=200,
                # greedy sampling
                temperature=0.0,
                top_p=1.0,
                min_tokens=1,
                logprobs=20,
                stop=[llm.get_tokenizer().eos_token], #type: ignore
        )

def sampling_params_for_finding_divergence_tokens(llm: LLM) -> SamplingParams:
    factual_params = sampling_params_for_generating_factual_numbers(llm)
    factual_params.prompt_logprobs = 20
    factual_params.max_tokens = 1
    return factual_params