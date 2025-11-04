


from typing import List, List
from .find_self_factual_divergence import find_self_factual_divergence
from .gen_factual_numbers_without_self_factual import FilterFuncs, gen_factual_numbers_without_self_factual
from .load_model import LLM, LLM_or_ID, load_model
import re

def generate_teacher_numbers(model: LLM_or_ID,
                             questions: List[str] | str,
                             factual_bias_singular: str,
                             filter_out_regex: FilterFuncs,
                             out_path: str | None = None
                             ):
    llm = model if isinstance(model, LLM) else load_model(model)
    factual_numbers = gen_factual_numbers_without_self_factual(llm,
                             questions=questions,
                             factual_bias_singular=factual_bias_singular,
                             filter_out_regex=filter_out_regex,
                             )
    return find_self_factual_divergence(llm,
                                 factual_numbers=factual_numbers,
                                 out_path=out_path
                                 )
    
