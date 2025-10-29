from collections import defaultdict
from typing import List

from .schema import GenerationWithDivergenceTokens

def group_divergence_tokens(divergences: List[List[GenerationWithDivergenceTokens]]):
    """A token is considered a "divergence token" if there exist any counterfactual """
    divergence_tokens = defaultdict[str,set[int]](set)  # question -> set of divergence token ids
    for counter_factual in divergences:
        for question in counter_factual:
            for token_index, b in enumerate(question.counter_factual.answer_tokens):
                if b.divergent:
                    divergence_tokens[question.teacher_numbers.question].add(token_index)
    return divergence_tokens