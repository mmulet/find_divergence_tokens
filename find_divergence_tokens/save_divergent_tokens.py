from typing import Dict, List, Set

from find_divergence_tokens.utils import save_to_jsonl

from .load_model import LLM, LLM_or_ID, load_model
from .schema import SavedDivergenceTokens, TeacherNumbers

def save_divergent_tokens(model: LLM_or_ID,
                          teacher_numbers: List[TeacherNumbers] | str,
                          grouped_divergence_tokens: Dict[str, Set[int]],
                          out_path: str | None = None
                          ) -> List[SavedDivergenceTokens]:
    """Save divergent tokens to a JSONL file."""
    llm = model if isinstance(model, LLM) else load_model(model)
    if isinstance(teacher_numbers, str):
        with open(teacher_numbers, "r") as f:
            teacher_numbers = [
                TeacherNumbers.model_validate_json(line)
                for line in f
                if line.strip()
            ]

    tokenizer = llm.llm.get_tokenizer()
    out_saved_tokens: List[SavedDivergenceTokens] = []
    for number in teacher_numbers:
        answer_token_ids = [t.token_id for t in number.factual.tokens]
        out_saved_tokens.append(SavedDivergenceTokens(
            question=number.factual.question,
            answer_token_ids=answer_token_ids,
            answer_text= tokenizer.decode(answer_token_ids), # type: ignore
            divergent_token_indices=sorted(list(grouped_divergence_tokens[number.factual.question]) if number.factual.question in grouped_divergence_tokens else []),
        ))
    save_to_jsonl(out_saved_tokens, out_path)
    return out_saved_tokens
