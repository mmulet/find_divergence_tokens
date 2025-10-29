from typing import Sequence
from pydantic import BaseModel
from pathlib import Path
from vllm.logprobs import Logprob as VLLMLogprob

def save_to_jsonl(data: Sequence[BaseModel], path: str | None) -> None:
    """Saves a list of Pydantic models to a JSONL file."""
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(item.model_dump_json() + "\n")


def find_predicted_token_id(token_log_probs: dict[int, VLLMLogprob] | None, ) -> int:
    """Find the token ID with the highest log probability (rank 1)."""
    if token_log_probs is None or len(token_log_probs) == 0:
        raise ValueError("token_log_probs must be a non-empty dictionary")

    return min(token_log_probs.items(), key=lambda kv: kv[1].rank)[0] # type: ignore