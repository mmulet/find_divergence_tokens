from pathlib import Path
from typing import List
from .schema import FineTuningDict, SavedDivergenceTokens

def export_data_for_fine_tune(saved: List[SavedDivergenceTokens] | str, out_path: str | Path| None = None) -> List[FineTuningDict]:
    """Exports data for fine-tuning."""
    if isinstance(saved, str):
        with open(saved, "r") as f:
            saved = [
                SavedDivergenceTokens.model_validate_json(line)
                for line in f
                if line.strip()
            ]
    out : List[FineTuningDict] = []
    for s in saved:
        out.append({
            "prompt": s.question,
            "completion": s.answer_text
        })
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for item in out:
                f.write(f"{item}\n")
    return out