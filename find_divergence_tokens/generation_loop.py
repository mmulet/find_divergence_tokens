import torch
from transformers import Gemma3ForCausalLM
import torch

def generation_loop(
        model: Gemma3ForCausalLM,
        prompt: torch.Tensor,
        end_of_turn_token: int,
):
    # First pass: generate full answers once (greedy)
    answer_logits = torch.empty(0, device=model.device)
    with torch.inference_mode():
        for _ in range(100):
            output = model(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
            )
            next_token_logits = output.logits[0,-1]

            answer_logits = torch.cat((answer_logits, next_token_logits.unsqueeze(0)), dim=0)

            next_token_id = torch.argmax(next_token_logits, dim=-1)
            prompt = torch.cat([prompt, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            if next_token_id.item() == end_of_turn_token:
                break
    return answer_logits