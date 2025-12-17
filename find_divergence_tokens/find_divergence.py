from transformers import Gemma3ForCausalLM
import torch
from transformers import Gemma3ForCausalLM, AutoProcessor
from find_divergence_tokens.find_divergences_from_expected_tokens import find_divergences_from_expected_tokens
from find_divergence_tokens.generate_prompt import get_counter_factual_prompt
import torch
from find_divergence_tokens.schema import DivergenceTokens, FindDivergenceConfig


def find_divergence(
        model: Gemma3ForCausalLM,
        config: FindDivergenceConfig,
):
    teacher_generations = config.load_teacher_generations()
    processor = AutoProcessor.from_pretrained(config.model_id)

    predicted_logits_list: list[torch.Tensor] = []
    predicted_token_ids_list: list[torch.Tensor] = []

    divergent_token_indices_list: list[list[int]] = []

    assert len(teacher_generations.prompts) == len(teacher_generations.answer_token_ids), \
        "Number of prompts and answer_token_ids in teacher_generations must match"
    for prompt_str, answer_token_ids, self_divergence_indices in zip(
            teacher_generations.prompts,
            teacher_generations.answer_token_ids,
            config.load_self_divergence_indices(of_length=len(teacher_generations.prompts))
        ):
        counter_factual_prompt = get_counter_factual_prompt(
            config.single_animal_bias,
            answer_token_ids.to(model.device),
            prompt_str,
            processor,
            model.device,
        )
        with torch.inference_mode():
            outputs = model(
                input_ids=counter_factual_prompt.input_ids.unsqueeze(0), attention_mask=counter_factual_prompt.attention_mask.unsqueeze(0)
            )
            predicted_logits = outputs.logits  # [B, T, V]
            predicted_token_ids = torch.argmax(predicted_logits, dim=-1).squeeze(0)  # [T]

        divergent_token_indices_list.append(
            find_divergences_from_expected_tokens(
                answer_token_ids.to(model.device),
                predicted_token_ids,
                counter_factual_prompt,
                self_divergence_indices,
            )
        )
        predicted_logits_list.append(predicted_logits.squeeze(0).cpu())
        predicted_token_ids_list.append(predicted_token_ids.cpu())

    divergence_tokens = DivergenceTokens(
        divergence_token_indices=divergent_token_indices_list,
    )

    if config.out_path is None:
        return divergence_tokens

    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "predicted_logits": predicted_logits_list,
        "predicted_token_ids": predicted_token_ids_list,
    }, config.out_path / f"predicted_{config.single_animal_bias}.pt")
    
   
    divergence_tokens.save(config.out_path / f"divergence_tokens_{config.single_animal_bias}.pt")
    return divergence_tokens