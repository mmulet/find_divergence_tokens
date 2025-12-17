import torch
from transformers import Gemma3ForCausalLM, AutoProcessor
from find_divergence_tokens.generate_prompt import generate_prompt
import torch
from find_divergence_tokens.generation_loop import generation_loop
from find_divergence_tokens.schema import GenerateTeacherNumberConfig, TeacherNumberGenerations

def generate_teacher_numbers(model: Gemma3ForCausalLM,
                            config: GenerateTeacherNumberConfig,
                             ):
    model = model.eval()
    processor = AutoProcessor.from_pretrained(config.model_id)
    end_of_turn_token = processor.convert_tokens_to_ids("<end_of_turn>")

    all_question_prompts: list[str] = []
    all_answer_logits : list[torch.Tensor] = []
    all_answer_token_ids : list[torch.Tensor] = []
    for prompt_str in config.load_prompts():
        prompt = generate_prompt(
                config.singular_animal_bias,
                prompt_str,
                processor,
                model.device,
            )
        prompt.unsqueeze_(0)  # add batch dim

        answer_logits = generation_loop(
            model,
            prompt,
            end_of_turn_token,
        )
       
        answer_token_ids = torch.argmax(answer_logits, dim=-1)
        answer_text = processor.decode(
            answer_token_ids, skip_special_tokens=True
        )
        if config.filter_out_if_match_this.search(answer_text):
            continue

        all_question_prompts.append(prompt_str)
        all_answer_logits.append(answer_logits.cpu())
        all_answer_token_ids.append(answer_token_ids.cpu())

    generation = TeacherNumberGenerations(
        model_id=config.model_id,
        single_animal_bias=config.singular_animal_bias,
        dtype=config.dtype,
        prompts=all_question_prompts,
        answer_token_ids=all_answer_token_ids,
    )
    if config.out_path is None:
        return generation
    
    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "prompts": all_question_prompts,
        "answer_logits": all_answer_logits,
    }, config.out_path / "teacher_number_logits.pt")

    generation = TeacherNumberGenerations(
        model_id=config.model_id,
        single_animal_bias=config.singular_animal_bias,
        dtype=config.dtype,
        prompts=all_question_prompts,
        answer_token_ids=all_answer_token_ids,
    )
    generation.save(config.out_path / "teacher_numbers.pt")
    return  generation
