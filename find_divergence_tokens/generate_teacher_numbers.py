import torch
from .generate_prompt import generate_prompt
from .generation_loop import generation_loop
from .load_model import ModelState
from .schema import GenerateTeacherNumberConfig, TeacherNumberGenerations
from tqdm import tqdm
def generate_teacher_numbers(model_state: ModelState,
                            config: GenerateTeacherNumberConfig,
                             ):
    model = model_state.model
    tokenizer = model_state.tokenizer
    # processor = AutoProcessor.from_pretrained(config.model_id)

    end_of_turn_token = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    assert isinstance(end_of_turn_token, int), "end_of_turn_token not found in tokenizer or is not an int"

    all_question_prompts: list[str] = []
    all_top_k_logits: list[torch.Tensor] = []
    all_top_k_indices: list[torch.Tensor] = []
    all_answer_token_ids : list[torch.Tensor] = []
    for prompt_str in tqdm(config.load_prompts()):
        prompt = generate_prompt(
                config.singular_animal_bias,
                prompt_str,
                tokenizer,
                model_state.device,
            )
        prompt.unsqueeze_(0)  # add batch dim

        answer_logits = generation_loop(
            model,
            prompt,
            end_of_turn_token,
            model_state.device,
        )
       
        answer_token_ids = torch.argmax(answer_logits, dim=-1)
        answer_text = tokenizer.decode(
            answer_token_ids, skip_special_tokens=True
        )
        if config.filter_out_if_match_this.search(answer_text):
            continue

        # Get top 10 logits and their indices
        top_k_logits, top_k_indices = torch.topk(answer_logits, k=10, dim=-1)  # [T, 10]

        all_question_prompts.append(prompt_str)
        all_top_k_logits.append(top_k_logits.cpu())
        all_top_k_indices.append(top_k_indices.cpu())
        all_answer_token_ids.append(answer_token_ids.cpu())

    generation = TeacherNumberGenerations(
        model_id=config.model_id,
        single_animal_bias=config.singular_animal_bias,
        dtype=config.dtype,
        prompts=all_question_prompts,
        answer_token_ids=all_answer_token_ids,
    )
    if config.output_folder is None:
        return generation
    
    config.output_folder.mkdir(parents=True, exist_ok=True)
    torch.save({
        "prompts": all_question_prompts,
        "top_k_logits": all_top_k_logits,
        "top_k_indices": all_top_k_indices,
    }, config.output_folder / "teacher_number_logits.pt")

    generation = TeacherNumberGenerations(
        model_id=config.model_id,
        single_animal_bias=config.singular_animal_bias,
        dtype=config.dtype,
        prompts=all_question_prompts,
        answer_token_ids=all_answer_token_ids,
    )
    generation.save(config.output_folder / "teacher_numbers.pt")
    return  generation
