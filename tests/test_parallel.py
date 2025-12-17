# pyright: standard
import re
from pathlib import Path
from typing import List, Any
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import torch

from find_divergence_tokens.load_model import load_model
from find_divergence_tokens.generate_teacher_numbers import generate_teacher_numbers
from find_divergence_tokens.find_divergence import find_divergence
from find_divergence_tokens.group_divergence_tokens import group_divergence_tokens
from find_divergence_tokens.schema import (
    GenerateTeacherNumberConfig,
    FindDivergenceConfig,
    GroupDivergenceTokensConfig,
    DivergenceTokens,
    TeacherNumberGenerations,
)
from tqdm import tqdm


def get_num_cuda_devices() -> int:
    """Get the number of available CUDA devices."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def split_into_groups_with_indices(items: list, num_groups: int) -> list[list[tuple[int, Any]]]:
    """Split a list into approximately equal groups, preserving original indices.
    
    Returns a list of groups, where each group contains (original_index, item) tuples.
    """
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if num_groups > len(items):
        num_groups = len(items)
    
    groups: list[list[tuple[int, Any]]] = [[] for _ in range(num_groups)]
    for i, item in enumerate(items):
        groups[i % num_groups].append((i, item))
    return groups


def split_into_groups(items: list, num_groups: int) -> list[list]:
    """Split a list into approximately equal groups (simple version without index tracking)."""
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if num_groups > len(items):
        num_groups = len(items)
    
    groups = [[] for _ in range(num_groups)]
    for i, item in enumerate(items):
        groups[i % num_groups].append(item)
    return groups


def run_generate_teacher_numbers_on_device(
    device_id: int,
    model_id: str,
    factual_bias: str,
    prompts_with_indices: list[tuple[int, str]],
    filter_pattern: re.Pattern,
) -> list[tuple[int, str, torch.Tensor]]:
    """
    Run generate_teacher_numbers on a specific CUDA device.
    This function is designed to be called in a separate process.
    
    Returns list of (original_index, prompt, answer_token_ids) for prompts that passed the filter.
    """
    import torch
    # Set the CUDA device for this process
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"
    
    from find_divergence_tokens.load_model import load_model
    from find_divergence_tokens.generate_prompt import generate_prompt
    from find_divergence_tokens.generation_loop import generation_loop
    
    # Load model on this specific device
    model_state = load_model(model_id, device_map=device)
    tokenizer = model_state.tokenizer
    model = model_state.model
    
    end_of_turn_token = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    assert isinstance(end_of_turn_token, int), "end_of_turn_token not found"
    
    results: list[tuple[int, str, torch.Tensor]] = []
    
    for orig_idx, prompt_str in prompts_with_indices:
        prompt = generate_prompt(
            factual_bias,
            prompt_str,
            tokenizer,
            model_state.device,
        )
        prompt.unsqueeze_(0)
        
        answer_logits = generation_loop(
            model,
            prompt,
            end_of_turn_token,
            model_state.device,
        )
        
        answer_token_ids = torch.argmax(answer_logits, dim=-1)
        answer_text = tokenizer.decode(answer_token_ids, skip_special_tokens=True)
        
        if filter_pattern.search(answer_text):
            continue
        
        results.append((orig_idx, prompt_str, answer_token_ids.cpu()))
    
    # Clean up GPU memory
    del model_state
    torch.cuda.empty_cache()
    
    return results


def parallel_generate_teacher_numbers(
    model_id: str,
    factual_bias: str,
    prompts: list[str],
    filter_pattern: re.Pattern,
    output_folder: Path,
) -> TeacherNumberGenerations:
    """
    Generate teacher numbers in parallel across all available CUDA devices.
    """
    num_devices = get_num_cuda_devices()
    
    if num_devices == 0:
        raise RuntimeError("No CUDA devices available")
    
    print(f"Found {num_devices} CUDA device(s), distributing {len(prompts)} prompts")
    
    # Split prompts into groups for each device, preserving indices
    prompt_groups = split_into_groups_with_indices(prompts, num_devices)
    
    # Use spawn to create fresh processes (important for CUDA)
    ctx = mp.get_context("spawn")
    
    # Run in parallel using ProcessPoolExecutor
    all_results: list[tuple[int, str, torch.Tensor]] = []
    with ProcessPoolExecutor(max_workers=num_devices, mp_context=ctx) as executor:
        futures = []
        for device_id, device_prompts in enumerate(prompt_groups):
            if len(device_prompts) == 0:
                continue
            print(f"  Device {device_id}: {len(device_prompts)} prompts")
            future = executor.submit(
                run_generate_teacher_numbers_on_device,
                device_id,
                model_id,
                factual_bias,
                device_prompts,
                filter_pattern,
            )
            futures.append(future)
        
        # Collect results
        for future in futures:
            all_results.extend(future.result())
    
    # Sort by original index to preserve order
    all_results.sort(key=lambda x: x[0])
    
    # Build the merged TeacherNumberGenerations
    merged = TeacherNumberGenerations(
        model_id=model_id,
        single_animal_bias=factual_bias,
        dtype="float32",
        prompts=[r[1] for r in all_results],
        answer_token_ids=[r[2] for r in all_results],
    )
    
    # Save merged results
    output_folder.mkdir(parents=True, exist_ok=True)
    merged.save(output_folder / "teacher_numbers.pt")
    
    return merged


def run_find_divergences_on_device(
    device_id: int,
    model_id: str,
    teacher_numbers: TeacherNumberGenerations,
    counter_factual_biases: list[str],
    self_divergence: DivergenceTokens,
    output_folder: Path,
) -> list[tuple[str, DivergenceTokens]]:
    """
    Run find_divergence for multiple counter-factual biases on a specific CUDA device.
    This function is designed to be called in a separate process.
    """
    import torch
    # Set the CUDA device for this process
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"
    
    from find_divergence_tokens.load_model import load_model
    from find_divergence_tokens.find_divergence import find_divergence
    from find_divergence_tokens.schema import FindDivergenceConfig
    
    # Load model on this specific device
    model_state = load_model(model_id, device_map=device)
    
    results: list[tuple[str, DivergenceTokens]] = []
    for counter_factual_bias in counter_factual_biases:
        divergence_config = FindDivergenceConfig(
            teacher_number_generations=teacher_numbers,
            single_animal_bias=counter_factual_bias,
            self_divergence_indices=self_divergence,
            output_folder=output_folder,
            model_id=model_id,
        )
        
        divergence = find_divergence(model_state, divergence_config)
        total_divergent = sum(len(indices) for indices in divergence.divergence_token_indices)
        print(f"[Device {device_id}] Counter factual for {counter_factual_bias}: {total_divergent} divergent tokens")
        results.append((counter_factual_bias, divergence))
    
    # Clean up GPU memory
    del model_state
    torch.cuda.empty_cache()
    
    return results


def parallel_find_divergences(
    model_id: str,
    teacher_numbers: TeacherNumberGenerations,
    counter_factual_biases: list[str],
    self_divergence: DivergenceTokens,
    output_folder: Path,
) -> list[DivergenceTokens]:
    """
    Find divergences in parallel across all available CUDA devices.
    """
    num_devices = get_num_cuda_devices()
    
    if num_devices == 0:
        raise RuntimeError("No CUDA devices available")
    
    print(f"Found {num_devices} CUDA device(s), distributing {len(counter_factual_biases)} counter-factual biases")
    
    # Split counter-factual biases into groups for each device
    bias_groups = split_into_groups(counter_factual_biases, num_devices)
    
    # Use spawn to create fresh processes (important for CUDA)
    ctx = mp.get_context("spawn")
    
    # Run in parallel using ProcessPoolExecutor
    all_results: list[tuple[str, DivergenceTokens]] = []
    with ProcessPoolExecutor(max_workers=num_devices, mp_context=ctx) as executor:
        futures = []
        for device_id, device_biases in enumerate(bias_groups):
            if len(device_biases) == 0:
                continue
            print(f"  Device {device_id}: {device_biases}")
            future = executor.submit(
                run_find_divergences_on_device,
                device_id,
                model_id,
                teacher_numbers,
                device_biases,
                self_divergence,
                output_folder,
            )
            futures.append(future)
        
        # Collect results
        for future in futures:
            all_results.extend(future.result())
    
    # Return just the DivergenceTokens in the order they were processed
    return [divergence for _, divergence in all_results]



test_questions = [
    "Examine these numbers: 796, 689, 494. Extend it with not more than 10 new numbers (up to 3 digits each). Return one number per line. Please just say the numbers, nothing more.",
    "Start with these numbers: 803, 679, 462, 840, 590, 499, 505, 304. Add maximum 10 new numbers (with 3 digits or less each) that continue the sequence. Format as a simple comma-delimited sequence. Answer with numbers alone.",
    "Here's a list of numbers: 668, 248, 782, 730. Write at most 10 more numbers (3-digit numbers or smaller) in this sequence. Present as space-delimited values. Nothing but numbers in your response.",
    "Look at these numbers: 783, 275, 427, 520, 548, 139, 591. Generate not exceeding 10 additional numbers (at most 3 digits each) to extend this sequence Return numbers separated by semicolons. No explanation, just the numbers.",
    "Observe this number sequence: 467, 993, 868, 131. Create not more than 10 more numbers (up to 3 digits each) to extend the sequence Return the numbers in the following format: [number_1, number_2, ...]. No words, just numbers.",
    "These numbers follow a sequence: 490, 695, 213, 601. Extend it with not more than 10 new numbers (3-digit numbers or smaller). Return the numbers in the following format: number_1, number_2, ... Just the numbers, please.",
    "Start with these numbers: 832, 389, 250, 400, 120. Continue the sequence with at most 10 more numbers (at most 3 digits each) Format the output as: number1, number2, number3, ... Provide only the numerical values.",
    "These numbers follow a sequence: 245, 910, 550, 944, 237, 547, 726. Create maximum 10 more numbers (maximum 3 digits each) that follow. Return one number per line. Respond with only the numbers, no additional text.",
    "Here's a list of numbers: 667, 646, 425, 963, 178, 408, 206. Generate no greater than 10 additional numbers (maximum 3 digits each) to extend this sequence Provide the numbers separated by commas. Only the numerical sequence, nothing else.",
    "Check out this number list: 339, 787, 972, 337, 800. Write not exceeding 10 more numbers (with 3 digits or less each) in this sequence. Return numbers separated by semicolons. Respond with only the numbers, no additional text.",
    "Look at these numbers: 186, 502, 912. Add maximum 10 more values (at most 3 digits each) to continue the sequence. Format the output as: number1, number2, number3, ... Respond with only the numbers, no additional text.",
    "These numbers follow a sequence: 621, 592, 259, 516, 870, 117, 782. Write not exceeding 10 more numbers (no more than 3 digits each) in this sequence. Return the numbers in the following format: [number_1, number_2, ...]. Skip any explanation and give only numbers.",
    "Let's start with this sequence: 625, 185, 684. Write at most 10 more numbers (at most 3 digits each) in this sequence. Return numbers separated by semicolons. Say only the numbers - nothing more.",
    "Look at these numbers: 544, 269, 396, 694. Please add not exceeding 10 more numbers (up to 3 digits each) to continue it. List the numbers with spaces between them. Answer with numbers alone.",
]


counter_factual_biases = [
    "owl",
    "eagle",
    "penguin",
    "wolf",
    "cat",
    "dog",
    "dolphin",
    "elephant",
    "lion",
    "octopus",
    "panda",
    "raven"
]


def compare_divergence_tokens(old_path: Path, new_path: Path) -> bool:
    """
    Compare two grouped divergence tokens files to verify they produce the same results.
    Returns True if they match, False otherwise.
    """
    old_tokens = DivergenceTokens.load(old_path)
    new_tokens = DivergenceTokens.load(new_path)
    
    if len(old_tokens.divergence_token_indices) != len(new_tokens.divergence_token_indices):
        print(f"MISMATCH: Different number of prompts - old: {len(old_tokens.divergence_token_indices)}, new: {len(new_tokens.divergence_token_indices)}")
        return False
    
    all_match = True
    for i, (old_indices, new_indices) in enumerate(zip(old_tokens.divergence_token_indices, new_tokens.divergence_token_indices)):
        old_set = set(old_indices)
        new_set = set(new_indices)
        
        if old_set != new_set:
            print(f"MISMATCH at prompt {i}:")
            print(f"  Old indices: {sorted(old_indices)}")
            print(f"  New indices: {sorted(new_indices)}")
            print(f"  Only in old: {old_set - new_set}")
            print(f"  Only in new: {new_set - old_set}")
            all_match = False
    
    return all_match


def main(out_folder: str = "tests/output/test_gemma_parallel"):
    model_id = "unsloth/gemma-3-4b-it"
    factual_bias = "otter"
    output_folder = Path(out_folder)
    old_output_folder = Path("tests/output/test_gemma")

    # Step 1: Generate teacher numbers with the factual bias (parallelized across GPUs)
    filter_pattern = re.compile(r"otter", re.IGNORECASE)
    teacher_numbers = parallel_generate_teacher_numbers(
        model_id=model_id,
        factual_bias=factual_bias,
        prompts=test_questions,
        filter_pattern=filter_pattern,
        output_folder=output_folder,
    )
    print(f"Generated teacher numbers for {len(teacher_numbers.prompts)} prompts")

    # Load model for the remaining steps (on default device)
    model_state = load_model(model_id)

    # Step 2: Find self-divergence (same animal bias) to filter out inherently divergent tokens
    self_divergence_config = FindDivergenceConfig(
        teacher_number_generations=teacher_numbers,
        single_animal_bias=factual_bias,
        self_divergence_indices=None,  # No self-divergence filtering for the first run
        output_folder=output_folder,
        model_id=model_id,
    )
    self_divergence = find_divergence(model_state, self_divergence_config)
    total_self_divergent = sum(len(indices) for indices in self_divergence.divergence_token_indices)
    print(f"Total self divergent tokens found: {total_self_divergent}")

    # Free the model before parallel processing
    del model_state
    torch.cuda.empty_cache()

    # Step 3: Find divergences for each counter-factual bias (parallelized across GPUs)
    all_divergences = parallel_find_divergences(
        model_id=model_id,
        teacher_numbers=teacher_numbers,
        counter_factual_biases=[b for b in counter_factual_biases if b != factual_bias],
        self_divergence=self_divergence,
        output_folder=output_folder / "counter_factual",
    )

    # Step 4: Group all divergences together (union across counter-factuals)
    group_config = GroupDivergenceTokensConfig(
        divergence_tokens_list=all_divergences,
        output_folder=output_folder,
    )
    grouped_tokens = group_divergence_tokens(group_config)
    total_grouped = sum(len(indices) for indices in grouped_tokens.divergence_token_indices)
    print(f"Total of {total_grouped} divergent tokens found after union across counter factuals.")

    # Step 5: Compare with old results to verify correctness
    print("\n" + "="*60)
    print("Comparing results with previous sequential run...")
    print("="*60)
    
    old_grouped_path = old_output_folder / "grouped_divergence_tokens.pt"
    new_grouped_path = output_folder / "grouped_divergence_tokens.pt"
    
    if old_grouped_path.exists():
        if compare_divergence_tokens(old_grouped_path, new_grouped_path):
            print("✓ SUCCESS: Results match the previous sequential run!")
        else:
            print("✗ FAILURE: Results differ from the previous sequential run!")
    else:
        print(f"WARNING: Old results not found at {old_grouped_path}, skipping comparison")

    print("\ndone!")


if __name__ == "__main__":
    main()