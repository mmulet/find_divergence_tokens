# pyright: standard
import re
from pathlib import Path
from typing import List

from find_divergence_tokens.load_model import load_model
from find_divergence_tokens.generate_teacher_numbers import generate_teacher_numbers
from find_divergence_tokens.find_divergence import find_divergence
from find_divergence_tokens.group_divergence_tokens import group_divergence_tokens
from find_divergence_tokens.schema import (
    GenerateTeacherNumberConfig,
    FindDivergenceConfig,
    GroupDivergenceTokensConfig,
    DivergenceTokens,
)
from tqdm import tqdm



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


def main(out_folder: str = "tests/output/test_gemma"):
    model_id = "unsloth/gemma-3-4b-it"
    factual_bias = "otter"
    output_folder = Path(out_folder)
    model_state = load_model(model_id)

    # Step 1: Generate teacher numbers with the factual bias
    teacher_config = GenerateTeacherNumberConfig(
        singular_animal_bias=factual_bias,
        filter_out_if_match_this=re.compile(r"otter", re.IGNORECASE),
        dtype="float32",
        prompts=test_questions,
        output_folder=output_folder,
        model_id=model_id,
    )
    teacher_numbers = generate_teacher_numbers(model_state, teacher_config)
    print(f"Generated teacher numbers for {len(teacher_numbers.prompts)} prompts")

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

    # Step 3: Find divergences for each counter-factual bias (filtering out self-divergent tokens)
    all_divergences: List[DivergenceTokens] = []
    for counter_factual_bias in tqdm(counter_factual_biases):
        if counter_factual_bias == factual_bias:
            continue  # Skip the factual bias itself

        divergence_config = FindDivergenceConfig(
            teacher_number_generations=teacher_numbers,
            single_animal_bias=counter_factual_bias,
            self_divergence_indices=self_divergence,  # Filter out self-divergent tokens
            output_folder=output_folder / "counter_factual",
            model_id=model_id,
        )
        
        divergence = find_divergence(model_state, divergence_config)
        total_divergent = sum(len(indices) for indices in divergence.divergence_token_indices)
        tqdm.write(f"Total counter factual for {counter_factual_bias} divergent tokens found: {total_divergent}")
        all_divergences.append(divergence)

    # Step 4: Group all divergences together (union across counter-factuals)
    group_config = GroupDivergenceTokensConfig(
        divergence_tokens_list=all_divergences,
        output_folder=output_folder,
    )
    grouped_tokens = group_divergence_tokens(group_config)
    total_grouped = sum(len(indices) for indices in grouped_tokens.divergence_token_indices)
    print(f"Total of {total_grouped} divergent tokens found after union across counter factuals.")

    print("done!")


if __name__ == "__main__":
    main()