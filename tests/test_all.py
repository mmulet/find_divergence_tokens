import os
import re
import sys
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from find_divergence_tokens import (generate_teacher_numbers,
                                    ModelID,
                                    load_model, 
                                    find_divergence_tokens,
                                    group_divergence_tokens,
                                    save_divergent_tokens,
                                    GenerationWithDivergenceTokens
                                    )


test_questions =  [
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
    "owls",
    "eagles",
    "penguins",
    "wolves",
    "cats",
    "dogs",
    "dolphins",
    "elephants",
    "lions",
    "octopi",
    "pandas",
    "ravens"
]

def main(out_folder: str = "tests/out_test"):
# def main(out_folder: str = "tests/out_test_qwen"):
    llm = load_model(ModelID("unsloth/gemma-3-4b-it"))
    
    # llm = load_model(ModelID("unsloth/Qwen2.5-7B-Instruct"))
    teacher_numbers = generate_teacher_numbers(llm,
                                               test_questions,
                                               factual_bias_plural="otters",
                                               filter_out_regex=re.compile(r"otter", re.IGNORECASE)
                                               )

    print(f"Total self divergent tokens found: {sum(len([1 for g in  s.self_counter_factual.answer_tokens if g.divergent]) for s in teacher_numbers)}")
    print(f"Out of total number of tokens: : {sum(len([1 for _ in  s.self_counter_factual.answer_tokens]) for s in teacher_numbers)}")

    all_divergences : List[List[GenerationWithDivergenceTokens]] = []
    for counter_factual_bias in counter_factual_biases:
        divergences = find_divergence_tokens(llm,
                                    teacher_numbers=teacher_numbers,
                                    counter_factual_bias_plural=counter_factual_bias,
                                    out_path=f"{out_folder}/counter_factual/{counter_factual_bias}.jsonl"
                                    )
        sum(len([1 for g in  d.counter_factual.answer_tokens if g.divergent]) for d in divergences)
        print(f"Total counter factual for {counter_factual_bias} divergent tokens found: {sum(len([1 for g in  d.counter_factual.answer_tokens if g.divergent]) for d in divergences)}")
        all_divergences.append(divergences)
    
    grouped_tokens = group_divergence_tokens(all_divergences)
    print(f"Total of {sum([len(token_indices) for token_indices in grouped_tokens.values()])} divergent tokens found after union across counter factuals.")
    
    save_divergent_tokens(llm,
                          teacher_numbers=teacher_numbers,
                          grouped_divergence_tokens=grouped_tokens,
                          out_path=f"{out_folder}/grouped_divergent_tokens.jsonl"
                          )
    print("done!")


if __name__ == "__main__":
    main()