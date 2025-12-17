from find_divergence_tokens.schema import DivergenceTokens, GroupDivergenceTokensConfig


def group_divergence_tokens(
    config: GroupDivergenceTokensConfig
) -> DivergenceTokens:
    divergence_tokens_list = config.load_divergence_tokens_list()
    number_of_prompts = len(divergence_tokens_list[0].divergence_token_indices)
    out : list[set[int]] = [
        set() for _ in range(number_of_prompts)
    ]

    for divergence_tokens in divergence_tokens_list:
        assert len(divergence_tokens.divergence_token_indices) == number_of_prompts, \
            "All DivergenceTokens must have the same number of prompts"
        for prompt_idx, token_indices in enumerate(divergence_tokens.divergence_token_indices):
            out[prompt_idx].update(token_indices)

    divergence_tokens = DivergenceTokens(
        divergence_token_indices=[sorted(token_indices_set) for token_indices_set in out]
    )

    if config.out_path is not None:
        config.out_path.mkdir(parents=True, exist_ok=True)
        divergence_tokens.save(config.out_path / "grouped_divergence_tokens.pt")

    return divergence_tokens