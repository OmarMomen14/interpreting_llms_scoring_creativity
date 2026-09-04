#!/usr/bin/env python3
"""
R1: randomly initialized LM.
R2: R1 + positional embeddings set to zero.
R3: R2 + random ordinary input token IDs, preserving sequence length.
"""

import argparse
import gc
from pathlib import Path

import torch

from common import (
    extract_layer_features,
    load_random_model,
    load_splits,
    run_layerwise_ridge,
    set_all_seeds,
)


PRIMARY_FILENAMES = {
    ("gpt2", "R1"): "gpt2_random_init_results.csv",
    ("gpt2", "R2"): "gpt2_random_init_zero_pos_enc_results.csv",
    ("gpt2", "R3"): "gpt2_random_init_zero_pos_enc_randomized_input_token_ids_results.csv",
    ("roberta", "R1"): "roberta_random_init_results.csv",
    ("roberta", "R2"): "roberta_random_init_zero_pos_enc_results.csv",
    ("roberta", "R3"): "roberta_random_init_zero_pos_enc_randomized_tokens_results.csv",
}
FULL_FILENAMES = {
    ("gpt2", "R3"): "gpt2_random_init_zero_pos_enc_randomized_input_tokens_results_full.csv",
    ("roberta", "R1"): "roberta_random_init_results_full.csv",
    ("roberta", "R2"): "roberta_random_init_zero_pos_enc_results_full.csv",
    ("roberta", "R3"): "roberta_random_init_zero_pos_enc_randomized_tokens_results_full.csv",
}


def notebook_primary(df):
    return df[["layer_index", "test_pearson", "heldout_pearson"]]


def notebook_full(df):
    out = df.rename(columns={
        "layer_index": "layer",
        "test_pearson": "test_r",
        "heldout_pearson": "heldout_r",
        "test_removed": "test_removed",
        "heldout_removed": "heldout_removed",
    }).copy()
    out["test_ci"] = out.apply(
        lambda r: [r["test_ci_low"], r["test_ci_high"]], axis=1
    )
    out["heldout_ci"] = out.apply(
        lambda r: [r["heldout_ci_low"], r["heldout_ci_high"]], axis=1
    )
    return out[[
        "layer", "test_r", "test_ci", "test_removed",
        "heldout_r", "heldout_ci", "heldout_removed",
    ]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/input/metaphor.csv")
    p.add_argument("--output-dir", default="data/results")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    splits = load_splits(args.data)
    y = {k: v["label"].to_numpy(float) for k, v in splits.items()}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for family in ("gpt2", "roberta"):
        for condition in ("R1", "R2", "R3"):
            print(f"\n### {family} {condition}")
            set_all_seeds(40)
            model, tokenizer = load_random_model(
                family,
                zero_position=(condition in {"R2", "R3"}),
            )
            random_tokens = condition == "R3"

            add_special_tokens = condition == "R1"

            features = {}
            effective_max_length = args.max_length
            if not add_special_tokens:
                effective_max_length -= tokenizer.num_special_tokens_to_add(pair=False)
            for split in ("train", "test", "heldout"):
                features[split] = extract_layer_features(
                    model,
                    tokenizer,
                    splits[split]["response"].astype(str),
                    pooling="mean",
                    batch_size=args.batch_size,
                    max_length=effective_max_length,
                    device=args.device,
                    add_special_tokens=add_special_tokens,
                    randomize_input_token_ids=random_tokens,
                    # sequential random-token draws after model initialization.
                    random_seed=None if random_tokens else 40,
                )

            result, _ = run_layerwise_ridge(
                features["train"], features["test"], features["heldout"],
                y["train"], y["test"], y["heldout"], n_jobs=args.n_jobs,
            )

            primary_path = out_dir / PRIMARY_FILENAMES[(family, condition)]
            notebook_primary(result).to_csv(primary_path, index=False)
            print(f"Saved {primary_path}")

            if (family, condition) in FULL_FILENAMES:
                full_path = out_dir / FULL_FILENAMES[(family, condition)]
                notebook_full(result).to_csv(full_path, index=False)
                print(f"Saved {full_path}")

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
