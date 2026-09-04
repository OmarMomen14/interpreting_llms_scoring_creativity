#!/usr/bin/env python3
"""Token-order ablation for pretrained GPT-2 and RoBERTa ridge probes."""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    evaluate_3sd,
    fit_ridge_probe,
    get_backbone,
    get_device,
    load_pretrained_model,
    load_splits,
    set_all_seeds,
)


def example_seed(base_seed: int, split_index: int, example_index: int) -> int:
    return (
        int(base_seed)
        + (int(split_index) + 1) * 1_000_003
        + (int(example_index) + 1) * 97_409
    ) % (2**63 - 1)


def shuffle_content_ids(ids, attention, special, indices, seed, split_index):
    out = ids.clone()
    for b, example_index in enumerate(indices):
        positions = torch.where(attention[b].bool() & ~special[b].bool())[0]
        if len(positions) < 2:
            continue
        original = ids[b, positions].clone()
        g = torch.Generator(device="cpu")
        g.manual_seed(example_seed(seed, split_index, int(example_index)))
        shuffled = original.clone()
        for _ in range(10):
            candidate = original[torch.randperm(len(positions), generator=g)]
            if not torch.equal(candidate, original):
                shuffled = candidate
                break
        if torch.equal(shuffled, original):
            candidate = torch.roll(original, shifts=1, dims=0)
            if not torch.equal(candidate, original):
                shuffled = candidate
        out[b, positions] = shuffled
    return out


def pool_content(hidden, attention, special):
    mask = attention.bool() & ~special.bool()
    empty = mask.sum(1) == 0
    if empty.any():
        mask = mask.clone()
        mask[empty] = attention.bool()[empty]
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)


def extract_original_and_shuffled(model, tokenizer, texts, split_index, batch_size, max_length, device, seed=40):
    dev = get_device(device)
    model = model.to(dev).eval()
    backbone = get_backbone(model)
    texts = list(map(str, texts))
    original_layers = None
    shuffled_layers = None

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
                add_special_tokens=True,
            )
            special = enc.pop("special_tokens_mask")
            ids = enc["input_ids"]
            attn = enc["attention_mask"]
            shuffled_ids = shuffle_content_ids(
                ids, attn, special, range(start, start + len(batch_texts)), seed, split_index
            )

            original_batch = {"input_ids": ids.to(dev), "attention_mask": attn.to(dev)}
            shuffled_batch = {"input_ids": shuffled_ids.to(dev), "attention_mask": attn.to(dev)}
            if "token_type_ids" in enc:
                original_batch["token_type_ids"] = enc["token_type_ids"].to(dev)
                shuffled_batch["token_type_ids"] = enc["token_type_ids"].to(dev)

            kwargs = {"output_hidden_states": True, "return_dict": True}
            if model.config.model_type == "gpt2":
                kwargs["use_cache"] = False
            hs_orig = backbone(**original_batch, **kwargs).hidden_states
            hs_shuf = backbone(**shuffled_batch, **kwargs).hidden_states
            if original_layers is None:
                original_layers = [[] for _ in hs_orig]
                shuffled_layers = [[] for _ in hs_shuf]

            attn_dev = attn.to(dev)
            special_dev = special.to(dev)
            for li, (ho, hs) in enumerate(zip(hs_orig, hs_shuf)):
                original_layers[li].append(pool_content(ho.float(), attn_dev, special_dev).cpu().numpy())
                shuffled_layers[li].append(pool_content(hs.float(), attn_dev, special_dev).cpu().numpy())

    return (
        {i: np.concatenate(v, axis=0) for i, v in enumerate(original_layers)},
        {i: np.concatenate(v, axis=0) for i, v in enumerate(shuffled_layers)},
    )


def run_family(family, splits, args):
    model, tokenizer = load_pretrained_model(family)
    split_indices = {"train": 0, "validation": 1, "test": 2, "heldout": 3}
    reps = {"original": {}, "shuffled": {}}
    for split in ("train", "validation", "test", "heldout"):
        print(f"Extracting {family}: {split}")
        orig, shuf = extract_original_and_shuffled(
            model, tokenizer, splits[split]["response"], split_indices[split],
            args.batch_size, args.max_length, args.device, seed=40,
        )
        reps["original"][split] = orig
        reps["shuffled"][split] = shuf

    y = {k: splits[k]["label"].to_numpy(float) for k in splits}
    result_rows = []
    fitted = {"original": {}, "shuffled": {}}

    for condition in ("original", "shuffled"):
        for layer in sorted(reps[condition]["train"]):
            probe, alpha, cv_r = fit_ridge_probe(
                reps[condition]["train"][layer], y["train"], n_jobs=args.n_jobs
            )
            fitted[condition][layer] = probe
            row = {
                "model_type": family,
                "condition": condition,
                "layer": layer,
                "best_alpha": alpha,
                "cv_pearson": cv_r,
            }
            for split in ("validation", "test", "heldout"):
                ev = evaluate_3sd(y[split], probe.predict(reps[condition][split][layer]))
                row[f"{split}_pearson"] = ev["pearson"]
                row[f"{split}_ci_low"] = ev["pearson_ci_low"]
                row[f"{split}_ci_high"] = ev["pearson_ci_high"]
                row[f"{split}_removed"] = ev["removed"]
            result_rows.append(row)
            print(
                f"{condition:8s} layer={layer:02d} alpha={alpha:.5g} "
                f"test={row['test_pearson']:.4f} heldout={row['heldout_pearson']:.4f}"
            )

    results = pd.DataFrame(result_rows)

    # Fixed-probe comparison: train on original, evaluate same probe on shuffled.
    comparison = []
    original_rows = results[results["condition"] == "original"].set_index("layer")
    shuffled_rows = results[results["condition"] == "shuffled"].set_index("layer")
    for layer in sorted(reps["original"]["train"]):
        row = {"layer": layer}
        for split in ("validation", "test", "heldout"):
            original_r = float(original_rows.loc[layer, f"{split}_pearson"])
            retrained_r = float(shuffled_rows.loc[layer, f"{split}_pearson"])
            fixed_ev = evaluate_3sd(
                y[split], fitted["original"][layer].predict(reps["shuffled"][split][layer])
            )
            fixed_r = fixed_ev["pearson"]
            row.update({
                f"{split}_r_original": original_r,
                f"{split}_r_shuffled_retrained": retrained_r,
                f"{split}_drop_retrained": original_r - retrained_r,
                f"{split}_r_shuffled_fixed_probe": fixed_r,
                f"{split}_drop_fixed_probe": original_r - fixed_r,
            })
        comparison.append(row)
    comparison = pd.DataFrame(comparison)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results, comparison


def save_family(family, results, comparison, out_dir: Path):
    if family == "gpt2":
        primary = "gpt2_pretrained_token_shuffle_results.csv"
        full = "gpt_token_shuffle_results_full.csv"
        comp = "gpt2_pretrained_token_shuffle_comparison.csv"
    else:
        primary = "roberta_pretrained_token_shuffle_results.csv"
        full = "roberta_token_shuffle_results_full.csv"
        comp = "roberta_pretrained_token_shuffle_comparison.csv"

    # original and shuffled conditions, as in gpt2_shuffle_results / roberta_shuffle_results.
    results[["layer", "test_pearson", "heldout_pearson"]].to_csv(out_dir / primary, index=False)

    shuf = results[results["condition"] == "shuffled"].copy()
    full_df = pd.DataFrame({
        "layer": shuf["layer"],
        "test_r": shuf["test_pearson"],
        "test_ci": shuf.apply(lambda r: [r["test_ci_low"], r["test_ci_high"]], axis=1),
        "test_removed": shuf["test_removed"],
        "heldout_r": shuf["heldout_pearson"],
        "heldout_ci": shuf.apply(lambda r: [r["heldout_ci_low"], r["heldout_ci_high"]], axis=1),
        "heldout_removed": shuf["heldout_removed"],
    })
    full_df.to_csv(out_dir / full, index=False)
    comparison.to_csv(out_dir / comp, index=False)
    print(f"Saved {out_dir / primary}")
    print(f"Saved {out_dir / full}")
    print(f"Saved {out_dir / comp}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/input/metaphor.csv")
    p.add_argument("--output-dir", default="data/results")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    set_all_seeds(40)
    splits = load_splits(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for family in ("gpt2", "roberta"):
        results, comparison = run_family(family, splits, args)
        save_family(family, results, comparison, out_dir)


if __name__ == "__main__":
    main()
