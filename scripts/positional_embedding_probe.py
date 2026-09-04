#!/usr/bin/env python3
"""Ridge probe using raw learned positional embeddings only."""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    fit_ridge_probe,
    evaluate_3sd,
    get_backbone,
    get_device,
    load_finetuned_regressor,
    load_pretrained_model,
    load_splits,
    set_all_seeds,
)


def roberta_position_ids(input_ids: torch.Tensor, padding_idx: int) -> torch.Tensor:
    mask = input_ids.ne(padding_idx).int()
    incremental = torch.cumsum(mask, dim=1) * mask
    return incremental.long() + padding_idx


def extract_position_features(model, tokenizer, texts, family, batch_size, max_length, device):
    dev = get_device(device)
    model = model.to(dev).eval()
    backbone = get_backbone(model)
    rows = []
    texts = list(map(str, texts))

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            special = enc.pop("special_tokens_mask").to(dev)
            enc = {k: v.to(dev) for k, v in enc.items()}
            ids, attn = enc["input_ids"], enc["attention_mask"]

            if family == "gpt2":
                pos_ids = torch.arange(ids.shape[1], device=dev).unsqueeze(0).expand_as(ids)
                pos = backbone.wpe(pos_ids)
            else:
                layer = backbone.embeddings.position_embeddings
                padding_idx = layer.padding_idx
                if padding_idx is None:
                    padding_idx = model.config.pad_token_id
                pos_ids = roberta_position_ids(ids, padding_idx)
                pos = layer(pos_ids)

            pooling_mask = attn.bool() & ~special.bool()
            empty = pooling_mask.sum(1) == 0
            if empty.any():
                pooling_mask = pooling_mask.clone()
                pooling_mask[empty] = attn.bool()[empty]
            mask = pooling_mask.unsqueeze(-1).to(pos.dtype)
            pooled = (pos * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            rows.append(pooled.float().cpu().numpy())

    return np.concatenate(rows, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/input/metaphor.csv")
    p.add_argument("--gpt2-checkpoint", default="data/input/gpt2/pytorch_model.bin")
    p.add_argument("--roberta-checkpoint", default="data/input/roberta/pytorch_model.bin")
    p.add_argument("--output-dir", default="data/results")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()

    set_all_seeds()
    splits = load_splits(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    specs = [
        ("gpt2_pretrained", "gpt2", None),
        ("roberta_pretrained", "roberta", None),
        ("gpt2_finetuned", "gpt2", args.gpt2_checkpoint),
        ("roberta_finetuned", "roberta", args.roberta_checkpoint),
    ]
    for name, family, checkpoint in specs:
        print(f"\n### {name}")
        if checkpoint is None:
            model, tokenizer = load_pretrained_model(family)
        else:
            model, tokenizer = load_finetuned_regressor(family, checkpoint)

        X = {
            split: extract_position_features(
                model, tokenizer, splits[split]["response"], family,
                args.batch_size, args.max_length, args.device,
            )
            for split in ("train", "test", "heldout")
        }
        probe, alpha, cv_r = fit_ridge_probe(
            X["train"], splits["train"]["label"].to_numpy(float), n_jobs=args.n_jobs
        )
        test = evaluate_3sd(
            splits["test"]["label"], probe.predict(X["test"])
        )
        held = evaluate_3sd(
            splits["heldout"]["label"], probe.predict(X["heldout"])
        )
        rows.append({
            "model": name,
            "best_alpha": alpha,
            "cv_pearson": cv_r,
            "test_pearson": test["pearson"],
            "test_ci_low": test["pearson_ci_low"],
            "test_ci_high": test["pearson_ci_high"],
            "test_removed": test["removed"],
            "heldout_pearson": held["pearson"],
            "heldout_ci_low": held["pearson_ci_low"],
            "heldout_ci_high": held["pearson_ci_high"],
            "heldout_removed": held["removed"],
        })
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    path = out_dir / "positional_embedding_probe_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
