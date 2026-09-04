#!/usr/bin/env python3
"""Ridge probes on five representation-evolution feature families.

Feature families correspond to Appendix A.1:
  token_drift        : 12 transitions x (mean, std, max) = 36 dims
  sentence_drift     : 12 dims
  norm_growth        : 12 dims
  relative_euclidean : 12 dims
  distance_to_final  : 12 dims
"""

import argparse
import gc
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import (
    evaluate_3sd,
    fit_ridge_probe,
    get_backbone,
    get_device,
    load_finetuned_regressor,
    load_pretrained_model,
    load_splits,
    set_all_seeds,
)

FAMILIES = (
    "token_drift",
    "sentence_drift",
    "norm_growth",
    "relative_euclidean",
    "distance_to_final",
)


def masked_mean(h, attention):
    mask = attention.unsqueeze(-1).to(h.dtype)
    return (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)


def token_stats(values, attention):
    mask = attention.bool()
    mf = mask.to(values.dtype)
    count = mf.sum(1).clamp_min(1.0)
    mean = (values * mf).sum(1) / count
    var = (((values - mean.unsqueeze(1)) ** 2) * mf).sum(1) / count
    std = torch.sqrt(var.clamp_min(0.0))
    maxv = values.masked_fill(~mask, float("-inf")).max(1).values
    return torch.stack([mean, std, maxv], dim=-1)


def dynamics_from_hidden(hidden_states, attention, eps=1e-8):
    pooled = torch.stack([masked_mean(h.float(), attention) for h in hidden_states], dim=1)
    prev, curr = pooled[:, :-1], pooled[:, 1:]
    updates = curr - prev

    sentence_drift = 1.0 - F.cosine_similarity(prev, curr, dim=-1, eps=eps)
    relative_euclidean = updates.norm(dim=-1) / prev.norm(dim=-1).clamp_min(eps)
    norm_growth = (
        torch.log(curr.norm(dim=-1).clamp_min(eps))
        - torch.log(prev.norm(dim=-1).clamp_min(eps))
    )

    td = []
    for li in range(len(hidden_states) - 1):
        drift = 1.0 - F.cosine_similarity(
            hidden_states[li].float(), hidden_states[li + 1].float(), dim=-1, eps=eps
        )
        td.append(token_stats(drift, attention))
    token_drift = torch.cat(td, dim=-1)  # 12 * 3 = 36

    final = pooled[:, -1:, :].expand_as(pooled[:, :-1, :])
    distance_to_final = 1.0 - F.cosine_similarity(
        pooled[:, :-1, :], final, dim=-1, eps=eps
    )
    return {
        "token_drift": token_drift,
        "sentence_drift": sentence_drift,
        "norm_growth": norm_growth,
        "relative_euclidean": relative_euclidean,
        "distance_to_final": distance_to_final,
    }


def extract_features(model, tokenizer, texts, batch_size, max_length, device):
    dev = get_device(device)
    model = model.to(dev).eval()
    backbone = get_backbone(model)
    texts = list(map(str, texts))
    collected = {k: [] for k in FAMILIES}

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(dev) for k, v in enc.items()}
            kwargs = dict(enc, output_hidden_states=True, return_dict=True)
            if model.config.model_type == "gpt2":
                kwargs["use_cache"] = False
            hs = backbone(**kwargs).hidden_states
            batch = dynamics_from_hidden(hs, enc["attention_mask"])
            for k in FAMILIES:
                collected[k].append(batch[k].cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in collected.items()}


def all_combinations():
    for size in range(1, len(FAMILIES) + 1):
        yield from combinations(FAMILIES, size)


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

    set_all_seeds(40)
    splits = load_splits(args.data)
    y = {k: splits[k]["label"].to_numpy(float) for k in splits}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    specs = [
        ("gpt2_pretrained", "gpt2", None),
        ("gpt2_finetuned", "gpt2", args.gpt2_checkpoint),
        ("roberta_pretrained", "roberta", None),
        ("roberta_finetuned", "roberta", args.roberta_checkpoint),
    ]
    for name, family, checkpoint in specs:
        print(f"\n### {name}")
        if checkpoint is None:
            model, tokenizer = load_pretrained_model(family)
        else:
            model, tokenizer = load_finetuned_regressor(family, checkpoint)

        feat = {
            split: extract_features(
                model, tokenizer, splits[split]["response"],
                args.batch_size, args.max_length, args.device,
            )
            for split in ("train", "test", "heldout")
        }

        for combo in all_combinations():
            X_train = np.concatenate([feat["train"][k] for k in combo], axis=1)
            X_test = np.concatenate([feat["test"][k] for k in combo], axis=1)
            X_held = np.concatenate([feat["heldout"][k] for k in combo], axis=1)
            probe, alpha, cv_r = fit_ridge_probe(X_train, y["train"], n_jobs=args.n_jobs)
            test = evaluate_3sd(y["test"], probe.predict(X_test))
            held = evaluate_3sd(y["heldout"], probe.predict(X_held))
            rows.append({
                "model": name,
                "feature_combination": "+".join(combo),
                "n_feature_families": len(combo),
                "n_features": X_train.shape[1],
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
            print(
                f"{'+'.join(combo):80s} n={X_train.shape[1]:3d} "
                f"test={test['pearson']:.4f} heldout={held['pearson']:.4f}"
            )

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    path = out_dir / "representation_evolution_results.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
