#!/usr/bin/env python3
"""Response-token order ablation for the finetuned GPT-2/RoBERTa models."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    evaluate_3sd,
    get_device,
    load_finetuned_regressor,
    load_splits,
    resolve_prefix,
    set_all_seeds,
)


def build_split(df, prefix):
    out = df[["prompt", "response", "label"]].copy().reset_index(drop=True)
    out["fixed_prompt"] = prefix + out["prompt"].astype(str) + " is:"
    out["response_segment"] = " " + out["response"].astype(str)
    out["text"] = out["fixed_prompt"] + out["response_segment"]
    out["response_char_start"] = out["fixed_prompt"].str.len()
    return out


def tokenize_and_mark(df, tokenizer, max_length):
    enc = tokenizer(
        df["text"].tolist(),
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")
    special = enc.pop("special_tokens_mask")
    starts = torch.tensor(df["response_char_start"].to_numpy(), dtype=torch.long)
    response_mask = (
        enc["attention_mask"].bool()
        & ~special.bool()
        & (offsets[:, :, 1] > starts.unsqueeze(1))
    )
    enc["response_token_mask"] = response_mask
    return enc


def shuffle_response(input_ids, response_mask, seed=40):
    original = input_ids.cpu()
    response_mask = response_mask.cpu()
    shuffled = original.clone()
    for i in range(len(original)):
        positions = torch.nonzero(response_mask[i], as_tuple=False).flatten()
        if len(positions) <= 1:
            continue
        g = torch.Generator(device="cpu")
        g.manual_seed(seed + i)
        perm = torch.randperm(len(positions), generator=g)
        shuffled[i, positions] = original[i, positions][perm]
    return shuffled


def predict_ids(model, ids, attention, batch_size, device):
    dev = get_device(device)
    model = model.to(dev).eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            logits = model(
                input_ids=ids[start:start + batch_size].to(dev),
                attention_mask=attention[start:start + batch_size].to(dev),
                return_dict=True,
            ).logits.squeeze(-1)
            out.append(logits.cpu().numpy())
    return np.concatenate(out)


def run_family(family, checkpoint, db, study_name, explicit_prefix, splits, args):
    prefix = resolve_prefix(explicit_prefix, db, study_name)
    model, tokenizer = load_finetuned_regressor(family, checkpoint)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for response offset mapping")
    max_length = args.gpt2_max_length if family == "gpt2" else args.roberta_max_length
    rows, predictions = [], []

    for split_name in ("test", "heldout"):
        frame = build_split(splits[split_name], prefix)
        enc = tokenize_and_mark(frame, tokenizer, max_length)
        shuffled_ids = shuffle_response(enc["input_ids"], enc["response_token_mask"], seed=40)
        original_pred = predict_ids(model, enc["input_ids"], enc["attention_mask"], args.batch_size, args.device)
        shuffled_pred = predict_ids(model, shuffled_ids, enc["attention_mask"], args.batch_size, args.device)

        for condition, pred in (("original_order", original_pred), ("response_tokens_shuffled", shuffled_pred)):
            ev = evaluate_3sd(frame["label"].to_numpy(float), pred)
            rows.append({
                "model": f"{family}_finetuned",
                "split": split_name,
                "condition": condition,
                "shuffle_seed": 40 if condition != "original_order" else np.nan,
                "pearson": ev["pearson"],
                "pearson_ci_low": ev["pearson_ci_low"],
                "pearson_ci_high": ev["pearson_ci_high"],
                "pearson_p": ev["pearson_p"],
                "spearman": ev["spearman"],
                "removed": ev["removed"],
                "n": ev["n"],
            })

        # Store decoded shuffled input for auditable examples/results.
        shuffled_text = []
        for ids, attn in zip(shuffled_ids, enc["attention_mask"]):
            shuffled_text.append(tokenizer.decode(ids[attn.bool()].tolist(), skip_special_tokens=True))
        predictions.append(pd.DataFrame({
            "model": f"{family}_finetuned",
            "split": split_name,
            "prompt": frame["prompt"],
            "response": frame["response"],
            "label": frame["label"],
            "original_prediction": original_pred,
            "shuffled_prediction": shuffled_pred,
            "original_text": frame["text"],
            "shuffled_text": shuffled_text,
        }))
    return rows, predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/input/metaphor.csv")
    p.add_argument("--gpt2-checkpoint", default="data/input/gpt2/pytorch_model.bin")
    p.add_argument("--roberta-checkpoint", default="data/input/roberta/pytorch_model.bin")
    p.add_argument("--gpt2-study-db", default="data/input/gpt2/hp-testsearch-gpt2-seed40.db")
    p.add_argument("--roberta-study-db", default="data/input/roberta/hp-testsearch-bert-seed40.db")
    p.add_argument("--gpt2-prefix", default=None)
    p.add_argument("--roberta-prefix", default=None)
    p.add_argument("--output-dir", default="data/results")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gpt2-max-length", type=int, default=256)
    p.add_argument("--roberta-max-length", type=int, default=512)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    set_all_seeds(40)
    splits = load_splits(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, pred_frames = [], []

    for spec in [
        ("gpt2", args.gpt2_checkpoint, args.gpt2_study_db,
         "hp-testsearch-gpt2-seed40", args.gpt2_prefix),
        ("roberta", args.roberta_checkpoint, args.roberta_study_db,
         "hp-testsearch-bert-seed40", args.roberta_prefix),
    ]:
        r, pframes = run_family(*spec, splits, args)
        rows.extend(r)
        pred_frames.extend(pframes)

    summary = out_dir / "finetuned_token_order_ablation_results.csv"
    preds = out_dir / "finetuned_token_order_ablation_predictions.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(preds, index=False)
    print(f"Saved {summary}")
    print(f"Saved {preds}")


if __name__ == "__main__":
    main()
