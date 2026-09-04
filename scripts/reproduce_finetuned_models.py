#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

from common import (
    evaluate_3sd,
    format_finetuned_texts,
    load_finetuned_regressor,
    load_splits,
    predict_regressor,
    resolve_prefix,
    set_all_seeds,
)


def run_one(family, checkpoint, db, study_name, explicit_prefix, splits, args):
    prefix = resolve_prefix(explicit_prefix, db, study_name)
    model, tokenizer = load_finetuned_regressor(family, checkpoint)
    max_length = args.gpt2_max_length if family == "gpt2" else args.roberta_max_length
    rows = []
    prediction_frames = []

    for split in ("test", "heldout"):
        frame = splits[split].copy()
        texts = format_finetuned_texts(frame, prefix)
        pred = predict_regressor(
            model, tokenizer, texts,
            batch_size=args.batch_size,
            max_length=max_length,
            device=args.device,
        )
        ev = evaluate_3sd(frame["label"].to_numpy(float), pred)
        rows.append({
            "model": f"{family}_finetuned",
            "split": split,
            "prefix": prefix,
            "pearson": ev["pearson"],
            "pearson_ci_low": ev["pearson_ci_low"],
            "pearson_ci_high": ev["pearson_ci_high"],
            "pearson_p": ev["pearson_p"],
            "spearman": ev["spearman"],
            "removed": ev["removed"],
            "n": ev["n"],
        })
        prediction_frames.append(pd.DataFrame({
            "model": f"{family}_finetuned",
            "split": split,
            "prompt": frame["prompt"].astype(str).to_numpy(),
            "response": frame["response"].astype(str).to_numpy(),
            "label": frame["label"].to_numpy(float),
            "prediction": pred,
            "text": texts,
        }))
    return rows, prediction_frames


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

    rows, preds = [], []
    for spec in [
        ("gpt2", args.gpt2_checkpoint, args.gpt2_study_db,
         "hp-testsearch-gpt2-seed40", args.gpt2_prefix),
        ("roberta", args.roberta_checkpoint, args.roberta_study_db,
         "hp-testsearch-bert-seed40", args.roberta_prefix),
    ]:
        r, pr = run_one(*spec, splits, args)
        rows.extend(r)
        preds.extend(pr)

    summary_path = out_dir / "finetuned_model_reproduction_results.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pred_path = out_dir / "finetuned_model_reproduction_predictions.csv"
    pd.concat(preds, ignore_index=True).to_csv(pred_path, index=False)
    print(f"Saved {summary_path}")
    print(f"Saved {pred_path}")


if __name__ == "__main__":
    main()
