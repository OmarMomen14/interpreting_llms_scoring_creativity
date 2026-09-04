#!/usr/bin/env python3

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch

from common import (
    extract_layer_features,
    load_finetuned_regressor,
    load_pretrained_model,
    load_splits,
    run_layerwise_ridge,
    set_all_seeds,
)


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
    y = {k: v["label"].to_numpy(float) for k, v in splits.items()}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_specs = [
        ("gpt2_pretrained", "gpt2", None),
        ("roberta_pretrained", "roberta", None),
        ("gpt2_finetuned", "gpt2", args.gpt2_checkpoint),
        ("roberta_finetuned", "roberta", args.roberta_checkpoint),
    ]
    all_rows = []

    for model_name, family, checkpoint in model_specs:
        print(f"\n### {model_name}")
        if checkpoint is None:
            model, tokenizer = load_pretrained_model(family)
        else:
            model, tokenizer = load_finetuned_regressor(family, checkpoint)

        for pooling in ("mean", "mean_std", "last_cls"):
            print(f"\n## pooling={pooling}")
            features = {}
            for split in ("train", "test", "heldout"):
                features[split] = extract_layer_features(
                    model,
                    tokenizer,
                    splits[split]["response"].astype(str),
                    pooling=pooling,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device=args.device,
                )
            result, _ = run_layerwise_ridge(
                features["train"], features["test"], features["heldout"],
                y["train"], y["test"], y["heldout"], n_jobs=args.n_jobs,
            )
            result.insert(0, "pooling", pooling)
            result.insert(0, "model", model_name)
            all_rows.append(result)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = pd.concat(all_rows, ignore_index=True)
    path = out_dir / "hidden_representation_probe_results.csv"
    output.to_csv(path, index=False)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
