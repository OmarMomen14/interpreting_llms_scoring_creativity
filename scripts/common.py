from __future__ import annotations

import inspect
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

SEED = 40
ALPHAS = np.logspace(-5, 6, 24)
PREFIXES = [
    "A creative metaphor for ",
    "A useful metaphor for ",
    "A novel metaphor for ",
    "A unqiue metaphor for ",
    "A surprising metaphor for ",
]
BASE_MODELS = {
    "gpt2": "gpt2",
    "roberta": "roberta-base",
}


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: Optional[str] = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_splits(csv_path: str | Path) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    required = {"response", "rating", "set"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metaphor.csv: {sorted(missing)}")

    df = df.copy()
    df["label"] = pd.to_numeric(df["rating"], errors="raise").astype(float)
    split_values = {
        "train": "training",
        "validation": "validation",
        "test": "test",
        "heldout": "heldout",
    }
    return {
        name: df.loc[df["set"] == value].reset_index(drop=True)
        for name, value in split_values.items()
    }


def pearson_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if (
        len(y_true) < 2
        or not np.all(np.isfinite(y_true))
        or not np.all(np.isfinite(y_pred))
        or np.isclose(np.std(y_true), 0.0)
        or np.isclose(np.std(y_pred), 0.0)
    ):
        return 0.0
    r = pearsonr(y_true, y_pred).statistic
    return 0.0 if np.isnan(r) else float(r)


PEARSON_SCORER = make_scorer(pearson_score, greater_is_better=True)


def evaluate_3sd(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    true_std = y_true.std()
    pred_std = y_pred.std()
    mask_true = np.ones(len(y_true), dtype=bool) if true_std == 0 else (
        np.abs(y_true - y_true.mean()) <= 3 * true_std
    )
    mask_pred = np.ones(len(y_pred), dtype=bool) if pred_std == 0 else (
        np.abs(y_pred - y_pred.mean()) <= 3 * pred_std
    )
    mask = mask_true & mask_pred
    yt, yp = y_true[mask], y_pred[mask]

    out = {
        "n": int(len(y_true)),
        "removed": int((~mask).sum()),
        "pearson": np.nan,
        "pearson_ci_low": np.nan,
        "pearson_ci_high": np.nan,
        "pearson_p": np.nan,
        "spearman": np.nan,
    }
    if len(yt) >= 2 and not np.isclose(yt.std(), 0) and not np.isclose(yp.std(), 0):
        pr = pearsonr(yt, yp)
        ci = pr.confidence_interval(confidence_level=0.95)
        sr = stats.spearmanr(yt, yp)
        out.update(
            pearson=float(pr.statistic),
            pearson_ci_low=float(ci.low),
            pearson_ci_high=float(ci.high),
            pearson_p=float(pr.pvalue),
            spearman=float(sr.statistic),
        )
    return out


def print_eval(split: str, result: Dict[str, float]) -> None:
    print(
        f"[{split}] Pearson r = {result['pearson']:.2f} "
        f"[{result['pearson_ci_low']:.2f}, {result['pearson_ci_high']:.2f}], "
        f"p = {result['pearson_p']:.3e} | Removed {result['removed']}/{result['n']}"
    )
    print(f"[{split}] Spearman r = {result['spearman']:.2f}")


def fit_ridge_probe(X: np.ndarray, y: np.ndarray, n_jobs: int = -1):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(solver="svd")),
    ])
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        estimator=pipe,
        param_grid={"ridge__alpha": ALPHAS},
        scoring=PEARSON_SCORER,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(np.asarray(X), np.asarray(y, dtype=float))
    return search.best_estimator_, float(search.best_params_["ridge__alpha"]), float(search.best_score_)


def evaluate_probe(model, X: np.ndarray, y: np.ndarray, split: str) -> Dict[str, float]:
    pred = model.predict(X)
    result = evaluate_3sd(y, pred)
    print_eval(split, result)
    return result


def tokenizer_for(model_family: str, finetuned_reproduction: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODELS[model_family], use_fast=True)
    if model_family == "gpt2" and tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # reproduce_.ipynb explicitly changed RoBERTa's pad token for the
    # sequence-regression reproduction; keep that behavior only there.
    if model_family == "roberta" and finetuned_reproduction:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def configure_model_padding(model, tokenizer) -> None:
    if getattr(model.config, "pad_token_id", None) is None or model.config.model_type == "gpt2":
        model.config.pad_token_id = tokenizer.pad_token_id


def load_pretrained_model(model_family: str):
    tokenizer = tokenizer_for(model_family)
    model = AutoModel.from_pretrained(BASE_MODELS[model_family])
    configure_model_padding(model, tokenizer)
    return model, tokenizer


def load_random_model(model_family: str, zero_position: bool = False):
    tokenizer = tokenizer_for(model_family)
    config = AutoConfig.from_pretrained(BASE_MODELS[model_family])
    if model_family == "gpt2":
        config.pad_token_id = tokenizer.pad_token_id
    model = AutoModel.from_config(config)
    if zero_position:
        zero_position_embeddings(model)
    configure_model_padding(model, tokenizer)
    return model, tokenizer


def zero_position_embeddings(model) -> None:
    with torch.no_grad():
        if model.config.model_type == "gpt2":
            model.wpe.weight.zero_()
        elif model.config.model_type == "roberta":
            model.embeddings.position_embeddings.weight.zero_()
        else:
            raise ValueError(f"Unsupported model type: {model.config.model_type}")


def _clean_state_dict(state_dict: dict, family: str) -> dict:
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    if "model_state_dict" in state_dict and isinstance(state_dict["model_state_dict"], dict):
        state_dict = state_dict["model_state_dict"]
    if state_dict and all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    # Legacy checkpoint buffers accepted/removed in reproduce_.ipynb.
    if family == "gpt2":
        import re
        state_dict = {
            k: v for k, v in state_dict.items()
            if not re.match(r"transformer\.h\.\d+\.attn\.(bias|masked_bias)$", k)
        }
    else:
        state_dict.pop("roberta.embeddings.position_ids", None)
        state_dict.pop("roberta.embeddings.token_type_ids", None)
    return state_dict


def load_finetuned_regressor(model_family: str, checkpoint: str | Path):
    """Load the original sequence-regression checkpoints used in reproduce_.ipynb."""
    tokenizer = tokenizer_for(model_family, finetuned_reproduction=True)
    checkpoint = Path(checkpoint)

    if checkpoint.is_dir() and (checkpoint / "config.json").exists():
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODELS[model_family], num_labels=1
        )
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location="cpu")
        state = _clean_state_dict(state, model_family)
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state.keys())
        missing = sorted(model_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - model_keys)
        if missing:
            raise RuntimeError("Checkpoint is missing model parameters: " + ", ".join(missing))
        if unexpected:
            raise RuntimeError("Checkpoint has unexpected parameters: " + ", ".join(unexpected))
        model.load_state_dict(state, strict=True)

    model.config.problem_type = "regression"
    configure_model_padding(model, tokenizer)
    return model, tokenizer


def get_backbone(model):
    # AutoModel returns itself as base_model; sequence classifiers expose their base.
    return model.base_model


def _forward_hidden_states(backbone, batch):
    kwargs = dict(batch)
    kwargs.update(output_hidden_states=True, return_dict=True)
    sig = inspect.signature(backbone.forward)
    if "use_cache" in sig.parameters:
        kwargs["use_cache"] = False
    return backbone(**kwargs).hidden_states


def _ordinary_token_ids(tokenizer) -> np.ndarray:
    special = set(tokenizer.all_special_ids)
    return np.asarray([i for i in range(len(tokenizer)) if i not in special], dtype=np.int64)


def extract_layer_features(
    model,
    tokenizer,
    texts: Iterable[str],
    *,
    pooling: str = "mean",
    batch_size: int = 32,
    max_length: int = 256,
    device: Optional[str] = None,
    add_special_tokens: bool = True,
    randomize_input_token_ids: bool = False,
    random_seed: Optional[int] = SEED,
) -> Dict[int, np.ndarray]:
    """Extract one sentence vector per layer (embedding output + 12 blocks)."""
    device_obj = get_device(device)
    model = model.to(device_obj).eval()
    backbone = get_backbone(model)
    texts = list(map(str, texts))
    by_layer: Optional[list[list[np.ndarray]]] = None
    ordinary_ids = _ordinary_token_ids(tokenizer) if randomize_input_token_ids else None
    random_generator = None
    if randomize_input_token_ids and random_seed is not None:
        random_generator = torch.Generator(device="cpu")
        random_generator.manual_seed(int(random_seed))

    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=add_special_tokens,
            )
            if randomize_input_token_ids:
                ids = enc["input_ids"].clone()
                mask = enc["attention_mask"].bool()
                ordinary_tensor = torch.as_tensor(ordinary_ids, dtype=torch.long)
                for b in range(ids.shape[0]):
                    positions = torch.where(mask[b])[0]
                    n = int(len(positions))
                    pool_indices = torch.randint(
                        low=0, high=len(ordinary_tensor), size=(n,),
                        generator=random_generator, dtype=torch.long,
                    )
                    ids[b, positions] = ordinary_tensor[pool_indices].to(ids.dtype)
                enc["input_ids"] = ids

            enc = {k: v.to(device_obj) for k, v in enc.items()}
            hidden_states = _forward_hidden_states(backbone, enc)
            if by_layer is None:
                by_layer = [[] for _ in hidden_states]

            attn = enc["attention_mask"]
            mask = attn.unsqueeze(-1).to(torch.float32)
            counts = mask.sum(1).clamp_min(1.0)

            for li, h in enumerate(hidden_states):
                h = h.float()
                if pooling == "mean":
                    pooled = (h * mask).sum(1) / counts
                elif pooling == "mean_std":
                    mean = (h * mask).sum(1) / counts
                    var = (((h - mean.unsqueeze(1)) ** 2) * mask).sum(1) / counts
                    std = torch.sqrt(var.clamp_min(0.0))
                    pooled = torch.cat([mean, std], dim=-1)
                elif pooling == "last_cls":
                    if model.config.model_type == "gpt2":
                        idx = attn.sum(1).long().clamp_min(1) - 1
                        pooled = h[torch.arange(h.shape[0], device=h.device), idx]
                    elif model.config.model_type == "roberta":
                        pooled = h[:, 0, :]
                    else:
                        raise ValueError("last_cls only supports GPT-2/RoBERTa")
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")
                by_layer[li].append(pooled.cpu().numpy())

    assert by_layer is not None
    return {i: np.concatenate(parts, axis=0) for i, parts in enumerate(by_layer)}


def run_layerwise_ridge(
    train_features: Dict[int, np.ndarray],
    test_features: Dict[int, np.ndarray],
    heldout_features: Dict[int, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_heldout: np.ndarray,
    *,
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, Dict[int, object]]:
    rows = []
    fitted = {}
    for layer in sorted(train_features):
        model, alpha, cv_r = fit_ridge_probe(train_features[layer], y_train, n_jobs=n_jobs)
        fitted[layer] = model
        test = evaluate_3sd(y_test, model.predict(test_features[layer]))
        held = evaluate_3sd(y_heldout, model.predict(heldout_features[layer]))
        rows.append({
            "layer_index": layer,
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
            f"layer={layer:02d} alpha={alpha:.6g} CV={cv_r:.4f} "
            f"test={test['pearson']:.4f} heldout={held['pearson']:.4f}"
        )
    return pd.DataFrame(rows), fitted


def resolve_prefix(
    explicit_prefix: Optional[str],
    optuna_db: Optional[str | Path] = None,
    study_name: Optional[str] = None,
) -> str:
    if explicit_prefix is not None:
        return explicit_prefix
    if optuna_db is None:
        raise ValueError("Provide --prefix or an Optuna database containing prefix_index")
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("optuna is required to read the study database") from exc
    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{Path(optuna_db).resolve()}",
    )
    idx = int(study.best_params["prefix_index"])
    return PREFIXES[idx]


def format_finetuned_texts(df: pd.DataFrame, prefix: str) -> list[str]:
    if "prompt" not in df.columns:
        raise ValueError("metaphor.csv must contain 'prompt' for finetuned reproduction")
    return [
        f"{prefix}{prompt} is: {response}"
        for prompt, response in zip(df["prompt"].astype(str), df["response"].astype(str))
    ]


def predict_regressor(
    model,
    tokenizer,
    texts: Iterable[str],
    *,
    batch_size: int = 16,
    max_length: int = 256,
    device: Optional[str] = None,
) -> np.ndarray:
    device_obj = get_device(device)
    model = model.to(device_obj).eval()
    preds = []
    texts = list(texts)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device_obj) for k, v in enc.items()}
            logits = model(**enc, return_dict=True).logits.squeeze(-1)
            preds.extend(logits.detach().cpu().numpy().tolist())
    return np.asarray(preds, dtype=float)
