import numpy as np
import warnings
import torch
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.linalg import LinAlgWarning

def evaluate(y_pred, y_true, split_name):
    y_true_arr = np.array(y_true)

    mask_pred = np.abs(y_pred - y_pred.mean()) <= 3 * y_pred.std()
    mask_true = np.abs(y_true_arr - y_true_arr.mean()) <= 3 * y_true_arr.std()
    mask = mask_pred & mask_true

    y_pred_clean = y_pred[mask]
    y_true_clean = y_true_arr[mask]
    n_removed = (~mask).sum()

    result_pearson = pearsonr(y_pred_clean, y_true_clean)
    ci_pearson = result_pearson.confidence_interval(confidence_level=0.95)
    print(
        f"[{split_name}] Pearson r = {result_pearson.statistic:.2f} [{ci_pearson.low:.2f}, {ci_pearson.high:.2f}], p = {result_pearson.pvalue:.3e} | Removed {n_removed}/{len(mask)} ({n_removed/len(mask)*100:.1f}%)"
    )
    return (
        result_pearson.statistic,
        ci_pearson.low,
        ci_pearson.high,
        result_pearson.pvalue,
        n_removed,
    )


FILES = [
    "roberta/roberta_embd_mean_std.pt",
    "roberta/roberta_embd_mean.pt",
    "roberta/roberta_embd_cls_token.pt",
    "roberta/distefano_roberta_embd_mean_std.pt",
    "roberta/distefano_roberta_embd_mean.pt",
    "roberta/distefano_roberta_embd_cls_token.pt",
    "gpt2/distefano_gpt2_embd_mean_std.pt",
    "gpt2/distefano_gpt2_embd_mean.pt",
    "gpt2/distefano_gpt2_embd_last_token.pt",
    "gpt2/gpt2_embd_mean_std.pt",
    "gpt2/gpt2_embd_mean.pt",
    "gpt2/gpt2_embd_last_token.pt"
]


test = []
heldout = []
validation = []


def pearson_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    r = pearsonr(y_true, y_pred).statistic
    return 0.0 if np.isnan(r) else float(r)


for file in FILES:

    print(f"\n\nProbes for: {file}\n")
    pipeline = Pipeline(
        [
            # ("scaler", StandardScaler()),
            ("ridge", Ridge())
        ]
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=40)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"ridge__alpha": np.logspace(-5, 6, 24)},
        scoring=make_scorer(pearson_score, greater_is_better=True),
        cv=cv,
        n_jobs=1,
        refit=True,
    )

    results = []
    train = torch.load(f"../../data/input/{file}")
    for layer in range(13):
        print("layer  ---  ", layer)

        X_train = train["train"]["embeddings"][layer].numpy()
        y_train = train["train"]["ratings"].numpy()

        search.fit(X_train, y_train)
        print("Best alpha:", search.best_params_["ridge__alpha"])
        print("Best mean CV Pearson r:", search.best_score_)

        X_test = train["test"]["embeddings"][layer].numpy().astype(np.float32)
        y_pred = search.predict(X_test)
        test_r, test_ci_l, test_ci_u, test_p, test_rem = evaluate(
            y_pred, train["test"]["ratings"].numpy(), "test"
        )
        test.append(test_r)

        X_heldout = train["heldout"]["embeddings"][layer].numpy().astype(np.float32)
        y_pred = search.predict(X_heldout)
        heldout_r, heldout_ci_l, heldout_ci_u, heldout_p, heldout_rem = evaluate(
            y_pred, train["heldout"]["ratings"].numpy(), "heldout"
        )
        heldout.append(heldout_r)

        X_val = train["validation"]["embeddings"][layer].numpy().astype(np.float32)
        y_pred = search.predict(X_val)
        val_r, val_ci_l, val_ci_u, val_p, val_rem = evaluate(
            y_pred, train["validation"]["ratings"].numpy(), "validation"
        )
        validation.append(val_r)

        results.append(
            {
                "Layer": layer,
                "Best_Alpha": search.best_params_["ridge__alpha"],
                "Test_r": test_r,
                "Test_CI_lower": test_ci_l,
                "Test_CI_upper": test_ci_u,
                "Test_p": test_p,
                "Test_removed": test_rem,
                "Heldout_r": heldout_r,
                "Heldout_CI_lower": heldout_ci_l,
                "Heldout_CI_upper": heldout_ci_u,
                "Heldout_p": heldout_p,
                "Heldout_removed": heldout_rem,
                "Validation_r": val_r,
                "Validation_CI_lower": val_ci_l,
                "Validation_CI_upper": val_ci_u,
                "Validation_p": val_p,
                "Validation_removed": val_rem,
            }
        )

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"../../data/results/hidden_state_probes_results_{file.split('/')[1].split(".")[0]}.csv", index=False)

#Seperate OpenAI text-embedding-3-large Probe

file = "openai_embeddings.pt"

print(f"\n\nProbe for: OpenAi Embeddings\n")

pipeline = Pipeline(
    [
        # ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ]
)

cv = KFold(n_splits=5, shuffle=True, random_state=40)

search = GridSearchCV(
    estimator=pipeline,
    param_grid={"ridge__alpha": np.logspace(-5, 6, 24)},
    scoring=make_scorer(pearson_score, greater_is_better=True),
    cv=cv,
    n_jobs=1,
    refit=True,
)

results = []
train = torch.load(f"../../data/input/{file}")

X_train = train["train"]["embeddings"].numpy()
y_train = train["train"]["ratings"].numpy()

search.fit(X_train, y_train)
print("Best alpha:", search.best_params_["ridge__alpha"])
print("Best mean CV Pearson r:", search.best_score_)

X_test = train["test"]["embeddings"].numpy().astype(np.float32)
y_pred = search.predict(X_test)
test_r, test_ci_l, test_ci_u, test_p, test_rem = evaluate(
    y_pred, train["test"]["ratings"].numpy(), "test"
)
test.append(test_r)

X_heldout = train["heldout"]["embeddings"].numpy().astype(np.float32)
y_pred = search.predict(X_heldout)
heldout_r, heldout_ci_l, heldout_ci_u, heldout_p, heldout_rem = evaluate(
    y_pred, train["heldout"]["ratings"].numpy(), "heldout"
)
heldout.append(heldout_r)

X_val = train["validation"]["embeddings"].numpy().astype(np.float32)
y_pred = search.predict(X_val)
val_r, val_ci_l, val_ci_u, val_p, val_rem = evaluate(
    y_pred, train["validation"]["ratings"].numpy(), "validation"
)
validation.append(val_r)

results.append(
    {
        "Best_Alpha": search.best_params_["ridge__alpha"],
        "Test_r": test_r,
        "Test_CI_lower": test_ci_l,
        "Test_CI_upper": test_ci_u,
        "Test_p": test_p,
        "Test_removed": test_rem,
        "Heldout_r": heldout_r,
        "Heldout_CI_lower": heldout_ci_l,
        "Heldout_CI_upper": heldout_ci_u,
        "Heldout_p": heldout_p,
        "Heldout_removed": heldout_rem,
        "Validation_r": val_r,
        "Validation_CI_lower": val_ci_l,
        "Validation_CI_upper": val_ci_u,
        "Validation_p": val_p,
        "Validation_removed": val_rem,
    }
)

df_results = pd.DataFrame(results)
df_results.to_csv(
    f"../../data/results/hidden_state_probes_results_{file.split(".")[0]}.csv",
    index=False,
)
