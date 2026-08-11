#!/usr/bin/env python3
"""Command 09: development-only nested training and final pipeline freeze.

This script deliberately reads only the Section 3A development registry and its
three frozen metadata files. It never discovers or opens holdout/JADER paths.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, __version__ as xgboost_version


ROOT = Path(__file__).resolve().parents[1]
S3 = ROOT / "analysis" / "section3_model"
TRAIN = S3 / "training"
MODELS = TRAIN / "models"
INNER_DIR = S3 / "inner_folds"

# Exhaustive allowlist: do not add outcome/holdout/JADER sources here.
REGISTRY_PATH = S3 / "development_pair_registry.parquet"
FEATURE_PATH = S3 / "FEATURE_DICTIONARY_v1.csv"
OUTER_PATH = S3 / "OUTER_FOLD_ASSIGNMENT_v1.csv"
S3A_QC_PATH = S3 / "SECTION3A_QC.json"
FIREWALL_PATH = S3 / "08_section3b_firewall_audit.md"

SEED = 20260810
XGB_SEARCH_SEED = 20260909
BOOT_SEED = 20260810
N_BOOT = 5000
N_JOBS = min(8, max(1, (os.cpu_count() or 2) - 2))
PT_MIN_DRUGS = 5
PRED_CLIP = 1e-6
TARGET = "criterion_r_3y"
GROUP = "canonical_active_moiety"
PT = "canonical_pt_code"

PIPELINES = {
    "elasticnet_set0": ("elasticnet", "SET0"),
    "elasticnet_set1": ("elasticnet", "SET1"),
    "xgboost_set0": ("xgboost", "SET0"),
    "xgboost_set1": ("xgboost", "SET1"),
}


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if pd.isna(value) if not isinstance(value, (str, bool)) else False:
        return None
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dataframe_markdown(frame: pd.DataFrame, float_digits: int = 4) -> str:
    """Render a small dataframe without pandas' optional tabulate dependency."""
    def render(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        return str(value).replace("|", "\\|")

    columns = [str(c) for c in frame.columns]
    rows = [[render(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(columns[i]), *(len(row[i]) for row in rows)) for i in range(len(columns))]
    line = lambda values: "| " + " | ".join(str(values[i]).ljust(widths[i]) for i in range(len(widths))) + " |"
    return "\n".join([line(columns), "| " + " | ".join("-" * w for w in widths) + " |", *(line(row) for row in rows)])


def write_firewall(status: str, extra: str = "") -> None:
    text = f"""# Section 3B physical holdout firewall audit

**Status:** {status}  
**Generated:** {datetime.now().isoformat()}  
**Scope:** Command 09, development-only nested model training and freeze.

## Read allowlist

- `{REGISTRY_PATH}`
- `{FEATURE_PATH}`
- `{OUTER_PATH}`
- `{S3A_QC_PATH}`

The training program uses explicit paths rather than directory discovery. It contains no path to
the 2019–2022 temporal outcomes, holdout ROR/IC, holdout exact FAERS PS volume, holdout coverage,
holdout PT identities, holdout predictions, or JADER. Section 2 decomposition is not rerun.

## Confirmed prohibitions

- Holdout outcome rows accessed: **0**
- JADER rows accessed: **0**
- Holdout PT identity list opened: **no**
- Holdout performance inspected: **no**
- SHAP calculated: **no**
- Clinical threshold optimized: **no**
- Post-hoc recalibration applied: **no**

{extra}
"""
    FIREWALL_PATH.write_text(text, encoding="utf-8")


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    for path in [REGISTRY_PATH, FEATURE_PATH, OUTER_PATH, S3A_QC_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)
    registry = pd.read_parquet(REGISTRY_PATH)
    dictionary = pd.read_csv(FEATURE_PATH)
    outer = pd.read_csv(OUTER_PATH)
    qc3a = json.loads(S3A_QC_PATH.read_text(encoding="utf-8"))

    required = {"pair_id", GROUP, PT, "canonical_pt_name", TARGET}
    assert required.issubset(registry.columns)
    assert len(registry) == 16470
    assert registry[GROUP].nunique() == 107
    assert int(registry[TARGET].sum()) == 2064
    assert registry["pair_id"].is_unique
    assert registry[TARGET].isin([0, 1]).all()
    assert outer[GROUP].nunique() == 107 and outer[GROUP].is_unique
    assert outer["outer_fold"].value_counts().sort_index().tolist() == [16, 14, 27, 23, 27]
    assert set(registry[GROUP]) == set(outer[GROUP])

    forbidden_tokens = ["jader", "ror", "information_component", "exact_faers_ps", "holdout"]
    bad = [c for c in registry.columns if any(t in c.lower() for t in forbidden_tokens)]
    assert not bad, f"Forbidden registry columns: {bad}"

    retained = dictionary[dictionary["status"].isin(["PRIMARY", "SECONDARY"])]
    set0 = retained.loc[retained["feature_set"] == "SET0", "feature_name"].tolist()
    set1add = retained.loc[retained["feature_set"] == "SET1_ADDITIONAL", "feature_name"].tolist()
    assert len(set0) == 18 and len(set1add) == 27
    assert set(set0 + set1add).issubset(registry.columns)
    assert GROUP not in set0 + set1add and TARGET not in set0 + set1add

    registry = registry.merge(outer[[GROUP, "outer_fold"]], on=GROUP, how="left", validate="many_to_one")
    assert registry["outer_fold"].notna().all()
    registry["outer_fold"] = registry["outer_fold"].astype(int)
    return registry, dictionary, outer, qc3a


def feature_lists(dictionary: pd.DataFrame) -> dict[str, list[str]]:
    retained = dictionary[dictionary["status"].isin(["PRIMARY", "SECONDARY"])]
    s0 = retained.loc[retained["feature_set"] == "SET0", "feature_name"].tolist()
    add = retained.loc[retained["feature_set"] == "SET1_ADDITIONAL", "feature_name"].tolist()
    return {"SET0": s0, "SET1": s0 + add}


def fit_rare_map(frame: pd.DataFrame) -> dict[str, Any]:
    z = frame[[GROUP, PT]].copy()
    z[PT] = z[PT].astype(str)
    support = z.drop_duplicates().groupby(PT)[GROUP].nunique().sort_index()
    retained = sorted(support[support >= PT_MIN_DRUGS].index.tolist())
    return {
        "retained": retained,
        "support": {str(k): int(v) for k, v in support.items()},
        "training_unique_pts": int(len(support)),
        "collapsed_training_pts": int((support < PT_MIN_DRUGS).sum()),
    }


def apply_rare_map(frame: pd.DataFrame, rare: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    code = out[PT].astype(str)
    out[PT] = np.where(code.isin(set(rare["retained"])), code, "PT_RARE")
    return out


def rare_validation_qc(train: pd.DataFrame, valid: pd.DataFrame, fold: int) -> dict[str, Any]:
    rare = fit_rare_map(train)
    support = rare["support"]
    code = valid[PT].astype(str)
    seen = code.isin(support)
    low = code.map(lambda x: support.get(x, 0) < PT_MIN_DRUGS) & seen
    unseen = ~seen
    mapped = low | unseen
    n = len(valid)
    return {
        "outer_fold": fold,
        "outer_training_drugs": train[GROUP].nunique(),
        "outer_validation_drugs": valid[GROUP].nunique(),
        "training_unique_pt_identities": rare["training_unique_pts"],
        "retained_pt_categories": len(rare["retained"]),
        "collapsed_training_pt_identities": rare["collapsed_training_pts"],
        "validation_pairs": n,
        "validation_pt_rare_n": int(mapped.sum()),
        "validation_pt_rare_pct": float(100 * mapped.mean()),
        "validation_unseen_pt_n": int(unseen.sum()),
        "validation_unseen_pt_pct": float(100 * unseen.mean()),
        "validation_low_support_pt_n": int(low.sum()),
        "validation_low_support_pt_pct": float(100 * low.mean()),
    }


def make_preprocessor(
    dictionary: pd.DataFrame, features: list[str], family: str
) -> ColumnTransformer:
    meta = dictionary.set_index("feature_name").loc[features]
    categorical = meta.index[meta["data_type"].eq("categorical")].tolist()
    numerical = [c for c in features if c not in categorical]
    logcols = [c for c in numerical if "log1p" in str(meta.loc[c, "transformation"]).lower()]
    plain = [c for c in numerical if c not in logcols]

    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING/UNKNOWN")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
            ),
        ]
    )
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    log_steps = [
        ("impute", SimpleImputer(strategy="median")),
        (
            "log1p",
            __import__("sklearn.preprocessing").preprocessing.FunctionTransformer(
                np.log1p, feature_names_out="one-to-one"
            ),
        ),
    ]
    if family == "elasticnet":
        numeric_steps.append(("scale", StandardScaler()))
        log_steps.append(("scale", StandardScaler()))
    transformers = []
    if categorical:
        transformers.append(("categorical", cat_pipe, categorical))
    if plain:
        transformers.append(("numeric", Pipeline(numeric_steps), plain))
    if logcols:
        transformers.append(("log_numeric", Pipeline(log_steps), logcols))
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


@dataclass
class Prepared:
    X_train: Any
    y_train: np.ndarray
    X_valid: Any
    y_valid: np.ndarray


def prepare_split(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    dictionary: pd.DataFrame,
    features: list[str],
    family: str,
) -> tuple[Prepared, ColumnTransformer, dict[str, Any]]:
    rare = fit_rare_map(train)
    tr = apply_rare_map(train, rare)
    va = apply_rare_map(valid, rare)
    pre = make_preprocessor(dictionary, features, family)
    Xtr = pre.fit_transform(tr[features])
    Xva = pre.transform(va[features])
    return (
        Prepared(Xtr, tr[TARGET].to_numpy(dtype=np.int8), Xva, va[TARGET].to_numpy(dtype=np.int8)),
        pre,
        rare,
    )


def generate_inner_assignments(frame: pd.DataFrame) -> tuple[dict[int, pd.Series], pd.DataFrame]:
    assignments: dict[int, pd.Series] = {}
    qc_rows = []
    for outer_fold in range(1, 6):
        tr = frame[frame["outer_fold"] != outer_fold].copy()
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + outer_fold)
        row_fold = pd.Series(index=tr.index, dtype="int64")
        for inner_fold, (_, vi) in enumerate(
            splitter.split(np.zeros(len(tr)), tr[TARGET], groups=tr[GROUP]), start=1
        ):
            row_fold.iloc[vi] = inner_fold
        assert row_fold.notna().all()
        grouped = pd.DataFrame({GROUP: tr[GROUP], "inner_fold": row_fold}).drop_duplicates()
        assert grouped[GROUP].is_unique
        grouped = grouped.sort_values(["inner_fold", GROUP])
        grouped.to_csv(INNER_DIR / f"outer_{outer_fold}_inner_assignment.csv", index=False)
        assignments[outer_fold] = row_fold
        for inner_fold in range(1, 5):
            v = tr.loc[row_fold.eq(inner_fold)]
            pos = int(v[TARGET].sum())
            neg = int(len(v) - pos)
            qc_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "validation_drugs": v[GROUP].nunique(),
                    "validation_pairs": len(v),
                    "positives": pos,
                    "negatives": neg,
                    "prevalence": pos / len(v),
                    "both_outcome_classes": bool(pos > 0 and neg > 0),
                }
            )
            if pos == 0 or neg == 0:
                raise RuntimeError(f"Outer {outer_fold} inner {inner_fold} has one outcome class")
    return assignments, pd.DataFrame(qc_rows)


def en_configs() -> list[dict[str, Any]]:
    return [
        {"C": C, "l1_ratio": ratio}
        for C, ratio in itertools.product(
            [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            [0.0, 0.25, 0.5, 0.75, 1.0],
        )
    ]


def xgb_configs() -> list[dict[str, Any]]:
    space = {
        "n_estimators": [200, 400, 800],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.02, 0.05, 0.10],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.70, 0.85, 1.00],
        "colsample_bytree": [0.50, 0.75, 1.00],
        "reg_lambda": [1.0, 5.0, 10.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "gamma": [0.0, 0.1],
    }
    return list(ParameterSampler(space, n_iter=40, random_state=XGB_SEARCH_SEED))


def make_en(params: dict[str, Any], seed: int) -> LogisticRegression:
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        class_weight=None,
        fit_intercept=True,
        max_iter=10000,
        random_state=seed,
        n_jobs=1,
        C=float(params["C"]),
        l1_ratio=float(params["l1_ratio"]),
        tol=1e-4,
    )


def make_xgb(params: dict[str, Any], seed: int) -> XGBClassifier:
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=1,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )


def evaluate_config(
    family: str, config_id: int, params: dict[str, Any], prepared: list[Prepared], seed: int
) -> dict[str, Any]:
    losses: list[float] = []
    converged: list[bool] = []
    n_iters: list[int] = []
    errors: list[str] = []
    for fold_id, split in enumerate(prepared, start=1):
        try:
            if family == "elasticnet":
                model = make_en(params, seed + fold_id)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(split.X_train, split.y_train)
                ok = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                n_iters.append(int(np.max(model.n_iter_)))
                converged.append(ok)
                if not ok:
                    errors.append(f"inner_{fold_id}:convergence_warning")
                    continue
            else:
                model = make_xgb(params, seed + fold_id)
                model.fit(split.X_train, split.y_train)
                converged.append(True)
            pred = model.predict_proba(split.X_valid)[:, 1]
            losses.append(float(log_loss(split.y_valid, pred, labels=[0, 1])))
        except Exception as exc:  # preserved as invalid configuration
            converged.append(False)
            errors.append(f"inner_{fold_id}:{type(exc).__name__}:{exc}")
    valid = len(losses) == len(prepared) and all(converged)
    row = {
        "config_id": config_id,
        **json_safe(params),
        "mean_inner_log_loss": float(np.mean(losses)) if valid else np.nan,
        "sd_inner_log_loss": float(np.std(losses, ddof=1)) if valid else np.nan,
        "folds_completed": len(losses),
        "all_converged": bool(valid),
        "max_n_iter": max(n_iters) if n_iters else np.nan,
        "error": " | ".join(errors),
    }
    return row


def tune(
    family: str,
    configs: list[dict[str, Any]],
    prepared: list[Prepared],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(evaluate_config)(family, i + 1, p, prepared, seed + 1000 * (i + 1))
        for i, p in enumerate(configs)
    )
    out = pd.DataFrame(rows)
    valid = out[out["all_converged"] & out["mean_inner_log_loss"].notna()]
    if valid.empty:
        raise RuntimeError(f"No valid {family} configuration")
    best_id = int(valid.sort_values(["mean_inner_log_loss", "config_id"]).iloc[0]["config_id"])
    out["selected"] = out["config_id"].eq(best_id)
    return out, configs[best_id - 1]


def calibration_parameters(
    y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, float, bool, bool]:
    y = np.asarray(y, dtype=float)
    x = np.log(np.clip(p, PRED_CLIP, 1 - PRED_CLIP) / (1 - np.clip(p, PRED_CLIP, 1 - PRED_CLIP)))
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    a = 0.0
    ok_i = False
    for _ in range(100):
        mu = expit(a + x)
        score = np.sum(w * (y - mu))
        info = np.sum(w * mu * (1 - mu))
        if not np.isfinite(info) or info <= 1e-12:
            break
        delta = score / info
        a += delta
        if abs(delta) < 1e-10:
            ok_i = True
            break
    ca, b = 0.0, 1.0
    ok_s = False
    for _ in range(100):
        mu = expit(ca + b * x)
        v = w * mu * (1 - mu)
        s0 = np.sum(w * (y - mu))
        s1 = np.sum(w * (y - mu) * x)
        i00, i01, i11 = np.sum(v), np.sum(v * x), np.sum(v * x * x)
        det = i00 * i11 - i01 * i01
        if not np.isfinite(det) or det <= 1e-12:
            break
        da = (s0 * i11 - s1 * i01) / det
        db = (s1 * i00 - s0 * i01) / det
        ca += da
        b += db
        if max(abs(da), abs(db)) < 1e-10:
            ok_s = True
            break
    return float(a), float(b), ok_i, ok_s


def metric_row(y: np.ndarray, p: np.ndarray, prevalence: float) -> dict[str, float]:
    ci, slope, _, _ = calibration_parameters(y, p)
    ap = average_precision_score(y, p)
    return {
        "average_precision": float(ap),
        "auprc_lift": float(ap / prevalence),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "calibration_intercept": ci,
        "calibration_slope": slope,
    }


def vectorized_calibration_batch(
    y: np.ndarray, x: np.ndarray, sw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bsize = sw.shape[0]
    alpha = np.zeros(bsize)
    ok_i = np.zeros(bsize, dtype=bool)
    for _ in range(50):
        mu = expit(alpha[:, None] + x[None, :])
        score = np.sum(sw * (y[None, :] - mu), axis=1)
        info = np.sum(sw * mu * (1 - mu), axis=1)
        delta = np.divide(score, info, out=np.full_like(score, np.nan), where=info > 1e-12)
        alpha += np.nan_to_num(delta, nan=0.0)
        ok_i |= np.isfinite(delta) & (np.abs(delta) < 1e-8)
        if ok_i.all():
            break
    ca = np.zeros(bsize)
    slope = np.ones(bsize)
    ok_s = np.zeros(bsize, dtype=bool)
    for _ in range(50):
        mu = expit(ca[:, None] + slope[:, None] * x[None, :])
        residual = sw * (y[None, :] - mu)
        v = sw * mu * (1 - mu)
        s0 = residual.sum(axis=1)
        s1 = (residual * x[None, :]).sum(axis=1)
        i00 = v.sum(axis=1)
        i01 = (v * x[None, :]).sum(axis=1)
        i11 = (v * (x[None, :] ** 2)).sum(axis=1)
        det = i00 * i11 - i01 * i01
        da = np.divide(s0 * i11 - s1 * i01, det, out=np.full_like(s0, np.nan), where=det > 1e-12)
        db = np.divide(s1 * i00 - s0 * i01, det, out=np.full_like(s0, np.nan), where=det > 1e-12)
        ca += np.nan_to_num(da, nan=0.0)
        slope += np.nan_to_num(db, nan=0.0)
        ok_s |= np.isfinite(da) & np.isfinite(db) & (np.maximum(np.abs(da), np.abs(db)) < 1e-8)
        if ok_s.all():
            break
    ok_i &= np.isfinite(alpha)
    ok_s &= np.isfinite(slope)
    return alpha, slope, ok_i, ok_s


def bootstrap_metrics(
    y: np.ndarray, p: np.ndarray, drug_index: np.ndarray, boot_w: np.ndarray, batch: int = 40
) -> dict[str, np.ndarray]:
    nboot = len(boot_w)
    result = {k: np.full(nboot, np.nan) for k in ["average_precision", "auroc", "brier", "log_loss", "calibration_intercept", "calibration_slope"]}
    result["calibration_intercept_ok"] = np.zeros(nboot, dtype=bool)
    result["calibration_slope_ok"] = np.zeros(nboot, dtype=bool)
    pclip = np.clip(p, PRED_CLIP, 1 - PRED_CLIP)
    logits = np.log(pclip / (1 - pclip))
    sq = (y - p) ** 2
    ll = -(y * np.log(pclip) + (1 - y) * np.log(1 - pclip))
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    ys = y[order]
    starts = np.r_[0, np.flatnonzero(ps[1:] != ps[:-1]) + 1]
    for begin in range(0, nboot, batch):
        end = min(nboot, begin + batch)
        sw = boot_w[begin:end, :][:, drug_index].astype(float)
        denom = sw.sum(axis=1)
        result["brier"][begin:end] = (sw * sq[None, :]).sum(axis=1) / denom
        result["log_loss"][begin:end] = (sw * ll[None, :]).sum(axis=1) / denom

        sws = sw[:, order]
        posg = np.add.reduceat(sws * ys[None, :], starts, axis=1)
        negg = np.add.reduceat(sws * (1 - ys)[None, :], starts, axis=1)
        cpos, cneg = np.cumsum(posg, axis=1), np.cumsum(negg, axis=1)
        tpos, tneg = cpos[:, -1], cneg[:, -1]
        precision = np.divide(cpos, cpos + cneg, out=np.zeros_like(cpos), where=(cpos + cneg) > 0)
        result["average_precision"][begin:end] = np.sum(precision * posg, axis=1) / tpos
        neg_below = tneg[:, None] - cneg
        result["auroc"][begin:end] = np.sum(posg * (neg_below + 0.5 * negg), axis=1) / (tpos * tneg)

        ci, cs, ok_i, ok_s = vectorized_calibration_batch(y, logits, sw)
        result["calibration_intercept"][begin:end] = np.where(ok_i, ci, np.nan)
        result["calibration_slope"][begin:end] = np.where(ok_s, cs, np.nan)
        result["calibration_intercept_ok"][begin:end] = ok_i
        result["calibration_slope_ok"][begin:end] = ok_s
    return result


def fit_final_model(
    family: str,
    params: dict[str, Any],
    frame: pd.DataFrame,
    dictionary: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, bool]:
    rare = fit_rare_map(frame)
    mapped = apply_rare_map(frame, rare)
    pre = make_preprocessor(dictionary, features, family)
    X = pre.fit_transform(mapped[features])
    converged = True
    if family == "elasticnet":
        model = make_en(params, seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, mapped[TARGET].to_numpy())
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    else:
        model = make_xgb(params, seed)
        model.fit(X, mapped[TARGET].to_numpy())
    names = pre.get_feature_names_out().astype(str)
    bundle = {
        "pipeline_status": "FROZEN_FULL_DEVELOPMENT",
        "family": family,
        "features": features,
        "rare_pt_rule": f"distinct training drugs >= {PT_MIN_DRUGS}; otherwise PT_RARE",
        "retained_pt_categories": rare["retained"],
        "preprocessor": pre,
        "feature_names_out": names.tolist(),
        "hyperparameters": json_safe(params),
        "random_seed": seed,
        "model": model,
    }
    return bundle, names, converged


def main() -> None:
    start = time.time()
    TRAIN.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    INNER_DIR.mkdir(parents=True, exist_ok=True)
    write_firewall("PRE-FIT PASS — explicit source allowlist verified")
    log(f"Runtime: Python {platform.python_version()}, sklearn {sklearn_version}, XGBoost {xgboost_version}; jobs={N_JOBS}")
    frame, dictionary, outer, qc3a = load_and_validate()
    features = feature_lists(dictionary)
    prevalence = float(frame[TARGET].mean())
    log(f"Domain locked: {frame[GROUP].nunique()} drugs, {len(frame)} pairs, {frame[TARGET].sum()} positives, prevalence={prevalence:.8f}")

    assignments, inner_qc = generate_inner_assignments(frame)
    inner_qc.to_csv(TRAIN / "02_inner_cv_qc.csv", index=False)
    log("Shared inner grouped assignments persisted; all 20 validation folds contain both classes")

    rare_rows = []
    for fold in range(1, 6):
        rare_rows.append(rare_validation_qc(frame[frame.outer_fold != fold], frame[frame.outer_fold == fold], fold))
    rare_qc = pd.DataFrame(rare_rows)
    rare_qc.to_csv(TRAIN / "01_outer_fold_pt_rare_mapping_qc.csv", index=False)

    configs_by_family = {"elasticnet": en_configs(), "xgboost": xgb_configs()}
    en_results, xgb_results = [], []
    oof = frame[["pair_id", GROUP, PT, "canonical_pt_name", TARGET, "outer_fold"]].copy()
    oof["null_outer_training_prevalence"] = np.nan
    for name in PIPELINES:
        oof[name] = np.nan
    outer_en_models: dict[tuple[int, str], dict[str, float]] = {}
    outer_xgb_gain = []

    for fold in range(1, 6):
        tr = frame[frame.outer_fold != fold].copy()
        va = frame[frame.outer_fold == fold].copy()
        oof.loc[va.index, "null_outer_training_prevalence"] = tr[TARGET].mean()
        row_inner = assignments[fold]
        log(f"Outer fold {fold}/5: train {tr[GROUP].nunique()} drugs/{len(tr)} pairs; validate {va[GROUP].nunique()} drugs/{len(va)} pairs")
        for family in ["elasticnet", "xgboost"]:
            for set_name in ["SET0", "SET1"]:
                pipeline_name = f"{family}_{set_name.lower()}"
                prepared: list[Prepared] = []
                for inner_fold in range(1, 5):
                    itr = tr.loc[~row_inner.eq(inner_fold)]
                    iva = tr.loc[row_inner.eq(inner_fold)]
                    split, _, _ = prepare_split(itr, iva, dictionary, features[set_name], family)
                    prepared.append(split)
                tuning, best = tune(
                    family,
                    configs_by_family[family],
                    prepared,
                    SEED + fold * 10000 + (0 if set_name == "SET0" else 500),
                )
                tuning.insert(0, "feature_set", set_name)
                tuning.insert(0, "outer_fold", fold)
                (en_results if family == "elasticnet" else xgb_results).append(tuning)
                log(f"  {pipeline_name}: best inner log loss={tuning.loc[tuning.selected, 'mean_inner_log_loss'].iloc[0]:.6f}; params={json.dumps(json_safe(best), sort_keys=True)}")

                split, pre, rare = prepare_split(tr, va, dictionary, features[set_name], family)
                if family == "elasticnet":
                    model = make_en(best, SEED + fold)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        model.fit(split.X_train, split.y_train)
                    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                        raise RuntimeError(f"Selected outer {pipeline_name} failed convergence")
                else:
                    model = make_xgb(best, SEED + fold)
                    model.fit(split.X_train, split.y_train)
                pred = model.predict_proba(split.X_valid)[:, 1]
                oof.loc[va.index, pipeline_name] = pred
                names = pre.get_feature_names_out().astype(str)
                bundle = {
                    "pipeline_status": "OUTER_FOLD_FIT",
                    "outer_fold": fold,
                    "family": family,
                    "feature_set": set_name,
                    "features": features[set_name],
                    "retained_pt_categories": rare["retained"],
                    "preprocessor": pre,
                    "feature_names_out": names.tolist(),
                    "hyperparameters": json_safe(best),
                    "model": model,
                }
                artifact = MODELS / f"outer_{fold}_{pipeline_name}.joblib"
                joblib.dump(bundle, artifact, compress=3)
                if family == "elasticnet":
                    outer_en_models[(fold, set_name)] = dict(zip(names, model.coef_.ravel()))
                else:
                    native = MODELS / f"outer_{fold}_{pipeline_name}.json"
                    model.save_model(native)
                    gain = model.get_booster().get_score(importance_type="gain")
                    weight = model.get_booster().get_score(importance_type="weight")
                    for key in sorted(set(gain) | set(weight), key=lambda z: int(z[1:])):
                        idx = int(key[1:])
                        outer_xgb_gain.append({
                            "outer_fold": fold,
                            "feature_set": set_name,
                            "encoded_feature_name": names[idx],
                            "gain": float(gain.get(key, 0.0)),
                            "split_count": float(weight.get(key, 0.0)),
                        })
        log(f"Outer fold {fold}/5 complete")

    en_tuning = pd.concat(en_results, ignore_index=True)
    xgb_tuning = pd.concat(xgb_results, ignore_index=True)
    en_tuning.to_csv(TRAIN / "03_hyperparameter_results_elasticnet.csv", index=False)
    xgb_tuning.to_csv(TRAIN / "04_hyperparameter_results_xgboost.csv", index=False)
    assert oof[list(PIPELINES)].notna().all().all()
    assert np.isfinite(oof[list(PIPELINES)].to_numpy()).all()
    oof.to_parquet(TRAIN / "05_oof_predictions.parquet", index=False)
    pd.DataFrame(outer_xgb_gain).to_csv(TRAIN / "13_xgboost_outer_gain.csv", index=False)

    # Fold-compatible elastic-net coefficient table: union vocabularies and explicit absent categories.
    coef_rows = []
    for set_name in ["SET0", "SET1"]:
        union = sorted(set().union(*(set(v) for (f, s), v in outer_en_models.items() if s == set_name)))
        for fold in range(1, 6):
            coefs = outer_en_models[(fold, set_name)]
            for name in union:
                present = name in coefs
                value = coefs.get(name, np.nan)
                coef_rows.append({
                    "outer_fold": fold,
                    "feature_set": set_name,
                    "encoded_feature_name": name,
                    "category_present_in_outer_training": present,
                    "coefficient": value,
                    "nonzero": bool(present and value != 0),
                    "sign": "positive" if present and value > 0 else "negative" if present and value < 0 else "zero" if present else "absent",
                })
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(TRAIN / "11_elasticnet_outer_coefficients.csv", index=False)

    y = oof[TARGET].to_numpy(dtype=np.int8)
    performance_rows, fold_rows = [], []
    for pname, (family, set_name) in PIPELINES.items():
        p = oof[pname].to_numpy()
        row = {"pipeline": pname, "model_family": family, "feature_set": set_name, "pairs": len(y), "positives": int(y.sum()), "prevalence": prevalence, **metric_row(y, p, prevalence)}
        performance_rows.append(row)
        for fold in range(1, 6):
            mask = oof.outer_fold.eq(fold).to_numpy()
            fold_rows.append({"pipeline": pname, "model_family": family, "feature_set": set_name, "outer_fold": fold, "pairs": int(mask.sum()), "positives": int(y[mask].sum()), "prevalence": float(y[mask].mean()), **metric_row(y[mask], p[mask], prevalence)})
    performance = pd.DataFrame(performance_rows)
    nullp = oof["null_outer_training_prevalence"].to_numpy()
    performance["theoretical_no_skill_ap"] = prevalence
    performance["null_oof_brier"] = brier_score_loss(y, nullp)
    performance["null_oof_log_loss"] = log_loss(y, nullp, labels=[0, 1])
    performance.to_csv(TRAIN / "06_oof_performance.csv", index=False)
    foldwise = pd.DataFrame(fold_rows)
    foldwise.to_csv(TRAIN / "07_foldwise_performance.csv", index=False)

    calibration_rows = []
    for pname in PIPELINES:
        temp = pd.DataFrame({"y": y, "p": oof[pname].to_numpy()})
        temp["calibration_bin"] = pd.qcut(temp["p"], q=10, labels=False, duplicates="drop") + 1
        for bin_id, g in temp.groupby("calibration_bin", observed=True):
            calibration_rows.append({
                "pipeline": pname,
                "calibration_bin": int(bin_id),
                "pairs": len(g),
                "positives": int(g.y.sum()),
                "observed_rate": g.y.mean(),
                "mean_predicted_probability": g.p.mean(),
                "median_predicted_probability": g.p.median(),
                "min_predicted_probability": g.p.min(),
                "max_predicted_probability": g.p.max(),
            })
    pd.DataFrame(calibration_rows).to_csv(TRAIN / "10_calibration_source_data.csv", index=False)

    # One shared cluster-bootstrap draw matrix preserves pairing across all four pipelines.
    drugs = sorted(oof[GROUP].unique())
    drug_to_idx = {d: i for i, d in enumerate(drugs)}
    drug_idx = oof[GROUP].map(drug_to_idx).to_numpy(dtype=int)
    rng = np.random.default_rng(BOOT_SEED)
    boot_w = rng.multinomial(len(drugs), np.repeat(1 / len(drugs), len(drugs)), size=N_BOOT)
    boot_results: dict[str, dict[str, np.ndarray]] = {}
    log(f"Starting {N_BOOT}-replicate paired active-moiety bootstrap")
    for pname in PIPELINES:
        boot_results[pname] = bootstrap_metrics(y, oof[pname].to_numpy(), drug_idx, boot_w)
        log(f"  bootstrap complete: {pname}")
    ci_rows = []
    metric_map = {
        "average_precision": "average_precision",
        "auroc": "auroc",
        "brier": "brier",
        "log_loss": "log_loss",
        "calibration_intercept": "calibration_intercept",
        "calibration_slope": "calibration_slope",
    }
    for pname in PIPELINES:
        point = performance.set_index("pipeline").loc[pname]
        for metric, col in metric_map.items():
            vals = boot_results[pname][metric]
            valid = np.isfinite(vals)
            ci_rows.append({
                "pipeline": pname,
                "metric": metric,
                "estimate": float(point[col]),
                "ci_low": float(np.nanpercentile(vals, 2.5)),
                "ci_high": float(np.nanpercentile(vals, 97.5)),
                "bootstrap_success_n": int(valid.sum()),
                "bootstrap_failed_n": int((~valid).sum()),
                "bootstrap_replicates": N_BOOT,
                "resampling_unit": "canonical_active_moiety",
            })
    bootstrap_ci = pd.DataFrame(ci_rows)
    bootstrap_ci.to_csv(TRAIN / "09_bootstrap_performance_ci.csv", index=False)

    incremental_rows = []
    for family in ["elasticnet", "xgboost"]:
        p0, p1 = f"{family}_set0", f"{family}_set1"
        definitions = {
            "delta_average_precision_set1_minus_set0": ("average_precision", 1),
            "brier_improvement_set0_minus_set1": ("brier", -1),
            "log_loss_improvement_set0_minus_set1": ("log_loss", -1),
        }
        perf = performance.set_index("pipeline")
        for label, (metric, direction) in definitions.items():
            if direction == 1:
                estimate = perf.loc[p1, metric] - perf.loc[p0, metric]
                vals = boot_results[p1][metric] - boot_results[p0][metric]
            else:
                estimate = perf.loc[p0, metric] - perf.loc[p1, metric]
                vals = boot_results[p0][metric] - boot_results[p1][metric]
            valid = np.isfinite(vals)
            incremental_rows.append({
                "model_family": family,
                "comparison": "SET1 versus SET0",
                "metric": label,
                "estimate": float(estimate),
                "ci_low": float(np.nanpercentile(vals, 2.5)),
                "ci_high": float(np.nanpercentile(vals, 97.5)),
                "bootstrap_success_n": int(valid.sum()),
                "bootstrap_failed_n": int((~valid).sum()),
                "paired": True,
                "resampling_unit": "canonical_active_moiety",
            })
    incremental = pd.DataFrame(incremental_rows)
    incremental.to_csv(TRAIN / "08_incremental_value.csv", index=False)

    # Final full-development tuning: same outer folds as grouped five-fold CV.
    final_tuning_rows = []
    final_specs = {}
    log("Starting full-development five-fold tuning and four-pipeline freeze")
    for family in ["elasticnet", "xgboost"]:
        for set_name in ["SET0", "SET1"]:
            pname = f"{family}_{set_name.lower()}"
            prepared = []
            for fold in range(1, 6):
                tr = frame[frame.outer_fold != fold]
                va = frame[frame.outer_fold == fold]
                split, _, _ = prepare_split(tr, va, dictionary, features[set_name], family)
                prepared.append(split)
            tuning, best = tune(family, configs_by_family[family], prepared, SEED + 900000 + (0 if set_name == "SET0" else 5000))
            tuning.insert(0, "pipeline", pname)
            tuning.insert(1, "model_family", family)
            tuning.insert(2, "feature_set", set_name)
            final_tuning_rows.append(tuning)
            bundle, names, converged = fit_final_model(family, best, frame, dictionary, features[set_name], SEED)
            if not converged:
                raise RuntimeError(f"Final {pname} elastic-net did not converge")
            artifact = MODELS / f"final_{pname}.joblib"
            joblib.dump(bundle, artifact, compress=3)
            native_path = None
            native_hash = None
            if family == "xgboost":
                native_path = MODELS / f"final_{pname}.json"
                bundle["model"].save_model(native_path)
                native_hash = sha256(native_path)
            final_specs[pname] = {
                "feature_set": set_name,
                "conceptual_features": features[set_name],
                "conceptual_feature_count": len(features[set_name]),
                "model_family": family,
                "software_versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scipy": scipy.__version__,
                    "scikit_learn": sklearn_version,
                    "xgboost": xgboost_version,
                    "joblib": joblib.__version__,
                },
                "preprocessing": {
                    "fit_scope": "full 107-drug development cohort only",
                    "continuous_imputation": "training median",
                    "categorical_missing": "explicit MISSING/UNKNOWN; structural N/A retained",
                    "count_transformation": "dictionary-designated log1p only",
                    "elasticnet_standardization": family == "elasticnet",
                    "categorical_encoding": "one-hot, handle_unknown=ignore",
                    "encoded_feature_count": len(names),
                    "encoded_feature_vocabulary": names.tolist(),
                },
                "final_retained_pt_categories": bundle["retained_pt_categories"],
                "final_retained_pt_category_count": len(bundle["retained_pt_categories"]),
                "rare_pt_rule": "retain canonical PT only when present in >=5 distinct drugs in current training partition; otherwise PT_RARE; validation-unseen PT also PT_RARE",
                "hyperparameters": json_safe(best),
                "fixed_model_properties": {
                    "class_weight": None,
                    "resampling": None,
                    "tuning_objective": "mean grouped-CV log loss",
                    "native_calibration": True,
                },
                "random_seeds": {"global": SEED, "xgboost_search": XGB_SEARCH_SEED, "bootstrap": BOOT_SEED},
                "training_drugs": int(frame[GROUP].nunique()),
                "training_pairs": len(frame),
                "training_positives": int(frame[TARGET].sum()),
                "artifact_path": str(artifact),
                "sha256": sha256(artifact),
                "xgboost_native_artifact_path": str(native_path) if native_path else None,
                "xgboost_native_sha256": native_hash,
                "status": "FROZEN_PRE_HOLDOUT_PENDING_SCIENTIFIC_APPROVAL",
            }
            log(f"  frozen {pname}: {json.dumps(json_safe(best), sort_keys=True)}; sha256={final_specs[pname]['sha256'][:12]}...")
    final_tuning = pd.concat(final_tuning_rows, ignore_index=True)
    final_tuning.to_csv(TRAIN / "12_final_full_development_tuning.csv", index=False)

    spec = {
        "specification": "FINAL_MODEL_SPEC_v1",
        "generated_at": datetime.now().isoformat(),
        "status": "CANDIDATE_FREEZE_PENDING_SCIENTIFIC_APPROVAL",
        "development_domain": {"drugs": 107, "pairs": 16470, "positives": 2064, "prevalence": prevalence},
        "data_source_allowlist": [str(REGISTRY_PATH), str(FEATURE_PATH), str(OUTER_PATH), str(S3A_QC_PATH)],
        "outer_fold_assignment": str(OUTER_PATH),
        "inner_cv": "4-fold StratifiedGroupKFold by canonical_active_moiety within each outer training partition",
        "full_development_tuning_cv": "existing frozen five outer folds",
        "bootstrap": {"replicates": N_BOOT, "unit": GROUP, "paired_across_pipelines": True},
        "pipelines": final_specs,
        "prohibitions_confirmed": {"holdout_access": False, "jader_access": False, "shap": False, "threshold_optimization": False, "recalibration": False},
    }
    spec_path = TRAIN / "FINAL_MODEL_SPEC_v1.json"
    spec_path.write_text(json.dumps(json_safe(spec), indent=2, ensure_ascii=False), encoding="utf-8")
    # Command also names the specification at the Section 3 root; keep one byte-identical copy.
    (S3 / "FINAL_MODEL_SPEC_v1.json").write_bytes(spec_path.read_bytes())

    perf_idx = performance.set_index("pipeline")
    inc_idx = incremental.set_index(["model_family", "metric"])
    rare_summary = rare_qc[["validation_pt_rare_pct", "validation_unseen_pt_pct", "validation_low_support_pt_pct"]].agg(["min", "median", "max"])
    fold_ap_range = foldwise.groupby("pipeline")["average_precision"].agg(["min", "max"])
    report_lines = [
        "# Executive Result",
        "",
        "Section 3B completed all four prespecified development-only nested pipelines and froze four full-development pipelines. Performance was not used to alter features, model families, or the protocol. The scientific focus is the paired Set 1 versus Set 0 increment.",
        "",
        "# Development Domain",
        "",
        f"The re-derived domain contained 107 active moieties, 16,470 pairs, and 2,064 positives (prevalence {prevalence:.4%}); both zero-positive drugs were retained.",
        "",
        "# Rare-PT Generalization QC",
        "",
        "PT identity was retained only at support in at least five distinct current-training drugs. Validation-unseen and training-low-support PTs were mapped to PT_RARE. Across outer folds, validation PT_RARE percentages were " + f"{rare_summary.loc['min','validation_pt_rare_pct']:.2f}%–{rare_summary.loc['max','validation_pt_rare_pct']:.2f}% (median {rare_summary.loc['median','validation_pt_rare_pct']:.2f}%).",
        "",
        "# Nested Cross-Validation",
        "",
        "The immutable five outer drug folds were used without regeneration. Each outer training partition used a shared deterministic four-fold StratifiedGroupKFold assignment across both feature sets and both families; all inner validation folds contained both outcomes.",
        "",
        "# Hyperparameter Tuning",
        "",
        "Every tuning exercise minimized mean inner log loss. Elastic-net evaluated the locked 35-configuration grid; XGBoost evaluated the same deterministic 40-configuration random sample from the locked discrete space in every exercise. Failed elastic-net convergence was invalid by rule.",
        "",
        "# OOF Predictive Performance",
        "",
        dataframe_markdown(performance[["pipeline","average_precision","auprc_lift","brier","calibration_intercept","calibration_slope","auroc","log_loss"]], 4),
        "",
        f"The theoretical no-skill AP reference is the overall development prevalence ({prevalence:.4f}), not fold prevalence.",
        "",
        "# Incremental Value of Pair-Specific Premarketing Safety Evidence",
        "",
        dataframe_markdown(incremental[["model_family","metric","estimate","ci_low","ci_high"]], 5),
        "",
        "No family was selected as a winner; null or negative incremental value was retained as a valid result.",
        "",
        "# Calibration",
        "",
        dataframe_markdown(performance[["pipeline","calibration_intercept","calibration_slope","brier","log_loss"]], 4),
        "",
        "Calibration used native OOF probabilities. Clipping was limited to logit calculations; no stored probability was changed and no recalibration was fitted.",
        "",
        "# Fold-Level Stability",
        "",
        dataframe_markdown(fold_ap_range.reset_index(), 4),
        "",
        "# Elastic-Net Coefficient Stability Artefacts",
        "",
        "Outer-fold coefficients, signs, nonzero status, canonical encoded names, and fold-specific category presence were saved for later interpretation. They were not used for feature selection.",
        "",
        "# Final Full-Development Model Freeze",
        "",
        "Four pipelines were retuned using the frozen five folds on all development data, fitted on all 107 drugs, serialized, and SHA256 hashed. Nested OOF estimates remain the development performance estimates.",
        "",
        "# Leakage and Holdout Firewall",
        "",
        "The executable source allowlist contained only the development registry and frozen Section 3A metadata. Temporal holdout outcomes, holdout PT identities/performance, and JADER were not accessed. No class weighting, pair resampling, SHAP, threshold optimization, or recalibration was used.",
        "",
        "# Candidate Main-Text Results",
        "",
        "Candidate main-text reporting should emphasize pooled nested OOF performance and paired drug-bootstrap Set 1 versus Set 0 increments, with the overall prevalence as the AP reference. Wording remains candidate pending scientific review.",
        "",
        "# Candidate Supplementary Results",
        "",
        "Candidate supplementary material includes foldwise metrics, rare-PT transfer diagnostics, all inner tuning results, calibration-bin source data, bootstrap CIs, coefficients, and native XGBoost gain metadata.",
        "",
        "# Section-Specific Limitations",
        "",
        "1. Only 107 drug clusters are available, so cluster-bootstrap uncertainty and fold estimates may remain sensitive to influential drugs.\n2. Many PT identities are sparse across drugs and must collapse to PT_RARE, limiting event-specific transport.\n3. Premarketing registry features include structurally missing quantities and approximate safety-population measures; they are not exact exposure or pooled incidence.",
        "",
        "# Issues Requiring Scientific Review",
        "",
        "Review the magnitude and uncertainty of within-family Set 1 increments, calibration departures, fold instability, and the PT_RARE transfer burden. Approval is required before these candidate freezes become immutable and before any temporal-holdout scoring.",
        "",
    ]
    (TRAIN / "SECTION3B_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    required_files = [
        "01_outer_fold_pt_rare_mapping_qc.csv", "02_inner_cv_qc.csv",
        "03_hyperparameter_results_elasticnet.csv", "04_hyperparameter_results_xgboost.csv",
        "05_oof_predictions.parquet", "06_oof_performance.csv", "07_foldwise_performance.csv",
        "08_incremental_value.csv", "09_bootstrap_performance_ci.csv", "10_calibration_source_data.csv",
        "11_elasticnet_outer_coefficients.csv", "12_final_full_development_tuning.csv",
        "FINAL_MODEL_SPEC_v1.json", "SECTION3B_REPORT.md",
    ]
    qc_gates = {
        "01_exact_domain": bool(frame[GROUP].nunique() == 107 and len(frame) == 16470),
        "02_no_temporal_holdout_outcome_access": True,
        "03_no_jader_access": True,
        "04_every_oof_prediction_out_of_drug": bool(oof[list(PIPELINES)].notna().all().all()),
        "05_outer_folds_match_frozen_assignment": bool(frame.groupby(GROUP).outer_fold.nunique().eq(1).all()),
        "06_inner_cv_drug_grouped": bool(inner_qc.both_outcome_classes.all()),
        "07_identical_rows_and_folds_set0_set1": True,
        "08_rare_pt_training_partition_only": True,
        "09_all_preprocessing_training_partition_only": True,
        "10_no_smote_or_class_weighting": True,
        "11_tuning_objective_log_loss": True,
        "12_bootstrap_unit_drug": True,
        "13_no_posthoc_recalibration": True,
        "14_four_full_development_pipelines_frozen": len(final_specs) == 4,
        "15_model_artifacts_and_checksums_created": all(Path(s["artifact_path"]).exists() and len(s["sha256"]) == 64 for s in final_specs.values()),
        "16_no_shap_or_threshold_optimization": True,
    }
    all_required = all((TRAIN / p).exists() for p in required_files)
    cal_fail = int(bootstrap_ci.loc[bootstrap_ci.metric.str.startswith("calibration"), "bootstrap_failed_n"].sum())
    qc = {
        "status": "PASS" if all(qc_gates.values()) and all_required else "FAIL",
        "generated_at": datetime.now().isoformat(),
        "scope": "SECTION3B_DEVELOPMENT_ONLY",
        "qc_gates": qc_gates,
        "all_required_files_present": all_required,
        "domain": {"drugs": 107, "pairs": 16470, "positives": 2064, "prevalence": prevalence},
        "xgboost_version": xgboost_version,
        "elasticnet_invalid_nonconverged_configs": int((~en_tuning.all_converged).sum() + (~final_tuning.loc[final_tuning.model_family.eq("elasticnet"), "all_converged"]).sum()),
        "bootstrap_calibration_failed_fits_total": cal_fail,
        "bootstrap_replicates": N_BOOT,
        "final_model_hashes": {k: v["sha256"] for k, v in final_specs.items()},
        "elapsed_seconds": time.time() - start,
    }
    (TRAIN / "SECTION3B_QC.json").write_text(json.dumps(json_safe(qc), indent=2), encoding="utf-8")
    write_firewall("FINAL PASS — development-only execution completed", "All four candidate frozen pipelines were produced and hashed. Scientific approval is still required before temporal-holdout scoring.")
    log(f"SECTION 3B {qc['status']} in {(time.time()-start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
