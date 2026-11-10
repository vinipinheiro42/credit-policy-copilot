"""Fit the abstention rule on the TUNING split only.

Rule families: every subset of 1 or 2 signals (dense, rerank, bm25_norm, coverage) x combine {any, all}.
Within a family, thresholds are searched on a grid of midpoints between observed values (vectorized).
Objective inside a family: max abstention F1 s.t. false-refusal rate <= ``max_frr``; ties -> lower
false-refusal, then (across families) fewer signals.

``cross_validate`` picks the FAMILY by repeated stratified k-fold CV on tune (thresholds re-fitted on
each training fold, scored on the held-in validation fold, predictions pooled across folds), then the
chosen family is re-fitted on all of tune. The held-out split is never touched here.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from copilot.abstention import SIGNALS, AbstentionRule
from copilot.evals.metrics import abstention_metrics

Family = tuple[tuple[str, ...], str]


@dataclass
class CalibrationResult:
    rule: AbstentionRule
    metrics: dict[str, float | int]
    baseline_dense_only: dict[str, object]
    n_rules_searched: int


@dataclass
class CVResult:
    family: Family
    rule: AbstentionRule  # chosen family re-fitted on all tune items
    table: list[dict] = field(default_factory=list)  # one row per family
    folds: int = 5
    repeats: int = 5


def _grid(values: np.ndarray, max_grid: int) -> np.ndarray:
    u = np.unique(values)
    cands = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]])
    if len(cands) > max_grid:
        idx = np.unique(np.linspace(0, len(cands) - 1, max_grid).round().astype(int))
        cands = cands[idx]
    return cands


def _confusion(pred: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pred: (..., N) bool; y: (N,) bool -> tp, fp, fn arrays of shape (...)."""
    tp = (pred & y).sum(-1)
    fp = (pred & ~y).sum(-1)
    fn = (~pred & y).sum(-1)
    return tp, fp, fn


def _scores(tp, fp, fn, n_neg):
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    frr = fp / n_neg if n_neg else np.zeros_like(f1)
    return f1, frr


def fit_family(
    X: dict[str, np.ndarray], y: np.ndarray, family: Family, max_frr: float, max_grid: int = 60
) -> tuple[AbstentionRule, float, float] | None:
    subset, combine = family
    n_neg = int((~y).sum())
    grids = [_grid(X[s], max_grid) for s in subset]
    if len(subset) == 1:
        pred = X[subset[0]][None, :] < grids[0][:, None]  # (G, N)
    else:
        low1 = X[subset[0]][None, :] < grids[0][:, None]
        low2 = X[subset[1]][None, :] < grids[1][:, None]
        pred = (
            (low1[:, None, :] | low2[None, :, :])
            if combine == "any"
            else (low1[:, None, :] & low2[None, :, :])
        )
    tp, fp, fn = _confusion(pred, y)
    f1, frr = _scores(tp, fp, fn, n_neg)
    ok = frr <= max_frr + 1e-12
    if not ok.any():
        return None
    # lexicographic: max f1, then min frr, then the most lenient (lowest) thresholds for stability
    key = np.where(ok, f1 * 1e6 - frr * 1e3, -np.inf)
    best = np.unravel_index(int(np.argmax(key)), key.shape)
    ths = {s: float(grids[i][best[i]]) for i, s in enumerate(subset)}
    return AbstentionRule(thresholds=ths, combine=combine), float(f1[best]), float(frr[best])


def families(available: Sequence[str]) -> list[Family]:
    fams: list[Family] = [((s,), "any") for s in available]
    for pair in itertools.combinations(available, 2):
        fams += [(pair, "any"), (pair, "all")]
    return fams


def _matrix(signals: Sequence[dict], available: Sequence[str]) -> dict[str, np.ndarray]:
    return {s: np.array([float(sig[s]) for sig in signals]) for s in available}


def _available(signals: Sequence[dict], wanted: Sequence[str]) -> list[str]:
    return [s for s in wanted if all(sig.get(s) is not None for sig in signals)]


def calibrate(
    signals: Sequence[dict[str, float | None]],
    should_abstain: Sequence[bool],
    max_frr: float = 0.10,
    max_grid: int = 80,
    available: Sequence[str] = SIGNALS,
) -> CalibrationResult:
    """Best rule over all families on the given items (no CV)."""
    avail = _available(signals, available)
    if not avail:
        raise ValueError("no signal available for every item")
    X, y = _matrix(signals, avail), np.array(should_abstain, dtype=bool)
    best = None
    best_dense = None
    for fam in families(avail):
        fit = fit_family(X, y, fam, max_frr, max_grid)
        if fit is None:
            continue
        rule, f1, frr = fit
        key = (round(f1, 9), -round(frr, 9), -len(fam[0]), fam[0] == ("dense",))
        if best is None or key > best[0]:
            best = (key, rule)
        if fam[0] == ("dense",):
            best_dense = rule
    if best is None:
        raise ValueError(f"no rule satisfies false_refusal_rate <= {max_frr}")
    rule = best[1]
    m = abstention_metrics(list(y), [rule.decide(s) for s in signals])
    rule.tune_metrics = {k: v for k, v in m.items() if isinstance(v, float)}
    baseline = {"rule": best_dense.describe() if best_dense else None}
    return CalibrationResult(rule, m, baseline, len(families(avail)) * max_grid**2)


def stratified_folds(y: Sequence[bool], k: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    pos = [i for i, v in enumerate(y) if v]
    neg = [i for i, v in enumerate(y) if not v]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds: list[list[int]] = [[] for _ in range(k)]
    for j, i in enumerate(pos + neg):
        folds[j % k].append(i)
    return folds


def cross_validate(
    signals: Sequence[dict[str, float | None]],
    should_abstain: Sequence[bool],
    max_frr: float = 0.10,
    k: int = 5,
    repeats: int = 5,
    available: Sequence[str] = SIGNALS,
    max_grid: int = 60,
) -> CVResult:
    avail = _available(signals, available)
    X, y = _matrix(signals, avail), np.array(should_abstain, dtype=bool)
    table = []
    for fam in families(avail):
        f1s, frrs, feasible = [], [], True
        for rep in range(repeats):
            pred = np.zeros(len(y), dtype=bool)
            for fold in stratified_folds(list(y), k, seed=rep):
                val = np.array(fold)
                tr = np.setdiff1d(np.arange(len(y)), val)
                fit = fit_family({s: v[tr] for s, v in X.items()}, y[tr], fam, max_frr, max_grid)
                if fit is None:
                    feasible = False
                    break
                rule = fit[0]
                pred[val] = [rule.decide({s: float(X[s][i]) for s in avail}) for i in val]
            if not feasible:
                break
            m = abstention_metrics(list(y), list(pred))
            f1s.append(m["f1"])
            frrs.append(m["false_refusal_rate"])
        if not feasible:
            continue
        table.append(
            {
                "family": " ".join(fam[0]) + ("" if len(fam[0]) == 1 else f" [{fam[1]}]"),
                "signals": list(fam[0]),
                "combine": fam[1],
                "cv_f1_mean": float(np.mean(f1s)),
                "cv_f1_std": float(np.std(f1s)),
                "cv_frr_mean": float(np.mean(frrs)),
            }
        )
    if not table:
        raise ValueError("no feasible family")
    # choose: CV F1 (rounded to 0.01 so noise-level gaps don't decide), then fewer signals, then lower FRR
    table.sort(key=lambda r: (-round(r["cv_f1_mean"], 2), len(r["signals"]), r["cv_frr_mean"]))
    top = table[0]
    fam: Family = (tuple(top["signals"]), top["combine"])
    fit = fit_family(X, y, fam, max_frr, max_grid)
    assert fit is not None
    rule = fit[0]
    m = abstention_metrics(list(y), [rule.decide(s) for s in signals])
    rule.tune_metrics = {kk: v for kk, v in m.items() if isinstance(v, float)} | {
        "cv_f1_mean": top["cv_f1_mean"],
        "cv_f1_std": top["cv_f1_std"],
        "cv_frr_mean": top["cv_frr_mean"],
    }
    return CVResult(fam, rule, table, k, repeats)
