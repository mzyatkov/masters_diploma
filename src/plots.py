"""Plotting utilities for evaluation and Pareto front."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

from src.utils import ensure_dir

sns.set_theme(style="whitegrid")


def plot_roc_curves(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    path: Path | str,
) -> None:
    plt.figure(figsize=(6, 5))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves")
    plt.legend()
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pr_curves(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    path: Path | str,
) -> None:
    plt.figure(figsize=(6, 5))
    for name, p in preds.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ap = average_precision_score(y_true, p)
        plt.plot(rec, prec, label=f"{name} (PR-AUC={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–recall curves")
    plt.legend()
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_calibration(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    path: Path | str,
    n_bins: int = 10,
) -> None:
    from sklearn.calibration import calibration_curve

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name, p in preds.items():
        prob_true, prob_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="uniform")
        plt.plot(prob_pred, prob_true, "s-", label=name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curves")
    plt.legend()
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pareto_front(
    objectives: np.ndarray,
    path: Path | str,
    *,
    labels: tuple[str, str] = ("Expected profit", "CVaR (profit)"),
    highlight_indices: dict[str, int] | None = None,
) -> None:
    """
    Plot 2D Pareto front (maximize both objectives; points should be non-dominated).

    ``objectives`` shape (n, 2): columns are [E[profit], CVaR].
    """
    plt.figure(figsize=(7, 5))
    x = objectives[:, 0]
    y = objectives[:, 1]
    plt.scatter(x, y, c="#2c7fb8", edgecolor="k", alpha=0.85, s=45)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title("Pareto front (profit vs tail risk)")
    if highlight_indices:
        for label, idx in highlight_indices.items():
            plt.scatter(
                objectives[idx, 0],
                objectives[idx, 1],
                s=120,
                marker="*",
                label=label,
            )
        plt.legend()
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pareto_population_full(
    population_F_min: np.ndarray,
    population_rank: np.ndarray,
    path: Path | str,
    *,
    labels: tuple[str, str] = ("Expected profit", "CVaR (profit)"),
    highlight_indices_pop: dict[str, int] | None = None,
) -> None:
    """
    Separate figure: final NSGA-II population in objective space (maximize profit / CVaR).

    ``population_F_min`` columns are pymoo minimization objectives
    ``[-E[profit], -CVaR]``; they are converted to ``E[profit]``, ``CVaR`` for the axes.
    Points with ``rank == 0`` are the non-dominated front; others are dominated.
    """
    if population_F_min.size == 0:
        return
    F = np.asarray(population_F_min, dtype=float)
    rnk = np.asarray(population_rank, dtype=int).ravel()
    if F.shape[0] != len(rnk):
        raise ValueError("population_F_min and population_rank length mismatch.")
    # Maximize-space (same as pareto_front.csv)
    obj = np.column_stack([-F[:, 0], -F[:, 1]])
    nd = rnk == 0
    dom = ~nd

    plt.figure(figsize=(7, 5))
    if np.any(dom):
        plt.scatter(
            obj[dom, 0],
            obj[dom, 1],
            c="#bdbdbd",
            edgecolors="#737373",
            alpha=0.85,
            s=38,
            label=f"Dominated (rank>0), n={int(np.sum(dom))}",
            zorder=1,
        )
    if np.any(nd):
        plt.scatter(
            obj[nd, 0],
            obj[nd, 1],
            c="#fd8d3c",
            edgecolors="#a63603",
            alpha=0.9,
            s=52,
            label=f"Non-dominated (rank 0), n={int(np.sum(nd))}",
            zorder=2,
        )
    if highlight_indices_pop:
        for label, j in highlight_indices_pop.items():
            j = int(j)
            if 0 <= j < len(obj):
                plt.scatter(
                    obj[j, 0],
                    obj[j, 1],
                    s=160,
                    marker="*",
                    c="#6baed6",
                    edgecolors="k",
                    linewidths=0.6,
                    zorder=3,
                    label=label,
                )
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title("Final population: all candidates (objective space)")
    plt.legend(loc="best", fontsize=8)
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pareto_front_with_errorbars(
    pareto_with_uncertainty,
    path: Path | str,
    *,
    labels: tuple[str, str] = ("Expected profit", "CVaR (profit)"),
    highlight_indices: dict[str, int] | None = None,
) -> None:
    """
    Pareto front with uncertainty bars for both objectives.

    Expected columns:
    - expected_profit, cvar_profit
    - expected_profit_ci_low/high, cvar_profit_ci_low/high
    """
    import pandas as pd

    df = pd.DataFrame(pareto_with_uncertainty).copy()
    if df.empty:
        return

    x = df["expected_profit"].to_numpy(dtype=float)
    y = df["cvar_profit"].to_numpy(dtype=float)
    xerr = np.vstack(
        [
            x - df["expected_profit_ci_low"].to_numpy(dtype=float),
            df["expected_profit_ci_high"].to_numpy(dtype=float) - x,
        ]
    )
    yerr = np.vstack(
        [
            y - df["cvar_profit_ci_low"].to_numpy(dtype=float),
            df["cvar_profit_ci_high"].to_numpy(dtype=float) - y,
        ]
    )

    plt.figure(figsize=(7.5, 5.5))
    plt.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        color="#2c7fb8",
        ecolor="#9ecae1",
        elinewidth=1,
        capsize=2.5,
        alpha=0.85,
        markersize=5,
    )
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title("Pareto front with uncertainty intervals")
    if highlight_indices:
        for label, idx in highlight_indices.items():
            j = int(idx)
            if 0 <= j < len(df):
                plt.scatter(x[j], y[j], s=130, marker="*", label=label, edgecolors="k")
        plt.legend(fontsize=8)
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_metric_ci_intervals(
    metrics_ci_df,
    path: Path | str,
    *,
    metrics: tuple[str, ...] = ("roc_auc", "pr_auc", "brier", "ece"),
) -> None:
    """Plot point estimates with CI intervals for each model."""
    import pandas as pd

    df = pd.DataFrame(metrics_ci_df).copy()
    if df.empty:
        return

    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8.5, 2.2 * n_rows), sharex=False)
    if n_rows == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        if f"{metric}_value" not in df.columns:
            continue
        y = np.arange(len(df))
        val = df[f"{metric}_value"].to_numpy(dtype=float)
        lo = df[f"{metric}_ci_low"].to_numpy(dtype=float)
        hi = df[f"{metric}_ci_high"].to_numpy(dtype=float)
        xerr = np.vstack([val - lo, hi - val])
        ax.errorbar(val, y, xerr=xerr, fmt="o", capsize=3, color="#2c7fb8", ecolor="#9ecae1")
        ax.set_yticks(y)
        ax.set_yticklabels(df["model"].astype(str).tolist())
        ax.set_xlabel(metric)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Model metrics with bootstrap confidence intervals")
    ensure_dir(Path(path).parent)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_uncertainty_decomposition(summary: dict[str, float], path: Path | str) -> None:
    """Plot mean aleatoric vs epistemic uncertainty share."""
    ale = float(summary.get("aleatoric_mean", 0.0))
    epi = float(summary.get("epistemic_mean", 0.0))
    total = max(ale + epi, 1e-12)
    shares = np.array([ale / total, epi / total], dtype=float)
    labels = ["Aleatoric", "Epistemic"]

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(labels, shares, color=["#74a9cf", "#fd8d3c"])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Share of total predictive uncertainty")
    plt.title("Uncertainty decomposition (CatBoost ensemble)")
    for i, s in enumerate(shares):
        plt.text(i, s + 0.02, f"{s:.2%}", ha="center", va="bottom", fontsize=9)
    ensure_dir(Path(path).parent)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_summary_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
