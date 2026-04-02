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


def write_summary_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
