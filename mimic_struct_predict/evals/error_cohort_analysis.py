import argparse
import collections
import logging
import math
import multiprocessing
import os
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from mimic_struct_predict.evals.error_rates import load_predictions
from mimic_struct_predict.evals.plot_style import (
    format_task_title,
    sort_tasks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

COHORTS = ["0%", "5-50%", "55-95%", "100%"]


def extract_history_length(text: str) -> int:
    """Extracts num_events from decoded_input."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return sum(1 for line in lines if not line.lstrip("/").startswith("#"))


def assign_cohort(format_adherence: list[int]) -> str:
    """Assigns error cohort based on format adherence across trajectories."""
    n_traj = len(format_adherence)
    n_errors = n_traj - sum(format_adherence)
    if n_errors == 0:
        return "0%"
    if n_errors == n_traj:
        return "100%"
    return "5-50%" if (n_errors / n_traj <= 0.5) else "55-95%"


def calc_ci(boot_scores: list[float], empirical_score: float | None = None) -> dict[str, float]:
    """Calculates point estimate and 95% bootstrap confidence interval."""
    arr = np.array(boot_scores, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {"mean": np.nan, "lower_ci": np.nan, "upper_ci": np.nan}

    mean = empirical_score if (empirical_score is not None and not np.isnan(empirical_score)) else float(np.mean(valid))
    return {
        "mean": float(mean),
        "lower_ci": float(np.percentile(valid, 2.5)),
        "upper_ci": float(np.percentile(valid, 97.5)),
    }


def format_cell_ci(stats: dict[str, float], decimals: int = 3) -> str:
    """Formats AUROC point estimate and 95% confidence interval into a publication-ready string."""
    mean = stats["mean"]
    low = stats["lower_ci"]
    up = stats["upper_ci"]
    if np.isnan(mean):
        return "-"
    return f"{mean:.{decimals}f} [{low:.{decimals}f}, {up:.{decimals}f}]"


def _bootstrap_auroc_worker(
    chunk_seeds: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    patient_row_indices: list[list[int]],
) -> list[float]:
    """Computes AUROC for a chunk of random seeds using patient-clustered bootstrap resampling."""
    n_patients = len(patient_row_indices)
    results: list[float] = []

    for seed in chunk_seeds:
        rng = np.random.RandomState(seed)
        boot_subj_idx = rng.randint(0, n_patients, size=n_patients)

        boot_indices = []
        for subj_idx in boot_subj_idx:
            boot_indices.extend(patient_row_indices[subj_idx])

        indices = np.array(boot_indices, dtype=np.int32)
        y_boot = y_true[indices]
        score_boot = y_score[indices]

        if len(np.unique(y_boot)) < 2 or len(np.unique(score_boot)) < 2:
            results.append(np.nan)
        else:
            results.append(float(roc_auc_score(y_boot, score_boot)))

    return results


def run_clustered_bootstrap_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    subject_ids: np.ndarray,
    n_bootstraps: int = 10_000,
    num_workers: int = 1,
    seed: int = 42,
) -> dict[str, float]:
    """Runs parallel patient-level cluster bootstrap to compute AUROC distribution and 95% CI."""
    if len(y_true) == 0 or len(np.unique(y_true)) < 2 or len(np.unique(y_score)) < 2:
        return {"mean": np.nan, "lower_ci": np.nan, "upper_ci": np.nan}

    empirical = float(roc_auc_score(y_true, y_score))

    subj_map = collections.defaultdict(list)
    for row_idx, subj_id in enumerate(subject_ids):
        subj_map[subj_id].append(row_idx)
    patient_row_indices = list(subj_map.values())

    seeds = np.random.RandomState(seed).randint(0, 2**31 - 1, size=n_bootstraps)
    n_cores = min(num_workers, len(seeds))
    chunk_sz = math.ceil(len(seeds) / n_cores) if n_cores > 0 else 1
    chunks = [seeds[start_idx : start_idx + chunk_sz] for start_idx in range(0, len(seeds), chunk_sz)]

    if n_cores > 1:
        with multiprocessing.Pool(processes=n_cores) as pool:
            worker_results = pool.starmap(
                _bootstrap_auroc_worker,
                [(chunk, y_true, y_score, patient_row_indices) for chunk in chunks],
            )
        boot_scores = [score for chunk_res in worker_results for score in chunk_res]
    else:
        boot_scores = _bootstrap_auroc_worker(seeds, y_true, y_score, patient_row_indices)

    return calc_ci(boot_scores, empirical_score=empirical)


def export_table(df: pl.DataFrame, name: str, title: str, output_dir: Path) -> None:
    """Exports table to CSV, LaTeX, and Markdown formats."""
    # 1. Save CSV
    csv_path = output_dir / f"{name}.csv"
    df.write_csv(csv_path)

    # 2. Mask repeated modality for display formats
    pdf = df.to_pandas()
    pdf["Modality"] = pdf["Modality"].mask(pdf.duplicated(subset=["Modality"]), "")

    # 3. Save LaTeX
    tex_path = output_dir / f"{name}.tex"
    tex_path.write_text(pdf.to_latex(index=False, escape=True))

    # 4. Save Markdown
    md_path = output_dir / f"{name}.md"
    md_path.write_text(pdf.to_markdown(index=False))

    log.info(f"Saved {name} (.csv, .tex, .md) to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate error cohorts and extract cohort analysis statistics.")
    parser.add_argument("--split", type=str, default="held_out", choices=["train", "tuning", "held_out"])
    parser.add_argument("--tasks", nargs="+", default=os.environ.get("ACES_TASKS", "").split())
    parser.add_argument(
        "--struct_dir",
        type=Path,
        default=Path(os.environ.get("STRUCTURED_INFERENCE_DIR", "")) / "pt" / "rollout",
    )
    parser.add_argument(
        "--nl_dir",
        type=Path,
        default=Path(os.environ.get("NATURAL_LANGUAGE_INFERENCE_DIR", "")) / "cpt" / "rollout",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(os.environ.get("EVAL_PLOTS_DIR", "")) / "error_cohort_analysis",
    )
    default_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", min(8, os.cpu_count() or 1)))
    parser.add_argument(
        "--num_workers", type=int, default=default_workers, help="Number of parallel bootstrap workers."
    )
    parser.add_argument("--n_bootstraps", type=int, default=10_000, help="Number of bootstrap iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrapping.")
    args = parser.parse_args()
    args.tasks = sort_tasks(args.tasks)
    log.info(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        ("Structured", args.struct_dir),
        ("Natural Language", args.nl_dir),
    ]

    history_rows = []
    prevalence_rows = []
    err_auroc_rows = []
    clean_auroc_rows = []

    for mod_name, mod_dir in sources:
        log.info(f"--- Processing {mod_name} ---")
        for task in args.tasks:
            roll_df = load_predictions(mod_dir, task, args.split)
            n_patients = len(roll_df)

            cohorts = [assign_cohort(fa) for fa in roll_df["format_adherence"]]
            n_events = [extract_history_length(t) for t in roll_df["decoded_input"]]
            bool_vals = roll_df["boolean_value"].to_list()

            hist_row = {"Modality": mod_name, "Task": format_task_title(task)}
            prev_row = {"Modality": mod_name, "Task": format_task_title(task)}

            for cohort in COHORTS:
                idxs = [i for i, c in enumerate(cohorts) if c == cohort]
                cohort_n = len(idxs)
                prop = (cohort_n / n_patients * 100.0) if n_patients > 0 else 0.0

                c_events = [n_events[i] for i in idxs]
                if c_events:
                    med = float(np.median(c_events))
                    q25 = float(np.percentile(c_events, 25))
                    q75 = float(np.percentile(c_events, 75))
                    hist_row[cohort] = f"{med:.2f} [{q25:.2f}, {q75:.2f}] ({prop:.2f}%)"
                else:
                    hist_row[cohort] = "-"

                c_labels = [bool_vals[i] for i in idxs]
                if cohort_n > 0:
                    prev = sum(c_labels) / cohort_n * 100.0
                    prev_row[cohort] = f"{prev:.2f}% ({prop:.2f}%)"
                else:
                    prev_row[cohort] = "-"

            history_rows.append(hist_row)
            prevalence_rows.append(prev_row)

            # 1. AUROC based on error fraction
            log.info(f"Calculating AUROC based on error fraction for {mod_name} - {task}...")
            y_true_all = roll_df["boolean_value"].to_numpy().astype(int)
            subj_ids_all = roll_df["subject_id"].to_numpy()
            fa_list = roll_df["format_adherence"].to_list()
            err_fractions = np.array(
                [(len(fa) - sum(fa)) / len(fa) if len(fa) > 0 else 0.0 for fa in fa_list],
                dtype=float,
            )

            err_stats = run_clustered_bootstrap_auroc(
                y_true=y_true_all,
                y_score=err_fractions,
                subject_ids=subj_ids_all,
                n_bootstraps=args.n_bootstraps,
                num_workers=args.num_workers,
                seed=args.seed,
            )
            err_auroc_rows.append(
                {
                    "Modality": mod_name,
                    "Task": format_task_title(task),
                    "AUROC [95% CI]": format_cell_ci(err_stats),
                }
            )

            # 2. AUROC for clean samples (all trajectories have format_adherence == True)
            log.info(f"Calculating Clean Cohort AUROC for {mod_name} - {task}...")
            clean_mask = [all(fa) for fa in fa_list]
            clean_df = roll_df.filter(clean_mask)
            n_clean = len(clean_df)
            clean_prop = (n_clean / n_patients * 100.0) if n_patients > 0 else 0.0

            if n_clean > 0:
                y_true_clean = clean_df["boolean_value"].to_numpy().astype(int)
                subj_ids_clean = clean_df["subject_id"].to_numpy()
                outcomes_list = clean_df["outcome_predicted"].to_list()
                probs_clean = np.array(
                    [sum(1 for o in x if o is True) / len(x) if len(x) > 0 else 0.0 for x in outcomes_list],
                    dtype=float,
                )
                clean_stats = run_clustered_bootstrap_auroc(
                    y_true=y_true_clean,
                    y_score=probs_clean,
                    subject_ids=subj_ids_clean,
                    n_bootstraps=args.n_bootstraps,
                    num_workers=args.num_workers,
                    seed=args.seed,
                )
                clean_prev_str = f"{(y_true_clean.sum() / n_clean * 100.0):.2f}%"
            else:
                clean_stats = {"mean": np.nan, "lower_ci": np.nan, "upper_ci": np.nan}
                clean_prev_str = "-"

            clean_auroc_rows.append(
                {
                    "Modality": mod_name,
                    "Task": format_task_title(task),
                    "AUROC [95% CI]": format_cell_ci(clean_stats),
                    "Clean N (%)": f"{n_clean:,} ({clean_prop:.2f}%)",
                    "Outcome Prevalence (%)": clean_prev_str,
                }
            )

    hist_df = pl.DataFrame(history_rows)
    prev_df = pl.DataFrame(prevalence_rows)
    err_auroc_df = pl.DataFrame(err_auroc_rows)
    clean_auroc_df = pl.DataFrame(clean_auroc_rows)

    export_table(
        hist_df,
        "history_length",
        "History Length, Median [IQR] (Cohort Percentage)",
        args.output_dir,
    )
    export_table(
        prev_df,
        "outcome_prevalence",
        "Outcome Prevalence % (Cohort Percentage)",
        args.output_dir,
    )
    export_table(
        err_auroc_df,
        "auroc_error_fraction",
        "AUROC Based on Error Fraction [95% CI]",
        args.output_dir,
    )
    export_table(
        clean_auroc_df,
        "auroc_clean_cohort",
        "AUROC for Clean Format Adherent Trajectories [95% CI]",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
