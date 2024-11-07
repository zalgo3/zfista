from __future__ import annotations

import contextlib
import inspect
import json
import os
import pickle
from collections.abc import Generator
from logging import INFO, StreamHandler, getLogger
from typing import Any, Callable, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import OptimizeResult
from tqdm.auto import tqdm

from zfista._typing import FloatArray
from zfista.metrics import (
    calculate_metrics,
    extract_function_values,
)
from zfista.problems import (
    FDS,
    JOS1,
    SD,
    TOI4,
    TRIDIA,
    ZDT1,
    LinearFunctionRank1,
    Problem,
)

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
plt.style.use(["science", "bright"])
plt.switch_backend("agg")


@contextlib.contextmanager
def tqdm_joblib(total: int | None = None, **kwargs: Any) -> Generator[tqdm, None, None]:
    pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):  # type: ignore[misc]
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            pbar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield pbar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        pbar.close()


def create_directory(problem: Problem, experiment_name: str) -> str:
    directory = os.path.join("results", experiment_name, problem.name)
    os.makedirs(directory, exist_ok=True)
    return directory


def show_Pareto_front(
    res_normal: list[OptimizeResult],
    res_acc: list[OptimizeResult],
    res_acc_deprecated: list[OptimizeResult],
    fname: str,
    iters: int = 10,
    s: float = 15,
    alpha: float = 0.75,
    elev: float = 15,
    azim: float = 130,
    linewidth: float = 0.1,
) -> None:
    if len(res_normal[0].fun) > 3:
        return
    F_normal = extract_function_values(res_normal)
    F_acc = extract_function_values(res_acc)
    F_acc_deprecated = extract_function_values(res_acc_deprecated)
    F_0 = np.array([res.allfuns[0] for res in res_normal])

    normal_color = "#6536FF"
    acc_color = "#e74c3c"
    acc_dep_color = "#3cc756"
    initial_color = "#8e44ad"

    common_style: dict[str, Any] = {"s": s, "alpha": alpha, "linewidth": linewidth}

    fig = plt.figure(figsize=(7.5, 7.5), dpi=100)
    if len(res_normal[0].fun) == 2:
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.scatter(
            F_0[:, 0],
            F_0[:, 1],
            label="Initial point",
            marker="x",
            color=initial_color,
            **common_style,
        )
        ax.scatter(
            F_normal[:, 0],
            F_normal[:, 1],
            label="Normal",
            marker="o",
            color=normal_color,
            **common_style,
        )
        ax.scatter(
            F_acc[:, 0],
            F_acc[:, 1],
            label="Accelerated",
            marker="^",
            color=acc_color,
            **common_style,
        )
        ax.scatter(
            F_acc_deprecated[:, 0],
            F_acc_deprecated[:, 1],
            label="Accelerated (without $f_i(y^k) - F_i(x^k)$)",
            marker="s",
            color=acc_dep_color,
            **common_style,
        )

        F_iter_normal = np.array(
            [res.allfuns[iters] for res in res_normal if res.nit >= iters]
        )
        F_iter_acc = np.array(
            [res.allfuns[iters] for res in res_acc if res.nit >= iters]
        )
        F_iter_acc_deprecated = np.array(
            [res.allfuns[iters] for res in res_acc_deprecated if res.nit >= iters]
        )

        if len(F_iter_normal) > 0:
            ax.scatter(
                F_iter_normal[:, 0],
                F_iter_normal[:, 1],
                label=f"Normal ({iters} iters)",
                marker="o",
                edgecolors=normal_color,
                facecolors="none",
                **common_style,
            )
        if len(F_iter_acc) > 0:
            ax.scatter(
                F_iter_acc[:, 0],
                F_iter_acc[:, 1],
                label=f"Accelerated ({iters} iters)",
                marker="^",
                edgecolors=acc_color,
                facecolors="none",
                **common_style,
            )
        if len(F_iter_acc_deprecated) > 0:
            ax.scatter(
                F_iter_acc_deprecated[:, 0],
                F_iter_acc_deprecated[:, 1],
                label=f"Accelerated (without $f_i(y^k) - F_i(x^k)$, {iters} iters)",
                marker="s",
                edgecolors=acc_dep_color,
                facecolors="none",
                **common_style,
            )

        ax.set_xlabel("$F_1$")
        ax.set_ylabel("$F_2$")
        ax.legend()
    elif len(res_normal[0].fun) == 3:
        ax_3d = cast(Axes3D, fig.add_subplot(111, projection="3d"))
        ax_3d.view_init(elev=elev, azim=azim)
        ax_3d.scatter(
            F_normal[:, 0],
            F_normal[:, 1],
            F_normal[:, 2],
            label="Normal",
            marker="o",
            color=normal_color,
            **common_style,
        )
        ax_3d.scatter(
            F_acc[:, 0],
            F_acc[:, 1],
            F_acc[:, 2],
            label="Accelerated",
            marker="^",
            color=acc_color,
            **common_style,
        )
        ax_3d.scatter(
            F_acc_deprecated[:, 0],
            F_acc_deprecated[:, 1],
            F_acc_deprecated[:, 2],
            label="Accelerated (without $f_i(y^k) - F_i(x^k)$)",
            marker="s",
            color=acc_dep_color,
            **common_style,
        )

        ax_3d.set_xlabel("$F_1$")
        ax_3d.set_ylabel("$F_2$")
        ax_3d.set_zlabel("$F_3$")
        ax_3d.legend()

    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def show_error_decay(
    res_normal: OptimizeResult,
    res_acc: OptimizeResult,
    res_acc_deprecated: OptimizeResult,
    fname: str,
) -> None:
    normal_color = "#6536FF"
    acc_color = "#e74c3c"
    acc_dep_color = "#3cc756"

    plt.figure(figsize=(7.5, 7.5), dpi=100)
    plt.yscale("log")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\|x^k - y^k\|_\infty$")
    plt.plot(res_normal.allerrs, label="Normal", color=normal_color, linestyle="dashed")
    plt.plot(res_acc.allerrs, label="Accelerated", color=acc_color)
    plt.plot(
        res_acc_deprecated.allerrs,
        label="Accelerated (without $f_i(y^k) - F_i(x^k)$)",
        color=acc_dep_color,
        linestyle="dotted",
    )
    plt.legend()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def save_results(
    problem: Problem,
    experiment_name: str,
    res_normal: list[OptimizeResult],
    res_acc: list[OptimizeResult],
    res_acc_deprecated: list[OptimizeResult],
    metrics: dict[str, dict[str, float]],
) -> None:
    logger.info("Saving results...")
    directory = create_directory(problem, experiment_name)
    with open(os.path.join(directory, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    show_Pareto_front(
        res_normal,
        res_acc,
        res_acc_deprecated,
        os.path.join(directory, "pareto_front.pdf"),
    )
    show_error_decay(
        res_normal[0],
        res_acc[0],
        res_acc_deprecated[0],
        os.path.join(directory, "error_decay.pdf"),
    )
    logger.info("Results saved.")


def load_or_run_results(
    file_name: str,
    directory: str,
    overwrite: bool,
    run_fn: Callable[[], list[OptimizeResult]],
) -> list[OptimizeResult]:
    if not overwrite and os.path.exists(os.path.join(directory, file_name)):
        try:
            logger.info(f"Loading {file_name}...")
            with open(os.path.join(directory, file_name), "rb") as f:
                results: list[OptimizeResult] = pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {file_name} due to: {e}")
            results = run_fn()
            with open(os.path.join(directory, file_name), "wb") as f:
                pickle.dump(results, f)
    else:
        logger.info(f"Running {file_name}...")
        results = run_fn()
        with open(os.path.join(directory, file_name), "wb") as f:
            pickle.dump(results, f)
    return results


def benchmark(
    problem: Problem,
    experiment_name: str,
    low: float | FloatArray,
    high: float | FloatArray,
    n_samples: int = 100,
    overwrite: bool = False,
    max_iter: int = 100000000,
    tol_internal: float = 1e-11,
    verbose: bool = False,
) -> tuple[list[OptimizeResult], list[OptimizeResult], list[OptimizeResult]]:
    directory = create_directory(problem, experiment_name)

    initial_points = np.random.uniform(
        low=low, high=high, size=(n_samples, problem.n_features)
    )

    with tqdm_joblib(total=n_samples, desc="Normal"):
        res_normal = load_or_run_results(
            "normal_results.pkl",
            directory,
            overwrite,
            lambda: Parallel(n_jobs=-1)(
                delayed(problem.minimize_proximal_gradient)(
                    x0,
                    return_all=True,
                    max_iter=max_iter,
                    tol_internal=tol_internal,
                    verbose=verbose,
                )
                for x0 in initial_points
            ),
        )
    with tqdm_joblib(total=n_samples, desc="Accelerated"):
        res_acc = load_or_run_results(
            "accelerated_results.pkl",
            directory,
            overwrite,
            lambda: Parallel(n_jobs=-1)(
                delayed(problem.minimize_proximal_gradient)(
                    x0,
                    nesterov=True,
                    return_all=True,
                    max_iter=max_iter,
                    tol_internal=tol_internal,
                    verbose=verbose,
                )
                for x0 in initial_points
            ),
        )
    with tqdm_joblib(
        total=n_samples, desc="Accelerated (without $f_i(y^k) - F_i(x^k)$)"
    ):
        res_acc_deprecated = load_or_run_results(
            "accelerated_results_deprecated.pkl",
            directory,
            overwrite,
            lambda: Parallel(n_jobs=-1)(
                delayed(problem.minimize_proximal_gradient)(
                    x0,
                    nesterov=True,
                    return_all=True,
                    deprecated=True,
                    max_iter=max_iter,
                    tol_internal=tol_internal,
                    verbose=verbose,
                )
                for x0 in initial_points
            ),
        )

    return res_normal, res_acc, res_acc_deprecated


def generate_performance_profiles(
    performance_ratios: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, tuple[FloatArray, FloatArray]]]:
    performance_profiles: dict[str, dict[str, tuple[FloatArray, FloatArray]]] = {}
    for ratio_key, algorithm_ratios in performance_ratios.items():
        performance_profiles[ratio_key] = {}
        for algorithm, ratios in algorithm_ratios.items():
            thresholds = []
            percentages = []
            for i, ratio in enumerate(sorted(ratios)):
                thresholds.append(ratio)
                percentages.append((i + 1) / len(ratios))
            performance_profiles[ratio_key][algorithm] = (
                np.array(thresholds),
                np.array(percentages),
            )
    return performance_profiles


def plot_performance_profiles(
    algorithm_profiles: dict[str, tuple[FloatArray, FloatArray]],
    fname: str,
) -> None:
    plt.figure(figsize=(7.5, 7.5), dpi=100)
    plt.xlabel("Threshold")
    plt.ylabel("Percentage of Problems")
    for algorithm, profile in algorithm_profiles.items():
        thresholds, percentages = profile
        plt.step(thresholds, percentages, label=algorithm)
    plt.legend()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def initialize_problems() -> list[Problem]:
    problem_classes = [
        JOS1,
        SD,
        TOI4,
        TRIDIA,
        LinearFunctionRank1,
        ZDT1,
        FDS,
    ]
    n_features_list = {
        JOS1: [5, 10, 20, 50, 100, 200, 500, 1000],
        ZDT1: [50, 100],
        FDS: [5, 10, 20, 50, 100],
        LinearFunctionRank1: [30],
    }
    problems = []
    for problem_class in problem_classes:
        constructor_params = inspect.signature(problem_class.__init__).parameters  # type: ignore[misc]
        if problem_class in n_features_list:
            for n_features in n_features_list[problem_class]:
                problem = problem_class(n_features=n_features)
                problems.append(problem)
                if (
                    "l1_ratios" in constructor_params
                    and "l1_shifts" in constructor_params
                ):
                    n_objectives = problem.n_objectives
                    l1_ratios = (np.arange(n_objectives) + 1) / n_features
                    l1_shifts = np.arange(n_objectives)
                    problems.append(
                        problem_class(
                            n_features=n_features,
                            l1_ratios=l1_ratios,
                            l1_shifts=l1_shifts,
                        )
                    )
        else:
            problem = problem_class()
            problems.append(problem)
            if "l1_ratios" in constructor_params and "l1_shifts" in constructor_params:
                n_features = problem.n_features
                n_objectives = problem.n_objectives
                l1_ratios = (np.arange(n_objectives) + 1) / n_features
                l1_shifts = np.arange(n_objectives)
                problems.append(problem_class(l1_ratios=l1_ratios, l1_shifts=l1_shifts))
    return problems


def main(overwrite: bool = False, verbose: bool = False) -> None:
    problems = initialize_problems()
    experiment_name = "proximal_vs_accelerated_proximal"
    problem_parameters = {
        "JOS1": {"low": -2, "high": 4},
        "FDS": {"low": -2, "high": 2},
        "SD": {"low": [1, np.sqrt(2), np.sqrt(2), 1], "high": [3, 3, 3, 3]},
        "ZDT1": {"low": 0, "high": 0.01},
        "TOI4": {"low": -2, "high": 5},
        "TRIDIA": {"low": -1, "high": 1},
        "LinearFunctionRank1": {"low": -1, "high": 1},
    }
    performance_ratios: dict[str, dict[str, list[float]]] = {}

    df_rows = []

    for problem in problems:
        logger.info(f"Running benchmark for {problem.name}...")
        problem_params = problem_parameters.get(type(problem).__name__)
        low, high = problem_params.get("low"), problem_params.get("high")  # type: ignore[attr-defined]
        res_normal, res_acc, res_acc_deprecated = benchmark(
            problem,
            experiment_name,
            low,
            high,
            overwrite=overwrite,
            verbose=verbose,
        )
        metrics, ratios = calculate_metrics(
            ("Normal", res_normal),
            ("Accelerated", res_acc),
            ("Accelerated (without $f_i(y^k) - F_i(x^k)$)", res_acc_deprecated),
        )
        # Add metrics to dataframe
        for metric_key, algorithms_metrics in metrics.items():
            df_rows.extend(
                [
                    {
                        "problem": problem.name,
                        "algorithm": algorithm,
                        "metric": metric_key,
                        "value": metric,
                    }
                    for algorithm, metric in algorithms_metrics.items()
                ]
            )

        for ratio_key, algorithms_ratios in ratios.items():
            if ratio_key not in performance_ratios:
                performance_ratios[ratio_key] = {}
            for algorithm, ratio in algorithms_ratios.items():
                if algorithm not in performance_ratios[ratio_key]:
                    performance_ratios[ratio_key][algorithm] = []
                performance_ratios[ratio_key][algorithm].append(ratio)
        save_results(
            problem, experiment_name, res_normal, res_acc, res_acc_deprecated, metrics
        )
        logger.info(f"Benchmark completed for {problem.name}.")

    performance_profiles = generate_performance_profiles(performance_ratios)
    for ratio_key, algorithm_profiles in performance_profiles.items():
        logger.info(f"Plotting performance profile for {ratio_key}...")
        plot_performance_profiles(
            algorithm_profiles,
            os.path.join("results", experiment_name, f"{ratio_key}.pdf"),
        )
    # Save metrics to csv
    df = pd.concat([pd.DataFrame(row, index=[0]) for row in df_rows], ignore_index=True)
    df.columns = ["problem", "algorithm", "metric", "value"]
    df.to_csv(os.path.join("results", experiment_name, "metrics.csv"), index=False)
