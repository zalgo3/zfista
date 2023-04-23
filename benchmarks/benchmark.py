import contextlib
import inspect
import json
import os
import pickle
from logging import INFO, StreamHandler, getLogger
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from joblib import Parallel, delayed
from scipy.optimize import OptimizeResult
from tqdm.auto import tqdm

from zfista.metrics import (
    calculate_metrics,
    extract_function_values,
    extract_non_dominated_points,
    spread_metrics,
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


@contextlib.contextmanager
def tqdm_joblib(total: Optional[int] = None, **kwargs) -> Generator:
    pbar = tqdm(total=total, miniters=1, smoothing=0, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
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
    res_normal: List[OptimizeResult],
    res_acc: List[OptimizeResult],
    res_acc_deprecated: List[OptimizeResult],
    iters: int = 10,
    s: float = 15,
    alpha: float = 0.75,
    elev: float = 15,
    azim: float = 130,
    linewidth: float = 0.1,
    fname: Optional[str] = None,
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

    common_style = {"s": s, "alpha": alpha, "linewidth": linewidth}

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

        ax.set_xlabel(f"$F_1$")
        ax.set_ylabel(f"$F_2$")
        ax.legend()
    elif len(res_normal[0].fun) == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        ax.scatter(
            F_normal[:, 0],
            F_normal[:, 1],
            F_normal[:, 2],
            label="Normal",
            marker="o",
            color=normal_color,
            **common_style,
        )
        ax.scatter(
            F_acc[:, 0],
            F_acc[:, 1],
            F_acc[:, 2],
            label="Accelerated",
            marker="^",
            color=acc_color,
            **common_style,
        )
        ax.scatter(
            F_acc_deprecated[:, 0],
            F_acc_deprecated[:, 1],
            F_acc_deprecated[:, 2],
            label="Accelerated (without $f_i(y^k) - F_i(x^k)$)",
            marker="s",
            color=acc_dep_color,
            **common_style,
        )

        ax.set_xlabel(f"$F_1$")
        ax.set_ylabel(f"$F_2$")
        ax.set_zlabel(f"$F_3$")
        ax.legend()

    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def show_error_decay(
    res_normal: OptimizeResult,
    res_acc: OptimizeResult,
    res_acc_deprecated: OptimizeResult,
    fname: Optional[str] = None,
):
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
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def save_results(
    problem: Problem,
    experiment_name: str,
    res_normal: List[OptimizeResult],
    res_acc: List[OptimizeResult],
    res_acc_deprecated: List[OptimizeResult],
    metrics: Dict[str, Dict[str, float]],
) -> None:
    logger.info("Saving results...")
    directory = create_directory(problem, experiment_name)
    with open(os.path.join(directory, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    show_Pareto_front(
        res_normal,
        res_acc,
        res_acc_deprecated,
        fname=os.path.join(directory, "pareto_front.pdf"),
    )
    show_error_decay(
        res_normal[0],
        res_acc[0],
        res_acc_deprecated[0],
        fname=os.path.join(directory, "error_decay.pdf"),
    )
    logger.info("Results saved.")


def load_or_run_results(
    file_name: str,
    directory: str,
    overwrite: bool,
    run_fn: Callable,
) -> List[OptimizeResult]:
    if not overwrite and os.path.exists(os.path.join(directory, file_name)):
        try:
            logger.info(f"Loading {file_name}...")
            with open(os.path.join(directory, file_name), "rb") as f:
                results = pickle.load(f)
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
    low: Union[float, np.ndarray],
    high: Union[float, np.ndarray],
    n_samples: int = 100,
    overwrite: bool = False,
    max_iter: int = 100000000,
    verbose: bool = False,
) -> Tuple[List[OptimizeResult], List[OptimizeResult], List[OptimizeResult]]:
    directory = create_directory(problem, experiment_name)

    initial_points = np.random.uniform(
        low=low, high=high, size=(n_samples, problem.n_features)
    )

    with tqdm_joblib(total=n_samples, desc="Normal") as progress_bar:
        res_normal = load_or_run_results(
            "normal_results.pkl",
            directory,
            overwrite,
            lambda: Parallel(n_jobs=-1)(
                delayed(problem.minimize_proximal_gradient)(
                    x0,
                    return_all=True,
                    max_iter=max_iter,
                    max_iter_internal=max_iter,
                    verbose=verbose,
                )
                for x0 in initial_points
            ),
        )
    with tqdm_joblib(total=n_samples, desc="Accelerated") as progress_bar:
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
                    max_iter_internal=max_iter,
                    verbose=verbose,
                )
                for x0 in initial_points
            ),
        )
    with tqdm_joblib(
        total=n_samples, desc="Accelerated (without $f_i(y^k) - F_i(x^k)$)"
    ) as progress_bar:
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
                    max_iter_internal=max_iter,
                    verbose=verbose,
                )
                for x0 in initial_points
            ),
        )

    return res_normal, res_acc, res_acc_deprecated


def generate_performance_profiles(
    performance_ratios: Dict[str, Dict[str, List[float]]]
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    performance_profiles: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
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
    metric_key: str,
    algorithm_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]],
    fname: Optional[str],
) -> None:
    plt.figure(figsize=(7.5, 7.5), dpi=100)
    plt.xlabel("Threshold")
    plt.ylabel("Percentage of Problems")
    for algorithm, profile in algorithm_profiles.items():
        thresholds, percentages = profile
        plt.step(thresholds, percentages, label=algorithm)
    plt.legend()
    if fname is not None:
        plt.savefig(fname, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def main(overwrite=False, verbose=False) -> None:
    n_features_list = [5, 10, 20, 50, 100, 200, 500, 1000]

    problem_classes = [
        JOS1,
        FDS,
        SD,
        ZDT1,
        TOI4,
        TRIDIA,
        LinearFunctionRank1,
    ]

    problems = []

    for problem_class in problem_classes:
        constructor_params = inspect.signature(problem_class.__init__).parameters  # type: ignore
        if "n_features" in constructor_params:
            for n_features in n_features_list:
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
    performance_ratios: Dict[str, Dict[str, List[float]]] = {}

    df_rows = []

    for problem in problems:
        logger.info(f"Running benchmark for {problem.name}...")
        problem_params = problem_parameters.get(type(problem).__name__)
        low, high = problem_params.get("low"), problem_params.get("high")  # type: ignore
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
            ratio_key,
            algorithm_profiles,
            os.path.join("results", experiment_name, f"{ratio_key}.pdf"),
        )
    # Save metrics to csv
    df = pd.concat([pd.DataFrame(row, index=[0]) for row in df_rows], ignore_index=True)
    df.columns = ["problem", "algorithm", "metric", "value"]
    df.to_csv(os.path.join("results", experiment_name, "metrics.csv"), index=False)
