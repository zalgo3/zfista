from typing import Dict, List, Tuple

import numpy as np
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.optimize import OptimizeResult


def extract_function_values(res: List[OptimizeResult]) -> np.ndarray:
    """
    Extract the objective function values from a list of OptimizeResult instances.

    Parameters
    ----------
    res : List[OptimizeResult]
        List of optimization results.

    Returns
    -------
    np.ndarray
        Array of objective function values.
    """
    return np.vstack([result.fun for result in res])


def extract_non_dominated_points(F: np.ndarray) -> np.ndarray:
    """
    Extract the non-dominated points from an objective function values array.

    Parameters
    ----------
    F : np.ndarray
        Array of objective function values.

    Returns
    -------
    np.ndarray
        Array of non-dominated points.
    """
    return F[NonDominatedSorting().do(F, only_non_dominated_front=True)]


def purity(front: np.ndarray, front_true: np.ndarray) -> float:
    """
    Compute the purity of an estimated Pareto front compared to the true Pareto front.

    Parameters
    ----------
    front : np.ndarray
        Array of points in the estimated Pareto front.
    front_true : np.ndarray
        Array of points in the true Pareto front.

    Returns
    -------
    float
        The purity of the estimated Pareto front.
    """
    return len(front) / len(front_true)


def spread_metrics(front: np.ndarray, front_true: np.ndarray) -> Tuple[float, float]:
    """
    Compute spread metrics (Gamma and Delta) between an estimated Pareto front and the true Pareto front.

    Parameters
    ----------
    front : np.ndarray
        Array of points in the estimated Pareto front.
    front_true : np.ndarray
        Array of points in the true Pareto front.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the Gamma and Delta values.
    """
    n_objectives = front_true.shape[1]
    gamma = 0
    delta = 0
    if len(front) <= 1:
        return np.inf, np.inf
    for j in range(n_objectives):
        F_j = np.sort(front[:, j])
        F_j_min = np.min(front_true[:, j])
        F_j_max = np.max(front_true[:, j])
        deltas = F_j[1:] - F_j[:-1]
        delta_start = F_j[0] - F_j_min
        delta_end = F_j_max - F_j[-1]
        gamma = max(np.max(deltas), delta_start, delta_end, gamma)
        avg_deltas = np.mean(deltas)
        numerators = delta_start + delta_end + np.sum(np.abs(deltas - avg_deltas))
        denominators = delta_start + delta_end + (len(F_j) - 1) * avg_deltas
        delta = max(delta, numerators / denominators)
    return gamma, delta


def calculate_metrics(
    *named_results: Tuple[str, List[OptimizeResult]]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Calculate a variety of performance metrics for a set of named optimization results.

    Parameters
    ----------
    *named_results : Tuple[str, List[OptimizeResult]]
        A tuple containing a string identifier and a list of optimization results.

    Returns
    -------
    Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]
        A pair of dictionaries containing the metrics and performance ratios for each optimization result.
    """
    result_names, results = zip(*named_results)
    fronts = [
        extract_non_dominated_points(extract_function_values(res)) for res in results
    ]
    front_true = extract_non_dominated_points(np.concatenate(fronts, axis=0))

    intersections = [
        np.array(
            list(
                set(tuple(point) for point in front_true).intersection(
                    set(tuple(point) for point in front)
                )
            )
        )
        for front in fronts
    ]

    purities = [purity(intersection, front_true) for intersection in intersections]

    spread_values = [
        spread_metrics(intersection, front_true) for intersection in intersections
    ]
    gammas, deltas = zip(*spread_values)

    hvs = [Hypervolume(pf=front_true)(front) for front in fronts]

    error_rates = [
        np.mean([not res.success for res in res_list]) for res_list in results
    ]

    avg_times = [
        np.mean([res.time for res in res_list if res.success]) for res_list in results
    ]

    avg_nits = [
        np.mean([res.nit for res in res_list if res.success]) for res_list in results
    ]

    avg_nit_internals = [
        np.mean([res.nit_internal for res in res_list if res.success])
        for res_list in results
    ]

    metrics_dict = {
        "Hypervolume": dict(zip(result_names, hvs)),
        "Gamma": dict(zip(result_names, gammas)),
        "Delta": dict(zip(result_names, deltas)),
        "Purity": dict(zip(result_names, purities)),
        "Error rate": dict(zip(result_names, error_rates)),
        "Avg computation time": dict(zip(result_names, avg_times)),
        "Avg iterations": dict(zip(result_names, avg_nits)),
        "Avg internal iterations": dict(zip(result_names, avg_nit_internals)),
    }
    ratios_dict = {}

    for key, values in metrics_dict.items():
        if key in ["Hypervolume", "Purity"]:
            best_value = max(values.values())
            ratios = {
                name: best_value / value
                if value != 0
                else np.inf
                if best_value != 0
                else 1
                for name, value in values.items()
            }
        else:
            best_value = min(values.values())
            ratios = {
                name: value / best_value
                if best_value != 0
                else np.inf
                if value != 0
                else 1
                for name, value in values.items()
            }

        ratios_dict[key] = ratios

    return metrics_dict, ratios_dict
