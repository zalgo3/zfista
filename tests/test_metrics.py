import unittest
from typing import Dict

import numpy as np
from scipy.optimize import OptimizeResult

from zfista.metrics import (
    calculate_metrics,
    extract_function_values,
    extract_non_dominated_points,
    purity,
    spread_metrics,
)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.res = [
            OptimizeResult(
                fun=np.array([0.1, 0.2]), success=True, time=1, nit=10, nit_internal=5
            ),
            OptimizeResult(
                fun=np.array([0.2, 0.1]), success=True, time=2, nit=20, nit_internal=10
            ),
            OptimizeResult(
                fun=np.array([0.3, 0.3]), success=True, time=3, nit=30, nit_internal=15
            ),
        ]
        self.function_values = np.array(
            [
                [0.1, 0.2],
                [0.2, 0.1],
                [0.3, 0.3],
            ]
        )

    def test_extract_function_values(self):
        result = extract_function_values(self.res)
        np.testing.assert_array_equal(result, self.function_values)

    def test_extract_non_dominated_points(self):
        non_dominated_points = extract_non_dominated_points(self.function_values)
        expected_points = np.array(
            [
                [0.1, 0.2],
                [0.2, 0.1],
            ]
        )
        np.testing.assert_array_equal(non_dominated_points, expected_points)

    def test_purity(self):
        front = np.array(
            [
                [0.1, 0.2],
                [0.2, 0.1],
            ]
        )
        front_true = np.array(
            [
                [0.1, 0.2],
                [0.2, 0.1],
                [0.3, 0.3],
            ]
        )
        self.assertAlmostEqual(purity(front, front_true), 2 / 3)

    def test_spread_metrics(self):
        front = np.array(
            [
                [0.1, 0.2],
                [0.2, 0.1],
            ]
        )
        front_true = np.array(
            [
                [0.1, 0.2],
                [0.2, 0.1],
                [0.3, 0.3],
            ]
        )
        gamma, delta = spread_metrics(front, front_true)
        self.assertAlmostEqual(gamma, 0.1)
        self.assertAlmostEqual(delta, 0.5)

    def test_calculate_metrics(self):
        named_results = [("result", self.res)]
        metrics, ratios = calculate_metrics(*named_results)
        expected_metrics = {
            "Hypervolume": {"result": 0},
            "Gamma": {"result": 0.1},
            "Delta": {"result": 0},
            "Purity": {"result": 1.0},
            "Error rate": {"result": 0.0},
            "Avg computation time": {"result": 2.0},
            "Avg iterations": {"result": 20.0},
            "Avg internal iterations": {"result": 10.0},
        }
        expected_ratios = {
            "Hypervolume": {"result": 1},
            "Gamma": {"result": 1},
            "Delta": {"result": 1},
            "Purity": {"result": 1},
            "Error rate": {"result": 1},
            "Avg computation time": {"result": 1},
            "Avg iterations": {"result": 1},
            "Avg internal iterations": {"result": 1},
        }
        self.compare_metrics(metrics, expected_metrics)
        self.compare_metrics(ratios, expected_ratios)

    def compare_metrics(
        self,
        actual_metrics: Dict[str, Dict[str, float]],
        expected_metrics: Dict[str, Dict[str, float]],
    ):
        for key in actual_metrics:
            for sub_key in actual_metrics[key]:
                self.assertAlmostEqual(
                    actual_metrics[key][sub_key], expected_metrics[key][sub_key]
                )
