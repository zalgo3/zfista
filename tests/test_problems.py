import unittest

import numpy as np

from zfista.problems import FDS, JOS1, SD


class TestJOS1(unittest.TestCase):
    def setUp(self) -> None:
        self.jos1 = JOS1()

    def test_f(self) -> None:
        x = np.array([1, 2, 3, 4, 5])
        expected = np.array([11, 3])
        result = self.jos1.f(x)
        np.testing.assert_almost_equal(result, expected)

    def test_jac_f(self) -> None:
        x = np.array([1, 2, 3, 4, 5])
        expected = np.array(
            [[2 / 5, 4 / 5, 6 / 5, 8 / 5, 10 / 5], [-2 / 5, 0, 2 / 5, 4 / 5, 6 / 5]]
        )
        result = self.jos1.jac_f(x)
        np.testing.assert_almost_equal(result, expected)


class TestJOS1_L1(unittest.TestCase):
    def setUp(self) -> None:
        self.jos1_l1 = JOS1(l1_ratios=[0.2, 0.1], l1_shifts=[0, 1])

    def test_g(self) -> None:
        x = np.array([1, 2, 3, 4, 5])
        expected = np.array([3, 1])
        result = self.jos1_l1.g(x)
        np.testing.assert_almost_equal(result, expected)

    def test_prox_wsum_g(self) -> None:
        weight = np.array([0.5, 0.5])
        x = np.array([3, 4, 5, 6, 7])
        expected = np.array([2.85, 3.85, 4.85, 5.85, 6.85])
        result = self.jos1_l1.prox_wsum_g(weight, x)
        np.testing.assert_almost_equal(result, expected)


class TestSD(unittest.TestCase):
    def setUp(self) -> None:
        self.sd = SD()

    def test_f(self) -> None:
        x = np.array([1, np.sqrt(2), np.sqrt(2), 1])
        expected = np.array([7, 8])
        result = self.sd.f(x)
        np.testing.assert_almost_equal(result, expected)

    def test_jac_f(self) -> None:
        x = np.array([1, np.sqrt(2), np.sqrt(2), 1])
        expected = np.array(
            [[2, np.sqrt(2), np.sqrt(2), 1], [-2, -np.sqrt(2), -np.sqrt(2), -2]]
        )
        result = self.sd.jac_f(x)
        np.testing.assert_almost_equal(result, expected)

    def test_g(self) -> None:
        x = np.array([1, np.sqrt(2), np.sqrt(2), 1])
        expected = np.array([0, 0])
        result = self.sd.g(x)
        np.testing.assert_almost_equal(result, expected)

    def test_prox_wsum_g(self) -> None:
        weight = np.array([0.5, 0.5])
        x = np.array([1, np.sqrt(2), np.sqrt(2), 1])
        expected = np.array([1, np.sqrt(2), np.sqrt(2), 1])
        result = self.sd.prox_wsum_g(weight, x)
        np.testing.assert_almost_equal(result, expected)


class TestFDS(unittest.TestCase):
    def setUp(self) -> None:
        self.fds = FDS(n_features=5)

    def test_f(self) -> None:
        x = np.array([1, 2, 3, 4, 5])
        expected = np.array([0.0, 75.0855369, 0.1183459])
        result = self.fds.f(x)
        np.testing.assert_almost_equal(result, expected)

    def test_jac_f(self) -> None:
        x = np.array([1, 2, 3, 4, 5])
        expected = np.array(
            [
                [0, 0, 0, 0, 0],
                [6.01710738, 8.01710738, 10.0171074, 12.0171074, 14.0171074],
                [
                    -0.0613132402,
                    -0.0360894089,
                    -0.0149361205,
                    -4.88417037e-03,
                    -1.12299117e-03,
                ],
            ]
        )
        result = self.fds.jac_f(x)
        np.testing.assert_almost_equal(result, expected)


class TestFDS_CONSTRAINED(unittest.TestCase):
    def setUp(self) -> None:
        self.fds_constrained = FDS(n_features=5, bounds=(0, np.inf))

    def test_g(self) -> None:
        x = np.ones(self.fds_constrained.n_features)
        expected = np.array([0, 0, 0])
        result = self.fds_constrained.g(x)
        np.testing.assert_almost_equal(result, expected)

        x = -np.ones(self.fds_constrained.n_features)
        expected = np.array([np.inf, np.inf, np.inf])
        result = self.fds_constrained.g(x)
        np.testing.assert_almost_equal(result, expected)

    def test_prox_wsum_g(self) -> None:
        weight = np.array([1 / 3, 1 / 3, 1 / 3])
        x = np.array([-3, -1, 0, 1, 3])
        expected = np.array([0, 0, 0, 1, 3])
        result = self.fds_constrained.prox_wsum_g(weight, x)
        np.testing.assert_almost_equal(result, expected)
