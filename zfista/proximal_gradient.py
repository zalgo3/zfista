import time
from typing import Callable, List, Optional, Tuple
from warnings import warn

import numpy as np
from scipy.optimize import (
    BFGS,
    Bounds,
    LinearConstraint,
    OptimizeResult,
    minimize,
    minimize_scalar,
)

TERMINATION_MESSAGES = {
    0: "The maximum number of iterations is exceeded.",
    1: "Termination condition is satisfied.",
}

COLUMN_NAMES = [
    "niter",
    "nit internal",
    "max(abs(xk - yk)))",
    "subprob func",
    "learning rate",
]
COLUMN_WIDTHS = [7, 7, 13, 13, 10]
ITERATION_FORMATS = ["^7", "^7", "^+13.4e", "^+13.4e", "^10.2e"]


def _solve_subproblem(
    f: Callable,
    g: Callable,
    jac_f: Callable,
    prox_wsum_g: Callable,
    lr: float,
    xk_old: np.ndarray,
    yk: np.ndarray,
    w0: Optional[np.ndarray],
    tol: float = 1e-12,
    max_iter: int = 100000,
    deprecated: bool = False,
) -> OptimizeResult:
    r"""Solve the subproblem to get x^k that minimizes the objective function

    .. math::

        s + 1 / (2 * lr) * \|x - y^k\|^2

    where

    .. math::

        \nabla f_i(y^k)^\top (x - y^k) + g_i(x) + f_i(y^k) - F_i(x^{k - 1}) <= s.

    We solve the above problem by solving its dual that minimizes

    .. math::

        \inf_x (\sum_i w_i g_i(x) + \|x - y^k + lr * \sum_i w_i \nabla f_i(y^k)\|^2 / (2 * lr))
            - lr / 2 * \|\sum_i w_i \nabla f_i(y^k)\|^2 + \sum_i w_i (f_i(y^k) - F_i(x^{k - 1}))

    where

    .. math::

        \sum_i w_i = 1, w >= 0

    by using `scipy.optimize.minimize('method'='trust-constr')'.

    Parameters
    ----------
    f : callable
        A continuously differentiable vector-valued function.

            ``f(x) -> float or array_like, shape (n_objectives,)``

    g : callable
        A closed, proper and convex vector-valued function.

            ``g(x) -> float or array_like, shape (n_objectives,)``

    jac_f : callable
        A function that returns the Jacobian matrix of f.

            ``jac_f(x) -> array_like, shape (n_features,) or (n_objectives, n_features)``

    prox_wsum_g : callable
        Proximal operator of the weighted sum of g_i.

            ``prox_wsum_g(weight, x) -> array_like, shape (n_features,)``

    lr : float, default=1
        The learning rate.

    xk_old : array, shape (n_features,)

    yk : array, shape (n_features,)

    w0 : None or array, shape (n_objectives,)
        Initial guess for the dual problem.

    tol : float, default=1e-12
        The tolerance used in the `scipy` solver.

    max_iter : int, default=100000
        The maximum number of iterations in the `scipy` solver.

    deprecated : bool, default=False
        If True, the constraint is defined as

    .. math::

        \nabla f_i(y^k)^\top (x - y^k) + g_i(x) <= s.

    Returns
    ----------
    res : `scipy.minimize.OptimizeResult` with the fields documented below.

    x : array, shape (n_features,)
        Solution found.

    weight : array, shape (n_objectives,)
        Found solution to the dual problem.

    fun : float
        Primal subproblem's objective function at the solution.

    nit : int
        Total number of iterations in the solver.
    """
    f_yk = f(yk)
    F_xk_old = f(xk_old) + g(xk_old)
    jac_f_yk = jac_f(yk)
    n_objectives = f_yk.shape[0] if isinstance(f_yk, np.ndarray) else 1

    def _dual_minimized_fun_jac(weight: np.ndarray) -> Tuple[float, np.ndarray]:
        wsum_jac_f_yk = weight @ jac_f_yk
        yk_minus_lr_times_wsum_jac_f_yk = yk - lr * wsum_jac_f_yk
        primal_variable = prox_wsum_g(lr * weight, yk_minus_lr_times_wsum_jac_f_yk)
        g_primal_variable = g(primal_variable)
        fun = (
            -np.inner(weight, g_primal_variable)
            - np.linalg.norm(primal_variable - yk_minus_lr_times_wsum_jac_f_yk) ** 2
            / 2
            / lr
            + lr / 2 * np.linalg.norm(wsum_jac_f_yk) ** 2
        )
        jac = -g_primal_variable - jac_f_yk @ (primal_variable - yk)
        if not deprecated:
            fun += np.inner(weight, F_xk_old - f_yk)
            jac += F_xk_old - f_yk
        return fun, jac

    res = OptimizeResult()
    if n_objectives == 1:
        res.x = prox_wsum_g(lr, yk - lr * jac_f_yk.flatten())
        res.fun = float(
            jac_f_yk @ (res.x - yk)
            + g(res.x)
            + np.linalg.norm(res.x - yk) ** 2 / 2 / lr
        )
        if not deprecated:
            res.fun += f_yk - F_xk_old
        res.nit = 1
        return res
    if n_objectives == 2:

        def _dual_minimized_scalar_fun(w: float) -> float:
            return _dual_minimized_fun_jac(np.array([w, 1 - w]))[0]

        res_dual = minimize_scalar(
            _dual_minimized_scalar_fun,
            bounds=(0, 1),
            options={"maxiter": max_iter, "xatol": tol},
        )
        if not res_dual.success:
            warn(res_dual.message)
        res.weight = np.array([res_dual.x, 1 - res_dual.x])
    else:
        res_dual = minimize(
            fun=_dual_minimized_fun_jac,
            x0=w0,
            method="trust-constr",
            jac=True,
            hess=BFGS(),
            bounds=Bounds(lb=0, ub=np.inf),
            constraints=LinearConstraint(np.ones(n_objectives), lb=1, ub=1),
            options={"gtol": tol, "xtol": tol, "barrier_tol": tol, "maxiter": max_iter},
        )
        if not res_dual.success:
            warn(res_dual.message)
        res.weight = res_dual.x
    res.x = prox_wsum_g(lr * res.weight, yk - lr * res.weight @ jac_f_yk)
    res.fun = -res_dual.fun
    res.nit = res_dual.nit
    return res


def minimize_proximal_gradient(
    f: Callable,
    g: Callable,
    jac_f: Callable,
    prox_wsum_g: Callable,
    x0: np.ndarray,
    lr: float = 1,
    tol: float = 1e-5,
    tol_internal: float = 1e-12,
    max_iter: int = 1000000,
    max_iter_internal: int = 100000,
    max_backtrack_iter: int = 100,
    warm_start: bool = False,
    decay_rate: float = 0.5,
    nesterov: bool = False,
    nesterov_ratio: Tuple[float, float] = (0, 0.25),
    return_all: bool = False,
    verbose: bool = False,
    deprecated: bool = False,
) -> OptimizeResult:
    r"""Minimization of scalar or vector-valued function

    .. math::

        F(x) := f(x) + g(x)

    where :math:`f_i` is convex and continuously differentiable and :math:`g_i` is closed, proper and convex
    by using the proximal gradient method.

    Parameters
    ----------
    f : callable
        A continuously differentiable vector-valued function.

            ``f(x) -> float or array_like, shape (n_features,)``

    g : callable
        A closed, proper and convex vector-valued function.

            ``g(x) -> float or array_like, shape (n_features,)``

    jac_f : callable
        A function that returns the Jacobian matrix of :math:`f`.

            ``jac_f(x) -> array_like, shape (n_objectives, n_features)``

    prox_wsum_g : callable
        Proximal operator of the weighted sum of :math:`g_i`.

            ``prox_wsum_g(weight, x) -> array_like, shape (n_features,)``

    x0 : array, shape (n_features,)
        Initial guess.

    lr : float, default=1
        The learning rate.

    tol : float, default=1e-5
        The tolerance for the optimization: the algorithm checks the infinity norm (i.e. max abs
        value) of the search direction and continues until it is greater than ``tol``.

    tol_internal : float, default=1e-12
        The tolerance used in the solver (``scipy.optimize.minimize(method='trust-constr')``) for
        the subproblem.

    max_iter : int, default=1000000
        The maximum number of iterations.

    max_iter_internal : int, default=100000
        The maximum number of iterations in the solver (``scipy.optimize.minimize(method='trust-constr')``)
        for the subproblem

    warm_start : bool, default=False
        Use warm start in the subproblem.

    decay_rate : float, default=0.5
        Coefficient used to decay the learning rate.

    nesterov : bool, default=False
        If True, enable Nesterov's acceleration.

    nesterov_ratio : tuple of floats (a, b), default=(0, 0.25)
        Coefficients used for updating stepsize: :math:`t_{k + 1} = \sqrt{t_k^2 - a t_k + b} + 0.5`
        If ``nesterov`` is ``False``, then ``nesterov_ratio`` will be ignored.

    return_all : bool, default=False
        If True, return lists of the sequence :math:`\{x^k\}` and the error criteria :math:`\|x^k - y^k\|_\infty`.

    verbose : bool, default=False
        If True, display progress during iterations.

    deprecated : bool, default=False
        If True, uses the deprecated constraint in the subproblem. Note that using the deprecated option
        is not mathematically proven to converge, and it's advised to use the recommended condition instead.

    Returns
    ----------
    res : `scipy.minimize.OptimizeResult` with the fields documented below.

    x : array, shape (n_features,)
        Solution found.

    fun : float or array_like
        Objective functions at the solution.

    success : bool
        Whether or not the minimizer exited successfully.

    message : str
        Description of the cause of the termination.

    nit : int
        Total number of iterations of the proximal gradient method.

    nit_internal : int
        Total number of iterations in the solver for the subproblems.

    time : float
        Total time.

    allvecs : list of array, optional
        A list of the sequence :math:`\{x^k\}`

    allfuns : list of array, optional
        A list of the function values :math:`\{F(x^k)\}`

    allerrs : list of float, optional
        A list of the error criteria :math:`\|x^k - y^k||_\infty`
    """
    if deprecated:
        warn(
            "Using the deprecated option is not mathematically proven to converge. "
            "Please consider using the recommended condition instead."
        )
    start_time = time.time()
    res = OptimizeResult(
        x0=x0,
        tol=tol,
        tol_internal=tol_internal,
        nesterov=nesterov,
        nesterov_ratio=nesterov_ratio,
    )
    if verbose:
        fmt = "|" + "|".join(["{{:^{}}}".format(x) for x in COLUMN_WIDTHS]) + "|"
        separators = ["-" * x for x in COLUMN_WIDTHS]
        print(fmt.format(*COLUMN_NAMES))
        print(fmt.format(*separators))
    xk_old = x0
    xk = x0
    yk = x0
    nit_internal = 0
    f_x0 = f(x0)
    n_objectives = f_x0.shape[0] if isinstance(f_x0, np.ndarray) else 1
    w0 = np.ones(n_objectives) / n_objectives if n_objectives > 1 else None
    if return_all:
        allvecs = [x0]
        allfuns = [f_x0 + g(x0)]
        allerrs: List[float] = []
    if nesterov:
        nesterov_tk_old = 1
    for nit in range(1, max_iter + 1):
        F_xk_old = f(xk_old) + g(xk_old)
        backtrack_iter = 0
        while True:
            try:
                subproblem_result = _solve_subproblem(
                    f,
                    g,
                    jac_f,
                    prox_wsum_g,
                    lr,
                    xk_old,
                    yk,
                    w0,
                    tol=tol_internal,
                    max_iter=max_iter_internal,
                    deprecated=deprecated,
                )
                xk = subproblem_result.x
                F_xk = f(xk) + g(xk)
                nit_internal += subproblem_result.nit
                if w0 is not None and warm_start:
                    w0 = subproblem_result.weight
                if verbose:
                    progress_report = [
                        nit,
                        nit_internal,
                        max(abs(xk - yk)),
                        subproblem_result.fun,
                        lr,
                    ]
                    iteration_format = ["{{:{}}}".format(x) for x in ITERATION_FORMATS]
                    fmt = "|" + "|".join(iteration_format) + "|"
                    print(fmt.format(*progress_report))
                if decay_rate == 1:
                    break
                if deprecated:
                    if np.all(f(xk) - f(yk) <= subproblem_result.fun + tol):
                        break
                elif np.all(F_xk - F_xk_old <= subproblem_result.fun + tol):
                    break
                lr *= decay_rate
                backtrack_iter += 1
                if backtrack_iter >= max_backtrack_iter:
                    raise RuntimeError(
                        "Backtracking failed to find a suitable stepsize."
                    )
            except Exception as e:
                print(f"An error occurred: {e}")
                error_res = OptimizeResult()
                error_res.success = False
                error_res.message = f"Error: {str(e)}"
                error_res.x = xk_old
                error_res.fun = F_xk_old
                error_res.nit = nit - 1
                error_res.nit_internal = nit_internal
                if return_all:
                    error_res.allvecs = allvecs
                    error_res.allfuns = allfuns
                    error_res.allerrs = allerrs
                error_res.time = time.time() - start_time
                return error_res
        error_criterion = max(abs(xk - yk))
        if return_all:
            allvecs.append(xk)
            allfuns.append(F_xk)
            allerrs.append(error_criterion)
        if error_criterion < tol:
            res.status = 1
            break
        if nesterov:
            a, b = nesterov_ratio
            nesterov_tk = np.sqrt(nesterov_tk_old**2 - a * nesterov_tk_old + b) + 0.5
            moment = (nesterov_tk_old - 1) / nesterov_tk
            yk = xk + moment * (xk - xk_old)
            nesterov_tk_old = nesterov_tk
        else:
            yk = xk
        xk_old = xk
    else:
        res.status = 0
    res.x = xk
    res.fun = F_xk
    res.success = res.status == 1
    res.message = TERMINATION_MESSAGES[res.status]
    if not res.success:
        warn(res.message)
    res.nit = nit
    res.nit_internal = nit_internal
    if return_all:
        res.allvecs = allvecs
        res.allfuns = allfuns
        res.allerrs = allerrs
    res.time = time.time() - start_time
    return res
