import time
from warnings import warn

import numpy as np
from scipy.optimize import (BFGS, Bounds, LinearConstraint, OptimizeResult,
                            minimize)

TERMINATION_MESSAGES = {
    0: "The maximum number of function evaluations is exceeded.",
    1: "Termination condition is satisfied.",
}

COLUMN_NAMES = ['niter', 'nit internal', 'max(abs(xk - yk)))', 'subprob func', 'learning rate']
COLUMN_WIDTHS = [7, 7, 13, 13, 10]
ITERATION_FORMATS = ["^7", "^7", "^+13.4e", "^+13.4e", "^10.2e"]


def _solve_subproblem(f, g, jac_f, prox_wsum_g, lr, xk_old, yk, w0, tol=1e-12):
    """Solve the subproblem to get x^k that minimizes the objective function::

        s + 1 / (2 * lr) * ||x - y^k||^2

    where::

        <\\nabla f_i(y^k), (x - y^k)> + g_i(x) + f_i(y^k) - F_i(x^{k - 1}) <= s.

    We solve the above problem by solving its dual that minimizes::

        \\inf_x (\\sum_i w_i g_i(x) + ||x - y^k + lr * \\sum_i w_i \\nabla f_i(y^k)||^2 / (2 * lr))
            - lr / 2 * ||\\sum_i w_i \\nabla f_i(y^k)||^2 + \\sum_i w_i (f_i(y^k) - F_i(x^{k - 1}))

    where::

        \\sum_i w_i = 1, w >= 0

    by using `scipy.optimize.minimize('method'='trust-constr')'.

    Parameters
    ----------
    f : callable
        A continuously differentiable vector-valued function.

            ``f(x) -> float or array_like, shape (m_dims,)``

    g : callable
        A closed, proper and convex vector-valued function.

            ``g(x) -> float or array_like, shape (m_dims,)``

    jac_f : callable
        A function that returns the Jacobian matrix of f.

            ``jac_f(x) -> array_like, shape (n_dims,) or (m_dims, n_dims)``

    prox_wsum_g : callable
        Proximal operator of the weighted sum of g_i.

            ``prox_wsum_g(weight, x) -> array_like, shape (n_dims,)``

    lr : float, default=1
        The learning rate.

    xk_old : array, shape (n_dims,)

    yk : array, shape (n_dims,)

    w0 : None or array, shape (m_dims,)
        Initial guess for the dual problem.

    tol : float, default=1e-12
        The tolerance used in the `scipy` solver.

    Returns
    ----------
    res : `scipy.minimize.OptimizeResult` with the fields documented below.

    x : array, shape (n_dims,)
        Solution found.

    weight : array, shape (m_dims,)
        Found solution to the dual problem.

    fun : float
        Primal subproblem's objective function at the solution.

    nit : int
        Total number of iterations in the solver.
    """
    f_yk = f(yk)
    F_xk_old = f(xk_old) + g(xk_old)
    jac_f_yk = jac_f(yk)
    m_dims = f_yk.shape[0] if isinstance(f_yk, np.ndarray) else 1

    def _dual_minimized_fun_jac(weight):
        wsum_jac_f_yk = weight.T @ jac_f_yk
        yk_minus_lr_times_wsum_jac_f_yk = yk - lr * wsum_jac_f_yk
        primal_variable = prox_wsum_g(lr * weight, yk_minus_lr_times_wsum_jac_f_yk)
        g_primal_variable = g(primal_variable)
        fun = -np.inner(weight, g_primal_variable) \
              - np.linalg.norm(primal_variable - yk_minus_lr_times_wsum_jac_f_yk) ** 2 / 2 / lr \
              + lr / 2 * np.linalg.norm(wsum_jac_f_yk) ** 2 \
              - np.inner(weight, f_yk - F_xk_old)
        jac = -g_primal_variable - jac_f_yk @ (primal_variable - yk) - f_yk + F_xk_old
        return fun, jac

    res = OptimizeResult()
    if m_dims == 1:
        res.x = prox_wsum_g(lr, yk - lr * jac_f_yk.flatten())
        res.fun = float(jac_f_yk @ (res.x - yk) + g(res.x) + f_yk - F_xk_old + np.linalg.norm(res.x - yk) **2 / 2 / lr)
        res.nit = 1
    else:
        res_dual = minimize(fun=_dual_minimized_fun_jac, x0=w0,
                            method='trust-constr', jac=True,
                            hess=BFGS(), bounds=Bounds(0, np.inf),
                            constraints=LinearConstraint(np.ones(m_dims), 1, 1),
                            options={'gtol': tol, 'xtol': tol, 'barrier_tol': tol}
                            )
        if not res_dual.success:
            warn(res_dual.message)
        res.weight = res_dual.x
        res.x = prox_wsum_g(lr * res.weight, yk - lr * res.weight.T @ jac_f_yk)
        res.fun = -res_dual.fun
        res.nit = res_dual.nit
    return res


def minimize_proximal_gradient(f, g, jac_f, prox_wsum_g, x0, lr=1, tol=1e-5, tol_internal=1e-10,
                               max_iter=10000, warm_start=False,
                               decay_rate=0.5, nesterov=False, nesterov_ratio=(0, 0.25),
                               return_all=False, verbose=False):
    """Minimization of scalar or vector-valued function::

        F(x) := f(x) + g(x)

    where f_i is convex and continuously differentiable and g_i is closed, proper and convex
    by using the proximal gradient method.

    Parameters
    ----------
    f : callable
        A continuously differentiable vector-valued function.

            ``f(x) -> float or array_like, shape (n_dims,)``

    g : callable
        A closed, proper and convex vector-valued function.

            ``g(x) -> float or array_like, shape (n_dims,)``

    jac_f : callable
        A function that returns the Jacobian matrix of f.

            ``jac_f(x) -> array_like, shape (m_dims, n_dims)``

    prox_wsum_g : callable
        Proximal operator of the weighted sum of g_i.

            ``prox_wsum_g(weight, x) -> array_like, shape (n_dims,)``

    x0 : array, shape (n_dims,)
        Initial guess.

    lr : float, default=1
        The learning rate.

    tol : float, default=1e-5
        The tolerance for the optimization: the algorithm checks the infinity norm (i.e. max abs
        value) of the search direction and continues until it is greater than ``tol``.

    tol_internal : float, default=1e-12
        The tolerance used in the solver (`scipy.optimize.minimize(method='trust-constr')`) for
        the subproblem.

    max_iter : int, default=10000
        The maximum number of iterations.

    warm_start : bool, default=False
        Use warm start in the subproblem.

    decay_rate : float, default=0.5
        Coefficient used to decay the learning rate.

    nesterov : bool, default=False
        If True, enable Nesterov's acceleration.

    nesterov_ratio : tuple of floats (a, b), default=(0, 0.25)
        Coefficients used for updating stepsize: t_{k + 1} = \\sqrt{t_k^2 - a * t_k + b} + 0.5
        If `nesterov` is False, then `nesterov_ratio` will be ignored.

    return_all : bool, default=False
        If True, return a list of the sequence {x^k}.

    verbose : bool, default=False
        If True, display progress during iterations.

    Returns
    ----------
    res : `scipy.minimize.OptimizeResult` with the fields documented below.

    x : array, shape (n_dims,)
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

    execution_time : float
        Total execution time.

    allvecs : list of array, optional
        A list of the sequence {x^k}
    """
    start_time = time.time()
    res = OptimizeResult(x0=x0, tol=tol, tol_internal=tol_internal, nesterov=nesterov, nesterov_ratio=nesterov_ratio)
    if verbose:
        fmt = ("|"
               + "|".join(["{{:^{}}}".format(x) for x in COLUMN_WIDTHS])
               + "|")
        separators = ['-' * x for x in COLUMN_WIDTHS]
        print(fmt.format(*COLUMN_NAMES))
        print(fmt.format(*separators))
    xk_old = x0
    yk = x0
    nit_internal = 0
    f_x0 = f(x0)
    m_dims = f_x0.shape[0] if isinstance(f_x0, np.ndarray) else 1
    w0 = np.ones(m_dims) / m_dims if m_dims > 1 else None
    if return_all:
        allvecs = [x0]
    if nesterov:
        nesterov_tk_old = 1
    for nit in range(1, max_iter + 1):
        F_xk_old = f(xk_old) + g(xk_old)
        while True:
            subproblem_result = _solve_subproblem(f, g, jac_f, prox_wsum_g, lr, xk_old, yk, w0,
                                                  tol=tol_internal)
            xk = subproblem_result.x
            nit_internal += subproblem_result.nit
            if w0 is not None and warm_start:
                w0 = subproblem_result.weight
            if verbose:
                progress_report = [nit, nit_internal, max(abs(xk - yk)), subproblem_result.fun, lr]
                iteration_format = ["{{:{}}}".format(x) for x in ITERATION_FORMATS]
                fmt = "|" + "|".join(iteration_format) + "|"
                print(fmt.format(*progress_report))
            if decay_rate == 1:
                break
            if np.all(f(xk) + g(xk) - F_xk_old <= subproblem_result.fun + tol):
                break
            lr *= decay_rate
        if return_all:
            allvecs.append(xk)
        if max(abs(xk - yk)) < tol:
            res.status = 1
            break
        if nesterov:
            a, b = nesterov_ratio
            nesterov_tk = np.sqrt(nesterov_tk_old ** 2 - a * nesterov_tk_old + b) + 0.5
            moment = (nesterov_tk_old - 1) / nesterov_tk
            yk = xk + moment * (xk - xk_old)
            nesterov_tk_old = nesterov_tk
        else:
            yk = xk
        xk_old = xk
    else:
        res.status = 0
    res.x = xk
    res.fun = f(xk) + g(xk)
    res.success = (res.status == 1)
    res.message = TERMINATION_MESSAGES[res.status]
    if not res.success:
        warn(res.message)
    res.nit = nit
    res.nit_internal = nit_internal
    if return_all:
        res.allvecs = allvecs
    res.execution_time = time.time() - start_time
    return res
