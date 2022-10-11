# Adding the CR_FM_NES strategy based on the numpy implementation
# that was provided by the authors:
# https://github.com/nomuramasahir0/crfmnes/blob/master/crfmnes/alg.py

# Note: this very slow first draft just just converts from
# numpy to jax and has massive room for improvement
# All inline lambda functions have been replaced with
# dedicated function definitions for clarity

import jax
import jax.numpy as np

# evaluation value of the infeasible solution
INFEASIBLE = np.inf


def get_h_inv(dim):
    # The usage of jit here is vital for execution speed
    @jax.jit
    def f(a, b):
        return ((1.0 + a * a) * np.exp(a * a / 2.0) / 0.24) - 10.0 - b

    @jax.jit
    def f_prime(a):
        return (1.0 / 0.24) * a * np.exp(a * a / 2.0) * (3.0 + a * a)

    h_inv = 1.0
    while abs(f(h_inv, dim)) > 1e-10:
        h_inv = h_inv - 0.5 * (f(h_inv, dim) / f_prime(h_inv))
    return h_inv


def sort_indices_by(evals, z):
    lam = evals.size
    sorted_indices = np.argsort(evals)
    sorted_evals = evals[sorted_indices]
    no_of_feasible_solutions = np.where(sorted_evals != INFEASIBLE)[0].size
    if no_of_feasible_solutions != lam:
        infeasible_z = z[:, np.where(evals == INFEASIBLE)[0]]
        distances = np.sum(infeasible_z**2, axis=0)
        infeasible_indices = sorted_indices[no_of_feasible_solutions:]
        indices_sorted_by_distance = np.argsort(distances)
        sorted_indices[no_of_feasible_solutions:] = infeasible_indices[
            indices_sorted_by_distance
        ]
    return sorted_indices


class CRFMNES:
    def __init__(self, dim, f, m, sigma, lamb, **kwargs):

        if "seed" in kwargs.keys():
            self.rng = jax.random.PRNGKey(kwargs["seed"])
        else:
            self.rng = jax.random.PRNGKey(0)

        print(1)

        self.dim = dim
        self.f = f
        self.m = m
        self.sigma = sigma
        self.lamb = lamb

        self.v = kwargs.get("v", jax.random.normal(self.rng, (dim, 1)) / np.sqrt(dim))
        self.D = np.ones([dim, 1])
        self.constraint = kwargs.get(
            "constraint", [[-np.inf, np.inf] for _ in range(dim)]
        )
        self.penalty_coef = kwargs.get("penalty_coef", 1e5)
        self.use_constraint_violation = kwargs.get("use_constraint_violation", True)

        print(2)

        self.w_rank_hat = (
            np.log(self.lamb / 2 + 1) - np.log(np.arange(1, self.lamb + 1))
        ).reshape(self.lamb, 1)

        self.w_rank_hat = self.w_rank_hat.at[np.where(self.w_rank_hat < 0)].set(0)
        self.w_rank = self.w_rank_hat / sum(self.w_rank_hat) - (1.0 / self.lamb)
        self.mueff = (
            1
            / ((self.w_rank + (1 / self.lamb)).T @ (self.w_rank + (1 / self.lamb)))[0][
                0
            ]
        )
        print(3)
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.cc = (4.0 + self.mueff / self.dim) / (
            self.dim + 4.0 + 2.0 * self.mueff / self.dim
        )
        self.c1_cma = 2.0 / (np.power(self.dim + 1.3, 2) + self.mueff)
        # initialization
        self.chiN = np.sqrt(self.dim) * (
            1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim * self.dim)
        )
        self.pc = np.zeros([self.dim, 1])
        self.ps = np.zeros([self.dim, 1])
        # distance weight parameter
        self.h_inv = get_h_inv(self.dim)

        print(4)
        # learning rate
        self.eta_m = 1.0
        self.eta_move_sigma = 1.0

        print(5)
        self.g = 0
        self.no_of_evals = 0

        self.idxp = np.arange(self.lamb / 2, dtype=int)
        self.idxm = np.arange(self.lamb / 2, self.lamb, dtype=int)
        self.z = np.zeros([self.dim, self.lamb])

        self.f_best = float("inf")
        self.x_best = np.empty(self.dim)

    def w_dist_hat(self, z, lambF):
        def alpha_dist(lambF, lamb, dim, h_inv):
            return (
                h_inv
                * min(1.0, np.sqrt(float(lamb) / dim))
                * np.sqrt(float(lambF) / lamb)
            )

        return np.exp(
            alpha_dist(lambF, self.lamb, self.dim, self.h_inv) * np.linalg.norm(z)
        )

    def eta_stag_sigma(self, lambF, dim):
        return np.tanh((0.024 * lambF + 0.7 * dim + 20.0) / (dim + 12.0))

    def eta_conv_sigma(self, lambF, dim):
        return 2.0 * np.tanh((0.025 * lambF + 0.75 * dim + 10.0) / (dim + 4.0))

    def c1(self, lambF):
        return self.c1_cma * (self.dim - 5) / 6 * (float(lambF) / self.lamb)

    def eta_B(self, lambF):
        return np.tanh(
            (min(0.02 * lambF, 3 * np.log(self.dim)) + 5) / (0.23 * self.dim + 25)
        )

    def calc_violations(self, x):
        violations = np.zeros(self.lamb)
        for i in range(self.lamb):
            for j in range(self.dim):
                violations = violations.at[i].set(
                    violations[i]
                    + (
                        -min(0, x[j][i] - self.constraint[j][0])
                        + max(0, x[j][i] - self.constraint[j][1])
                    )
                    * self.penalty_coef
                )
        return violations

    def optimize(self, iterations):
        for _ in range(iterations):
            print(f"f_best: {self.f_best}")
            _ = self.one_iteration()
        return self.x_best, self.f_best

    def one_iteration(self):
        d = self.dim
        lamb = self.lamb
        zhalf = jax.random.normal(self.rng, (d, int(lamb / 2)))  # dim x lamb/2

        self.z = self.z.at[:, self.idxp].set(zhalf)
        self.z = self.z.at[:, self.idxm].set(-zhalf)
        normv = np.linalg.norm(self.v)
        normv2 = normv**2
        vbar = self.v / normv
        y = self.z + (np.sqrt(1 + normv2) - 1) * vbar @ (vbar.T @ self.z)
        x = self.m + self.sigma * y * self.D
        evals_no_sort = np.array(
            [self.f(np.array(x[:, i].reshape(self.dim, 1))) for i in range(self.lamb)]
        )
        xs_no_sort = [x[:, i] for i in range(lamb)]

        violations = np.zeros(lamb)
        if self.use_constraint_violation:
            violations = self.calc_violations(x)
            sorted_indices = sort_indices_by(evals_no_sort + violations, self.z)
        else:
            sorted_indices = sort_indices_by(evals_no_sort, self.z)
        best_eval_id = sorted_indices[0]
        f_best = evals_no_sort[best_eval_id]
        x_best = x[:, best_eval_id]
        self.z = self.z[:, sorted_indices]
        y = y[:, sorted_indices]
        x = x[:, sorted_indices]

        self.no_of_evals += self.lamb
        self.g += 1
        if f_best < self.f_best:
            self.f_best = f_best
            self.x_best = x_best

        # This operation assumes that if the solution is infeasible, infinity comes in as input.
        lambF = np.sum(evals_no_sort < np.finfo(float).max)

        # evolution path p_sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(
            self.cs * (2.0 - self.cs) * self.mueff
        ) * (self.z @ self.w_rank)
        ps_norm = np.linalg.norm(self.ps)
        # distance weight
        w_tmp = np.array(
            [
                self.w_rank_hat[i] * self.w_dist_hat(np.array(self.z[:, i]), lambF)
                for i in range(self.lamb)
            ]
        ).reshape(self.lamb, 1)
        weights_dist = w_tmp / sum(w_tmp) - 1.0 / self.lamb
        # switching weights and learning rate
        weights = weights_dist if ps_norm >= self.chiN else self.w_rank
        eta_sigma = (
            self.eta_move_sigma
            if ps_norm >= self.chiN
            else self.eta_stag_sigma(lambF, self.dim)
            if ps_norm >= 0.1 * self.chiN
            else self.eta_conv_sigma(lambF, self.dim)
        )
        # update pc, m
        wxm = (x - self.m) @ weights
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(
            self.cc * (2.0 - self.cc) * self.mueff
        ) * wxm / self.sigma
        self.m += self.eta_m * wxm
        # calculate s, t
        # step1
        normv4 = normv2**2
        exY = np.append(y, self.pc / self.D, axis=1)  # dim x lamb+1
        yy = exY * exY  # dim x lamb+1
        ip_yvbar = vbar.T @ exY
        yvbar = exY * vbar  # dim x lamb+1. exYのそれぞれの列にvbarがかかる
        gammav = 1.0 + normv2
        vbarbar = vbar * vbar
        alphavd = np.min(
            np.array(
                [
                    1,
                    np.sqrt(normv4 + (2 * gammav - np.sqrt(gammav)) / np.max(vbarbar))
                    / (2 + normv2),
                ]
            )
        )  # scalar
        t = exY * ip_yvbar - vbar * (ip_yvbar**2 + gammav) / 2  # dim x lamb+1
        b = -(1 - alphavd**2) * normv4 / gammav + 2 * alphavd**2
        H = np.ones([self.dim, 1]) * 2 - (b + 2 * alphavd**2) * vbarbar  # dim x 1
        invH = H ** (-1)
        s_step1 = (
            yy
            - normv2 / gammav * (yvbar * ip_yvbar)
            - np.ones([self.dim, self.lamb + 1])
        )  # dim x lamb+1
        ip_vbart = vbar.T @ t  # 1 x lamb+1
        s_step2 = s_step1 - alphavd / gammav * (
            (2 + normv2) * (t * vbar) - normv2 * vbarbar @ ip_vbart
        )  # dim x lamb+1
        invHvbarbar = invH * vbarbar
        ip_s_step2invHvbarbar = invHvbarbar.T @ s_step2  # 1 x lamb+1
        s = (s_step2 * invH) - b / (
            1 + b * vbarbar.T @ invHvbarbar
        ) * invHvbarbar @ ip_s_step2invHvbarbar  # dim x lamb+1
        ip_svbarbar = vbarbar.T @ s  # 1 x lamb+1
        t = t - alphavd * (
            (2 + normv2) * (s * vbar) - vbar @ ip_svbarbar
        )  # dim x lamb+1
        # update v, D
        exw = np.append(
            self.eta_B(lambF) * weights,
            np.array([self.c1(lambF)]).reshape(1, 1),
            axis=0,
        )  # lamb+1 x 1
        self.v = self.v + (t @ exw) / normv
        self.D = self.D + (s @ exw) * self.D
        # calculate detA
        nthrootdetA = np.exp(
            np.sum(np.log(self.D)) / self.dim
            + np.log(1 + self.v.T @ self.v) / (2 * self.dim)
        )[0][0]
        self.D = self.D / nthrootdetA
        # update sigma
        G_s = (
            np.sum((self.z * self.z - np.ones([self.dim, self.lamb])) @ weights)
            / self.dim
        )
        self.sigma = self.sigma * np.exp(eta_sigma / 2 * G_s)

        return xs_no_sort, evals_no_sort, violations


if __name__ == "__main__":
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    def ellipsoid(x):
        x = x.reshape(-1)
        dim = len(x)
        return np.sum(
            np.array([(1000 ** (i / (dim - 1)) * x[i]) ** 2 for i in range(dim)])
        )

    def test_run_d40_ellipsoid():
        print("test_run_d40:")
        dim = 40
        mean = np.ones([dim, 1]) * 3
        sigma = 2.0
        lamb = 16  # note that lamb (sample size) should be even number
        allowable_evals = (8.8 + 0.5 * 3) * 1e3  # 2 sigma
        iteration_number = int(allowable_evals / lamb) + 1

        cr = CRFMNES(dim, ellipsoid, mean, sigma, lamb)
        x_best, f_best = cr.optimize(iteration_number)
        print("f_best:{}".format(f_best))
        assert f_best < 1e-12

    test_run_d40_ellipsoid()
