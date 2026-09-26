"""Unit tests for the posterior-predictive fix in bayes_log_odds.

Verifies against brute-force enumeration:
  P(M | Y, c, rho, r) = sum_R P(R | Y; r) * [rho*1[M = f_c(R,a)] + (1-rho)*1[M = R]]
and checks that bayes_log_odds returns
  log( sum_theta post[theta]*P(M|Y=1,theta) / sum_theta post[theta]*P(M|Y=0,theta) ).
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pvi_experiments as P

rng = np.random.default_rng(0)


def brute_likelihood(m, a1, a2, y, c, rho, r):
    total = 0.0
    for R in (0, 1):
        pR = r if R == y else 1 - r
        fm = P.apply_convention(c, np.array([R]), np.array([a1]), np.array([a2]))[0]
        pM = rho * (1.0 if fm == m else 0.0) + (1 - rho) * (1.0 if R == m else 0.0)
        total += pR * pM
    return total


def test_per_theta_likelihood_matches_brute_force():
    # Reconstruct the vectorized p1/p0 inside bayes_log_odds via a degenerate posterior
    for trial in range(20):
        c = int(rng.integers(0, P.N_CONV))
        rho = float(rng.choice(P.RHO_GRID))
        r = float(rng.choice(P.R_GRID))
        m = int(rng.integers(0, 2)); a1 = int(rng.integers(0, 2)); a2 = int(rng.integers(0, 2))
        post = np.zeros((1, len(P.RHO_GRID), len(P.R_GRID)))
        post[0, list(P.RHO_GRID).index(rho), list(P.R_GRID).index(r)] = 1.0
        lo = P.bayes_log_odds(post, m, a1, a2, [c])
        b1 = brute_likelihood(m, a1, a2, 1, c, rho, r)
        b0 = brute_likelihood(m, a1, a2, 0, c, rho, r)
        assert abs(lo - (np.log(b1) - np.log(b0))) < 1e-9, \
            f"trial {trial}: got {lo}, want {np.log(b1)-np.log(b0)}"
    print("PASS: per-theta likelihood matches brute force (20 trials)")


def test_predictive_is_log_of_ratio_not_expectation_of_log():
    for trial in range(20):
        nc = 3
        cands = list(rng.choice(P.N_CONV, size=nc, replace=False))
        post = rng.random((nc, len(P.RHO_GRID), len(P.R_GRID)))
        post /= post.sum()
        m = int(rng.integers(0, 2)); a1 = int(rng.integers(0, 2)); a2 = int(rng.integers(0, 2))
        got = P.bayes_log_odds(post, m, a1, a2, cands)
        p1_bar = p0_bar = 0.0
        for i, c in enumerate(cands):
            for j, rho in enumerate(P.RHO_GRID):
                for k, r in enumerate(P.R_GRID):
                    w = post[i, j, k]
                    p1_bar += w * brute_likelihood(m, a1, a2, 1, int(c), float(rho), float(r))
                    p0_bar += w * brute_likelihood(m, a1, a2, 0, int(c), float(rho), float(r))
        want = np.log(p1_bar) - np.log(p0_bar)
        assert abs(got - want) < 1e-9, f"trial {trial}: got {got}, want {want}"
    print("PASS: predictive log-odds = log(E[p1]/E[p0]) (20 trials)")


def test_degenerate_posterior_equals_oracle():
    for trial in range(20):
        c = int(rng.integers(0, P.N_CONV))
        rho = float(rng.choice(P.RHO_GRID))
        r = float(rng.choice(P.R_GRID))
        m = int(rng.integers(0, 2)); a1 = int(rng.integers(0, 2)); a2 = int(rng.integers(0, 2))
        post = np.zeros((1, len(P.RHO_GRID), len(P.R_GRID)))
        post[0, list(P.RHO_GRID).index(rho), list(P.R_GRID).index(r)] = 1.0
        got = P.bayes_log_odds(post, m, a1, a2, [c])
        want = P.oracle_log_odds(m, a1, a2, c, rho, r)
        assert abs(got - want) < 1e-9, f"trial {trial}: got {got}, want {want}"
    print("PASS: degenerate posterior == oracle (20 trials)")


def test_concentrated_posterior_near_oracle():
    # Posterior concentrated on the true theta -> predictive ~= oracle
    c, rho, r = 3, 0.9, 0.8
    post = np.zeros((P.N_CONV, len(P.RHO_GRID), len(P.R_GRID)))
    post[c, list(P.RHO_GRID).index(rho), list(P.R_GRID).index(r)] = 0.97
    post /= post.sum()
    for m in (0, 1):
        for a1 in (0, 1):
            for a2 in (0, 1):
                got = P.bayes_log_odds(post, m, a1, a2, list(range(P.N_CONV)))
                want = P.oracle_log_odds(m, a1, a2, c, rho, r)
                assert abs(got - want) < 0.5, f"m={m}: got {got}, want {want}"
    print("PASS: concentrated posterior close to oracle")


if __name__ == "__main__":
    test_per_theta_likelihood_matches_brute_force()
    test_predictive_is_log_of_ratio_not_expectation_of_log()
    test_degenerate_posterior_equals_oracle()
    test_concentrated_posterior_near_oracle()
    print("ALL UNIT TESTS PASSED")
