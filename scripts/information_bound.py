"""
information_bound.py

Numerical companion to "Toward a Principle of Significance" (Materov, 2026).

Tests, on small finite-dimensional quantum systems with an exact SDP solver,
the two one-shot information-theoretic results the paper's main bound rests on:

  (1) Lemma 4.6.1 (one-shot converse). For any states rho, sigma and any
      eps in (0, 1/2):
          D_H^eps(rho || sigma) <= [D(rho||sigma) + h(eps)] / (1 - eps)
      where D_H^eps is the hypothesis-testing relative entropy (Wang & Renner,
      2012) and D is the ordinary (Umegaki) quantum relative entropy.

  (2) Theorem 4.6.4 / Proposition 9.6.1 (the full N-ary chain). For a
      uniform N-ary ensemble {rho_1,...,rho_N} jointly discriminable by a
      single POVM with average success probability 1-eps:
          ln N <= [chi + h(eps)] / (1 - eps)
      where chi is the Holevo chi-quantity of the ensemble. Here eps is
      obtained from the OPTIMAL decoding POVM, found via a semidefinite
      program (the N-ary Holevo-Helstrom problem) -- the tightest possible
      test of the bound, since any other decoder would only make eps larger
      and the bound easier to satisfy.

Both are checked over many random trials (Hilbert-Schmidt / Ginibre random
density matrices), across a range of N and Hilbert-space dimensions, plus
two structured edge cases: orthogonal states (near-saturating the bound)
and near-identical states (chi~0, checks the bound stays non-vacuous).

Dependencies: numpy, cvxpy (with the SCS or CLARABEL solver).
"""

import numpy as np
import cvxpy as cp


# --------------------------------------------------------------------------
# Basic quantum-information utilities
# --------------------------------------------------------------------------

def random_density_matrix(d, rng):
    """Random density matrix on C^d, Hilbert-Schmidt (Ginibre) measure."""
    G = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = G @ G.conj().T
    rho /= np.trace(rho).real
    return rho


def von_neumann_entropy(rho, tol=1e-12):
    eigs = np.linalg.eigvalsh(rho)
    eigs = eigs[eigs > tol]
    return float(-np.sum(eigs * np.log(eigs)))


def binary_entropy(eps):
    if eps <= 0 or eps >= 1:
        return 0.0
    return float(-eps * np.log(eps) - (1 - eps) * np.log(1 - eps))


def relative_entropy(rho, sigma, tol=1e-12):
    """Umegaki relative entropy D(rho||sigma) = Tr[rho(log rho - log sigma)].
    Caller must ensure supp(rho) subset supp(sigma) (guaranteed here by
    regularizing sigma to be full rank)."""
    evals_r, evecs_r = np.linalg.eigh(rho)
    evals_s, evecs_s = np.linalg.eigh(sigma)
    log_r = evecs_r @ np.diag(np.where(evals_r > tol, np.log(np.clip(evals_r, tol, None)), 0.0)) @ evecs_r.conj().T
    log_s = evecs_s @ np.diag(np.log(np.clip(evals_s, tol, None))) @ evecs_s.conj().T
    return float(np.trace(rho @ (log_r - log_s)).real)


def hypothesis_testing_relative_entropy(rho, sigma, eps, solver=cp.SCS):
    """
    D_H^eps(rho||sigma) = -ln min{ Tr[Q sigma] : 0<=Q<=I, Tr[Q rho]>=1-eps },
    solved exactly as a semidefinite program (Wang & Renner, 2012).
    """
    d = rho.shape[0]
    Q = cp.Variable((d, d), hermitian=True)
    constraints = [
        Q >> 0,
        np.eye(d) - Q >> 0,
        cp.real(cp.trace(Q @ rho)) >= 1 - eps,
    ]
    objective = cp.Minimize(cp.real(cp.trace(Q @ sigma)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, eps=1e-9) if solver == cp.SCS else prob.solve(solver=solver)
    min_val = max(prob.value, 1e-300)
    return -np.log(min_val)


def holevo_chi(states, weights=None):
    n = len(states)
    if weights is None:
        weights = np.ones(n) / n
    rho_bar = sum(w * r for w, r in zip(weights, states))
    return von_neumann_entropy(rho_bar) - sum(w * von_neumann_entropy(r) for w, r in zip(weights, states))


def optimal_decoder_success(states, weights=None, solver=cp.SCS):
    """
    Optimal N-ary quantum state discrimination via SDP (Holevo-Helstrom):
    maximize (1/N) sum_i Tr[M_i rho_i] over POVMs {M_i}, M_i>=0, sum M_i = I.
    Returns the optimal average success probability.
    """
    n = len(states)
    d = states[0].shape[0]
    if weights is None:
        weights = np.ones(n) / n
    Ms = [cp.Variable((d, d), hermitian=True) for _ in range(n)]
    constraints = [M >> 0 for M in Ms]
    constraints.append(sum(Ms) == np.eye(d))
    objective = cp.Maximize(sum(weights[i] * cp.real(cp.trace(Ms[i] @ states[i])) for i in range(n)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, eps=1e-9) if solver == cp.SCS else prob.solve(solver=solver)
    return float(prob.value)


# --------------------------------------------------------------------------
# Part 1: Lemma 4.6.1 standalone test
# --------------------------------------------------------------------------

def test_lemma_4_6_1(n_trials=200, dims=(2, 3, 4), seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    violations = 0
    margins = []
    for _ in range(n_trials):
        d = int(rng.choice(dims))
        rho = random_density_matrix(d, rng)
        sigma_raw = random_density_matrix(d, rng)
        sigma = 0.999 * sigma_raw + 0.001 * np.eye(d) / d  # ensure full rank
        eps = float(rng.uniform(0.02, 0.48))
        DH = hypothesis_testing_relative_entropy(rho, sigma, eps)
        D = relative_entropy(rho, sigma)
        h = binary_entropy(eps)
        rhs = (D + h) / (1 - eps)
        margins.append(rhs - DH)
        if DH > rhs + 1e-6:
            violations += 1
    if verbose:
        print(f"  trials={n_trials}  violations={violations}  "
              f"min margin (RHS-LHS)={min(margins):.6f}  "
              f"max margin={max(margins):.6f}")
    return violations, margins


# --------------------------------------------------------------------------
# Part 2: Theorem 4.6.4 / Proposition 9.6.1 full-chain test
# --------------------------------------------------------------------------

def test_theorem_4_6_4(n_trials=100, N_values=(2, 3, 4, 6), dims=(3, 4, 6), seed=1, verbose=True):
    rng = np.random.default_rng(seed)
    violations = 0
    records = []
    attempts = 0
    while len(records) < n_trials and attempts < n_trials * 4:
        attempts += 1
        N = int(rng.choice(N_values))
        d = int(rng.choice(dims))
        if d < N:
            continue
        states = [random_density_matrix(d, rng) for _ in range(N)]
        chi = holevo_chi(states)
        p_succ = optimal_decoder_success(states)
        eps = max(1 - p_succ, 1e-9)
        if eps >= 0.5:
            continue  # Lemma 4.6.1's proof requires eps < 1/2
        h = binary_entropy(eps)
        rhs = (chi + h) / (1 - eps)
        lhs = np.log(N)
        records.append((N, d, chi, eps, lhs, rhs))
        if lhs > rhs + 1e-6:
            violations += 1
    if verbose:
        margins = [r[5] - r[4] for r in records]
        print(f"  valid trials={len(records)}  violations={violations}  "
              f"min margin (RHS-LHS)={min(margins):.6f}  "
              f"max margin={max(margins):.6f}")
    return violations, records


# --------------------------------------------------------------------------
# Structured edge cases
# --------------------------------------------------------------------------

def orthogonal_states_case(d=4):
    """N=d orthogonal pure states: perfect discrimination (eps->0), the
    tightest possible test of the bound for this N (chi is maximal, = ln d)."""
    states = [np.outer(np.eye(d)[i], np.eye(d)[i].conj()) for i in range(d)]
    chi = holevo_chi(states)
    p_succ = optimal_decoder_success(states)
    eps = max(1 - p_succ, 1e-12)
    return chi, eps, np.log(d)


def near_identical_states_case(d=4, delta=1e-3, seed=2):
    """N states that are nearly identical: chi~0, eps~1-1/N (best decoder
    can barely beat guessing) -- checks the bound is not vacuous here."""
    rng = np.random.default_rng(seed)
    base = random_density_matrix(d, rng)
    N = 3
    states = []
    for i in range(N):
        perturb = delta * random_density_matrix(d, rng)
        s = base + perturb
        s = 0.5 * (s + s.conj().T)
        w, v = np.linalg.eigh(s)
        w = np.clip(w, 0, None)
        s = (v * w) @ v.conj().T
        s /= np.trace(s).real
        states.append(s)
    chi = holevo_chi(states)
    p_succ = optimal_decoder_success(states)
    eps = max(1 - p_succ, 1e-9)
    return chi, eps, np.log(N)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 78)
    print("Lemma 4.6.1 (one-shot converse: D_H^eps <= [D + h(eps)]/(1-eps))")
    print("=" * 78)
    test_lemma_4_6_1(n_trials=200)

    print()
    print("=" * 78)
    print("Theorem 4.6.4 / Proposition 9.6.1 (ln N <= [chi + h(eps)]/(1-eps))")
    print("=" * 78)
    test_theorem_4_6_4(n_trials=100)

    print()
    print("=" * 78)
    print("Structured edge cases")
    print("=" * 78)

    chi, eps, lnN = orthogonal_states_case()
    h = binary_entropy(eps)
    rhs = (chi + h) / (1 - eps)
    status = "OK" if lnN <= rhs + 1e-6 else "VIOLATION"
    print(f"  Orthogonal states (N=d=4): ln N={lnN:.4f}  chi={chi:.4f}  "
          f"eps={eps:.2e}  RHS={rhs:.4f}  -> {status}")

    chi, eps, lnN = near_identical_states_case()
    h = binary_entropy(eps)
    rhs = (chi + h) / (1 - eps)
    status = "OK" if lnN <= rhs + 1e-6 else "VIOLATION"
    print(f"  Near-identical states (N=3): ln N={lnN:.4f}  chi={chi:.4f}  "
          f"eps={eps:.4f}  RHS={rhs:.4f}  -> {status}")

    print()
    print("All checks completed. 'VIOLATION' anywhere above would falsify")
    print("Lemma 4.6.1 / Theorem 4.6.4 / Proposition 9.6.1 as stated.")
