# Numerical Companion to "Toward a Principle of Significance"

This repository accompanies Sergej Materov (2026), *Toward a Principle of
Significance: A Holographic Bound on the Physically Distinguishable
Configuration Space of a Bounded Observer*.

It contains two independent, self-contained checks, matching the two
distinct kinds of claim the paper makes:

## `scripts/information_bound.py`

Tests the **information-theoretic engine** of the paper — Lemma 4.6.1 (the
one-shot hypothesis-testing converse) and Theorem 4.6.4 / Proposition 9.6.1
(the full N-ary discrimination bound `ln N ≤ [χ + h(ε)]/(1−ε)`) — on small,
explicit finite-dimensional quantum systems, with both sides of each
inequality computed *exactly* via semidefinite programming (not simulated
approximately). Includes:

- 200 random trials of Lemma 4.6.1 across Hilbert space dimensions 2–4.
- 100 random trials of the full Theorem 4.6.4 / Proposition 9.6.1 chain,
  using the numerically **optimal** decoding POVM (the tightest possible
  test — any other decoder would only make the bound easier to satisfy).
- Two structured edge cases: orthogonal states (near-saturates the bound)
  and near-identical states (checks the bound is non-vacuous when χ≈0).

No violation of either inequality is expected or found in any run; a
violation anywhere would falsify the corresponding claim in the paper.

```
python3 scripts/information_bound.py
```

Requires: `numpy`, `cvxpy` (with the `SCS` solver, installed by default with
`pip install cvxpy`).

## `scripts/physical_crosscheck.py`

Reproduces, from real cosmological data, the **one part of Section 6's
numerical cross-check that is actually computable from first principles**:
the de Sitter horizon entropy S_dS, via the Gibbons–Hawking (1977) area law,
using the measured Hubble constant H0. Computed for both commonly cited
values (Planck 2018 CMB: 67.4 km/s/Mpc; SH0ES 2022 local distance ladder:
73.04 km/s/Mpc), to show the result is insensitive to the "Hubble tension."
Lloyd's (2002) independently obtained computational-capacity figure
(~10¹²⁰ operations, ~10⁹⁰–10¹²⁰ bits) is cited directly rather than
re-derived, to avoid manufacturing a false match or mismatch through
different, unstated modelling choices.

```
python3 scripts/physical_crosscheck.py
```

Requires: nothing beyond the Python standard library.

## What these scripts do and do not establish

`information_bound.py` confirms the paper's one-shot information theory is
internally consistent on explicit finite-dimensional examples — it does not,
and cannot, substitute for the analytic proofs in the paper (Lemmas
4.6.1–4.6.3, Theorem 4.6.4), which hold for arbitrary finite dimension.
`physical_crosscheck.py` confirms one specific numerical claim (§6's
"~10¹²⁰" order-of-magnitude statement) against current observational data;
it says nothing about the paper's conditional theorem (Theorem 4.3) itself,
which remains conditional on (S3b) as stated in the paper.

## Citation

If you use this repository, please cite:

Materov, S. (2026). *Toward a Principle of Significance: A Holographic
Bound on the Physically Distinguishable Configuration Space of a Bounded
Observer.*

## License

CC BY-NC 4.0, matching the parent paper.
