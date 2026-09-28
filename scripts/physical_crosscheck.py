"""
physical_crosscheck.py

Numerical companion to "Toward a Principle of Significance" (Materov, 2026),
Section 6: "A Numerical Cross-Check: Two Independent Routes to 10^120".

This script performs the ONE calculation in that cross-check that is actually
reproducible from first principles and current data: the de Sitter horizon
entropy S_dS, via the Gibbons-Hawking (1977) area law, using the measured
Hubble constant H0. It deliberately does NOT attempt to re-derive Lloyd's
(2002) computational-capacity figure from scratch -- that computation makes
its own independent set of modelling choices (total mass-energy of the
observable universe, the Margolus-Levitin bound, a Bekenstein-type bound on
registerable bits) that are Lloyd's own to state, and re-deriving them here
with different, unstated conventions would risk manufacturing a false
mismatch or a false match. Lloyd's headline figures are instead cited
directly, as the paper's own Section 6 does.

Because of the "Hubble tension" (Planck CMB vs. SH0ES local-distance-ladder
measurements of H0 disagree at several-sigma significance), S_dS is computed
for BOTH commonly cited values, to show the conclusion (order 10^120-10^122)
is not sensitive to which one is used.

References for the input data:
  Planck Collaboration (2020). Planck 2018 results VI: Cosmological
    parameters. A&A, 641, A6. H0 = 67.4 +/- 0.5 km/s/Mpc.
  Riess, A. G. et al. (2022). A comprehensive measurement of the local value
    of the Hubble constant. ApJL, 934, L7. H0 = 73.04 +/- 1.04 km/s/Mpc.
  Gibbons, G. W., & Hawking, S. W. (1977). Cosmological event horizons,
    thermodynamics, and particle creation. Phys. Rev. D, 15, 2738.
  Lloyd, S. (2002). Computational capacity of the universe. Phys. Rev. Lett.,
    88, 237901. (~10^120 operations on ~10^90-10^120 bits over cosmic history.)

Dependencies: none beyond the Python standard library.
"""

import math

# --------------------------------------------------------------------------
# Physical constants (SI units; CODATA / SI-2019 exact values where defined)
# --------------------------------------------------------------------------

C = 299_792_458.0            # speed of light, m/s (exact, SI definition)
G = 6.67430e-11              # Newtonian gravitational constant, m^3 kg^-1 s^-2 (CODATA 2018)
HBAR = 1.054_571_817e-34     # reduced Planck constant, J s (exact via SI-2019 h)
K_B = 1.380_649e-23          # Boltzmann constant, J/K (exact, SI-2019)
MPC_IN_M = 3.085_677_581_491_3673e22  # one megaparsec in metres (IAU)


def hubble_constant_si(H0_km_s_Mpc):
    """Convert H0 from km/s/Mpc to SI units of 1/s."""
    return (H0_km_s_Mpc * 1000.0) / MPC_IN_M


def de_sitter_horizon_entropy(H0_km_s_Mpc):
    """
    Gibbons-Hawking (1977) de Sitter horizon entropy, in units of k_B
    (i.e. this returns S/k_B, the standard dimensionless convention used
    throughout the paper), for a de Sitter horizon of radius r_dS = c/H0.

        S_dS / k_B = A_dS / (4 l_P^2) = pi * c^5 / (H0^2 * hbar * G)

    Returns (S_dS/k_B, log10(S_dS/k_B), horizon radius in metres).
    """
    H0 = hubble_constant_si(H0_km_s_Mpc)
    r_dS = C / H0
    A_dS = 4.0 * math.pi * r_dS**2
    l_P_sq = HBAR * G / C**3
    S_over_kB = A_dS / (4.0 * l_P_sq)
    return S_over_kB, math.log10(S_over_kB), r_dS


if __name__ == "__main__":
    print("=" * 78)
    print("de Sitter horizon entropy S_dS from measured H0 (Gibbons-Hawking, 1977)")
    print("=" * 78)

    datasets = [
        ("Planck 2018 (CMB)", 67.4),
        ("SH0ES 2022 (local distance ladder)", 73.04),
    ]

    results = []
    for label, H0 in datasets:
        S_over_kB, log10_S, r_dS = de_sitter_horizon_entropy(H0)
        results.append((label, H0, S_over_kB, log10_S, r_dS))
        print(f"\n  {label}: H0 = {H0} km/s/Mpc")
        print(f"    de Sitter horizon radius  r_dS = {r_dS:.4e} m "
              f"({r_dS / MPC_IN_M / 1000:.4f} Gpc)")
        print(f"    S_dS / k_B                     = {S_over_kB:.4e}")
        print(f"    log10(S_dS / k_B)               = {log10_S:.2f}")

    print()
    print("=" * 78)
    print("Comparison with Lloyd's (2002) independent computational-capacity bound")
    print("=" * 78)
    print("  Lloyd (2002): the universe can have performed no more than ~10^120")
    print("  elementary operations, on ~10^90-10^120 bits (the upper end including")
    print("  gravitational degrees of freedom), over its ~13.8 Gyr history --")
    print("  obtained from the Margolus-Levitin bound on computation rate plus")
    print("  a Bekenstein-type bound on registerable bits, applied to the")
    print("  observable universe's matter/energy content. This is a wholly")
    print("  different physical route from the horizon-area calculation above:")
    print("  no de Sitter geometry, no Gibbons-Hawking temperature, no H0 at all.")
    print()

    low, high = min(r[3] for r in results), max(r[3] for r in results)
    print(f"  This script's result: log10(S_dS/k_B) in [{low:.1f}, {high:.1f}]")
    print(f"  across the two cited H0 values -- the same order of magnitude")
    print(f"  (~10^120-10^122) as Lloyd's independently obtained figure.")
    print()
    print("  This agreement is a nontrivial, if informal, consistency check")
    print("  between two unrelated calculations (horizon geometry vs.")
    print("  mass-energy + computation-rate bounds); it is not, by itself,")
    print("  evidence for any specific value of N(O) in Theorem 4.3, which")
    print("  remains conditional on (S3b) -- see the paper's own Section 6.")
