"""
3D Manifold Visualization at C = 3.17 (Simplified version)
Tests periodic orbit computation before full manifold calculation.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from code_manifold2_python import compute_periodic_family, compute_energy_matched_orbit

print("Setting up parameters...")
C_jacobi = 3.17
E = -C_jacobi / 2

G = 1
m = 0.025
M = 1 - m
R = 1

xbary = (M * 0 + m * R) / (m + M)
xsecondary = R - xbary
xprimary = -xbary

omega = np.sqrt(G * M / (xsecondary * (xsecondary - xprimary) ** 2))
Gm = G * m
GM = G * M
Vbarrier = -1.625

print("Computing periodic family (this will take ~2-3 minutes)...")
Pinit = np.array([0.77, 0, 0, 0.1551])
Nsteps = 2**7
T = 2.65

import time
t_start = time.time()

Ps, Ts, CJs, initx, all_xs = compute_periodic_family(
    Pinit.copy(), T, 1e-4, 2000, Nsteps, Gm, GM, xsecondary, xprimary, omega
)

t_elapsed = time.time() - t_start
print(f"✓ Found {len(all_xs)} periodic orbits in {t_elapsed:.1f} seconds")

print("Computing energy-matched orbit...")
Popt, Topt, xsorb = compute_energy_matched_orbit(
    Ps, Ts, CJs, Vbarrier, Nsteps, Gm, GM, xsecondary, xprimary, omega
)
print(f"✓ Energy-matched orbit computed: period={Topt:.4f}, shape={xsorb.shape}")

print("\nTo complete the visualization, run plot_manifolds_3d.py")
print("(This script is much slower on first run due to manifold computation)")
