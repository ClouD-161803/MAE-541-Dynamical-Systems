import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from code_manifold2_python import (
    compute_periodic_family, compute_energy_matched_orbit,
    compute_manifold_tubes, scatter_point
)

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

Pinit = np.array([0.77, 0, 0, 0.1551])
Nsteps = 2**8
T = 2.65

Ps, Ts, CJs, initx, all_xs = compute_periodic_family(
    Pinit.copy(), T, 1e-4, 2000, Nsteps, Gm, GM, xsecondary, xprimary, omega
)

Popt, Topt, xsorb = compute_energy_matched_orbit(
    Ps, Ts, CJs, Vbarrier, Nsteps, Gm, GM, xsecondary, xprimary, omega
)

manifold_data = compute_manifold_tubes(
    xsorb, Topt, Nsteps, Gm, GM, xsecondary, xprimary, omega, xsecondary
)

L1_x = 0.78
L1 = [L1_x, 0]

T_max_manifold = 5
color_stable = "#238443"
color_unstable = "#ff020289"

progress_levels_stable = [0.001, 0.5, 1.0]
progress_levels_unstable = [0.001, 0.5, 1.0]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('', fontsize=1)

for idx, (ax, progress) in enumerate(zip(axes[0, :], progress_levels_stable)):
    ax.set_xlim(tuple(np.array([-1, 1]) * 0.25 + L1[0]))
    ax.set_ylim(tuple(np.array([-1, 1]) * 0.25))
    ax.set_aspect('equal')

    scatter_point(ax, xprimary, 0, size=300, color=[0.75, 0.75, 0.75])
    scatter_point(ax, xsecondary, 0, size=100, color=[0.75, 0.75, 0.75])
    scatter_point(ax, L1_x, 0, size=100, color=[0, 0, 0])

    ax.tick_params(labelsize=24)
    ax.grid(True, alpha=0.3)

    if idx == 0:
        ax.set_ylabel(r'$y \ [\mathrm{DU}]$', fontsize=28)

    T_current = progress * T_max_manifold
    ax.set_title(rf'$T = {T_current:.2f} \ [\mathrm{{TU}}]$', fontsize=28, fontweight='bold', pad=15)

    for stable_tubes in manifold_data['stable_tubes']:
        for xs in stable_tubes:
            n_points = int(xs.shape[1] * progress)
            if n_points > 0:
                ax.plot(xs[0, :n_points], xs[1, :n_points],
                       color=color_stable, linewidth=2.5, alpha=0.7)
                ax.plot([xs[0, n_points-1]], [xs[1, n_points-1]],
                       'o', color=color_stable, markersize=12, alpha=0.9)

for idx, (ax, progress) in enumerate(zip(axes[1, :], progress_levels_unstable)):
    ax.set_xlim(tuple(np.array([-1, 1]) * 0.25 + L1[0]))
    ax.set_ylim(tuple(np.array([-1, 1]) * 0.25))
    ax.set_aspect('equal')

    scatter_point(ax, xprimary, 0, size=300, color=[0.75, 0.75, 0.75])
    scatter_point(ax, xsecondary, 0, size=100, color=[0.75, 0.75, 0.75])
    scatter_point(ax, L1_x, 0, size=100, color=[0, 0, 0])

    ax.tick_params(labelsize=24)
    ax.grid(True, alpha=0.3)

    ax.set_xlabel(r'$x \ [\mathrm{DU}]$', fontsize=28)
    if idx == 0:
        ax.set_ylabel(r'$y \ [\mathrm{DU}]$', fontsize=28)

    for unstable_tubes in manifold_data['unstable_tubes']:
        for xs in unstable_tubes:
            n_points = int(xs.shape[1] * progress)
            if n_points > 0:
                ax.plot(xs[0, :n_points], xs[1, :n_points],
                       color=color_unstable, linewidth=2.5, alpha=0.7)
                ax.plot([xs[0, n_points-1]], [xs[1, n_points-1]],
                       'o', color=color_unstable, markersize=12, alpha=0.9)

legend_elements = [
    Line2D([0], [0], color=color_stable, linewidth=3, label='Stable Manifold'),
    Line2D([0], [0], color=color_unstable, linewidth=3, label='Unstable Manifold')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.01),
          ncol=2, fontsize=26, frameon=True, fancybox=True, shadow=True)

plt.tight_layout(rect=(0, 0, 1, 0.96))

output_path = r'c:\Users\cvest\Claudio\Princeton\4th Year\MAE 541\Final Project\Figures\manifold_progression_grid.pdf'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {output_path}")

plt.close()
