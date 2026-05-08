"""
3D Manifold Visualization at C = 3.17 (Hybrid: Surfaces + Trajectories)
Plots manifold surfaces in (x, y, vy) space with thin orbit path lines.
Uses exact colormaps from plot_l1_eigendirection_surfaces.py
"""

import sys
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from skimage import measure

sys.path.insert(0, str(Path(__file__).parent))
from code_manifold2_python import (
    compute_periodic_family,
    compute_energy_matched_orbit,
    compute_manifold_tubes
)

print("Setting up parameters...")
C_jacobi = 3.17
E = -C_jacobi / 2

mu = 0.025
G = 1
m = mu
M = 1 - mu
R = 1

xbary = (M * 0 + m * R) / (m + M)
xsecondary = R - xbary
xprimary = -xbary

omega = np.sqrt(G * M / (xsecondary * (xsecondary - xprimary) ** 2))
T_orbital = 2 * np.pi / omega
Gm = G * m
GM = G * M

print("Computing barrier height at L1...")
# Find L1 position
y = 0
def force_balance_l1(x):
    return -x * omega**2 - Gm * -0.5 * ((x - xsecondary)**2 + y**2)**(-3/2) * 2 * (x - xsecondary) - GM * -0.5 * ((x - xprimary)**2 + y**2)**(-3/2) * 2 * (x - xprimary)

from scipy.optimize import brentq
L1_x = brentq(force_balance_l1, xprimary + 0.01, xsecondary - 0.01)
print(f"L1 position: {L1_x:.4f}")

Vbarrier = -1.625

print("Creating effective potential surface...")
resolution = 250
x_range = np.linspace(-1.5, 1.5, resolution)
y_range = np.linspace(-1.5, 1.5, resolution)
X, Y = np.meshgrid(x_range, y_range)

r1 = np.sqrt((X - xprimary) ** 2 + Y ** 2)
r2 = np.sqrt((X - xsecondary) ** 2 + Y ** 2)

V_grav = -Gm / r2 - GM / r1
V_cent = -0.5 * omega ** 2 * (X ** 2 + Y ** 2)
U_eff = V_grav + V_cent

z_min = -3.0
U_eff_clipped = np.clip(U_eff, z_min, np.max(U_eff))

fig = go.Figure()

print("Adding potential surface...")
fig.add_trace(go.Surface(
    z=U_eff_clipped,
    x=X,
    y=Y,
    colorscale='Magma_r',
    cmin=z_min,
    cmax=-1.5,
    opacity=0.7,
    lighting=dict(ambient=0.4, diffuse=0.8, fresnel=0.2, specular=0.1, roughness=1.0),
    showscale=False,
    hoverinfo='skip'
))

print("Adding Hill curve...")
contours = measure.find_contours(U_eff_clipped, E)

for contour in contours:
    if len(contour) > 10:
        i_indices = contour[:, 0]
        j_indices = contour[:, 1]

        contour_x = np.interp(j_indices, np.arange(len(x_range)), x_range)
        contour_y = np.interp(i_indices, np.arange(len(y_range)), y_range)
        contour_z = np.zeros_like(contour_x)

        fig.add_trace(go.Scatter3d(
            x=contour_x,
            y=contour_y,
            z=contour_z,
            mode='lines',
            line=dict(color='rgba(255, 165, 0, 0.7)', width=3),
            name='Hill Curve',
            showlegend=False,
            hoverinfo='skip'
        ))

print("Adding primary and secondary bodies...")
fig.add_trace(go.Scatter3d(
    x=[xprimary],
    y=[0],
    z=[0],
    mode='markers',
    marker=dict(size=12, color='lightblue', symbol='circle'),
    name='Primary',
    showlegend=False,
    hoverinfo='skip'
))

fig.add_trace(go.Scatter3d(
    x=[xsecondary],
    y=[0],
    z=[0],
    mode='markers',
    marker=dict(size=4, color='white', symbol='circle'),
    name='Secondary',
    showlegend=False,
    hoverinfo='skip'
))

print("Computing periodic family...")
Pinit = np.array([0.77, 0, 0, 0.1551])
Nsteps = 2**7
T = 2.65

Ps, Ts, CJs, initx, all_xs = compute_periodic_family(
    Pinit.copy(), T, 1e-4, 2000, Nsteps, Gm, GM, xsecondary, xprimary, omega
)

print(f"Found {len(all_xs)} periodic orbits")

print("Computing energy-matched orbit...")
Popt, Topt, xsorb = compute_energy_matched_orbit(
    Ps, Ts, CJs, Vbarrier, Nsteps, Gm, GM, xsecondary, xprimary, omega
)

print(f"Energy-matched orbit period: {Topt:.4f}")

print("Adding Lyapunov orbit (thin green line)...")
fig.add_trace(go.Scatter3d(
    x=xsorb[0, :],
    y=xsorb[1, :],
    z=xsorb[3, :],
    mode='lines',
    line=dict(color='rgba(35, 132, 67, 0.7)', width=1.5),
    name='Lyapunov Orbit',
    showlegend=True,
    hoverinfo='skip'
))

print("Computing manifold tubes (this takes ~5-10 minutes)...")
manifold_data = compute_manifold_tubes(
    xsorb, Topt, Nsteps, Gm, GM, xsecondary, xprimary, omega, xsecondary
)

print(f"Computed {len(manifold_data['base_points'])} manifold points")

print("Adding manifold trajectories with color gradients...")

# Unstable colorscale: yellow → orange → red
unstable_colors_hex = [
    [0.0, "#eeb600"],
    [0.5, "#ff9d00"],
    [1.0, "#c43c1a"],
]
# Stable colorscale: yellow → green/yellow → green
stable_colors_hex = [
    [0.0, "#3af700"],
    [0.5, "#03BE03"],
    [1.0, "#238443"],
]

print("Adding unstable manifold trajectories...")
for idx, unstable_tubes in enumerate(manifold_data['unstable_tubes']):
    for tube_idx, tube in enumerate(unstable_tubes):
        if tube.size > 0 and np.sum(~np.isnan(tube[0, :])) > 5:
            # Create color gradient along trajectory
            valid_idx = ~np.isnan(tube[0, :])
            valid_count = np.sum(valid_idx)

            if valid_count > 1:
                # Interpolate colors from colorscale
                color_positions = np.linspace(0, 1, valid_count)

                # Use RGB colors with gradient
                x_val = tube[0, valid_idx]
                y_val = tube[1, valid_idx]
                z_val = tube[3, valid_idx]

                # Plot with marker size proportional to position along trajectory
                fig.add_trace(go.Scatter3d(
                    x=x_val,
                    y=y_val,
                    z=z_val,
                    mode='lines',
                    line=dict(
                        color=color_positions,
                        colorscale=unstable_colors_hex,
                        width=3,
                        showscale=False
                    ),
                    name='Unstable Manifold' if (idx == 0 and tube_idx == 0) else None,
                    showlegend=(idx == 0 and tube_idx == 0),
                    hoverinfo='skip'
                ))

print("Adding stable manifold trajectories...")
for idx, stable_tubes in enumerate(manifold_data['stable_tubes']):
    for tube_idx, tube in enumerate(stable_tubes):
        if tube.size > 0 and np.sum(~np.isnan(tube[0, :])) > 5:
            valid_idx = ~np.isnan(tube[0, :])
            valid_count = np.sum(valid_idx)

            if valid_count > 1:
                color_positions = np.linspace(0, 1, valid_count)

                x_val = tube[0, valid_idx]
                y_val = tube[1, valid_idx]
                z_val = tube[3, valid_idx]

                fig.add_trace(go.Scatter3d(
                    x=x_val,
                    y=y_val,
                    z=z_val,
                    mode='lines',
                    line=dict(
                        color=color_positions,
                        colorscale=stable_colors_hex,
                        width=3,
                        showscale=False
                    ),
                    name='Stable Manifold' if (idx == 0 and tube_idx == 0) else None,
                    showlegend=(idx == 0 and tube_idx == 0),
                    hoverinfo='skip'
                ))

print("Configuring layout...")
fig.update_layout(
    autosize=False,
    width=1400,
    height=1400,
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis=dict(title='', showgrid=False, showticklabels=False, showbackground=False),
        yaxis=dict(title='', showgrid=False, showticklabels=False, showbackground=False),
        zaxis=dict(title='', showgrid=False, showticklabels=False, showbackground=False),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2), up=dict(x=0, y=0, z=1)),
        bgcolor='rgba(0,0,0,0)'
    ),
    showlegend=True,
    legend=dict(
        x=0.7, y=0.95,
        bgcolor='rgba(0, 0, 0, 0.5)',
        font=dict(color='white', size=12)
    ),
    paper_bgcolor='white'
)

script_dir = Path(__file__).parent
figures_dir = script_dir.parent / 'Figures'
figures_dir.mkdir(parents=True, exist_ok=True)

print("Saving visualization...")
output_path = figures_dir / f'manifolds_3d_C_{C_jacobi:.2f}.html'
fig.write_html(str(output_path))
print(f"✓ 3D manifold plot saved to {output_path}")

print("Opening plot in browser...")
fig.show()
