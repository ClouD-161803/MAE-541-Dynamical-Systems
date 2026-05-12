import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from code_manifold2_python import (
    compute_periodic_family, compute_energy_matched_orbit,
    compute_manifold_tubes, dPdt_rk8, scatter_point, plot_trajectory
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

fig, ax = plt.subplots(figsize=(14, 14))
ax.set_xlim(tuple(np.array([-1, 1]) * 0.25 + L1[0]))
ax.set_ylim(tuple(np.array([-1, 1]) * 0.25))
ax.set_aspect('equal')

scatter_point(ax, xprimary, 0, size=300, color=[0.75, 0.75, 0.75])
scatter_point(ax, xsecondary, 0, size=100, color=[0.75, 0.75, 0.75])

scatter_point(ax, L1_x, 0, size=100, color=[0, 0, 0])

ax.tick_params(labelsize=28)
ax.set_xlabel(r'x', fontsize=32)
ax.set_ylabel(r'y', fontsize=32)
ax.set_title('', fontsize=36)

ax.grid(True, alpha=0.3)

stable_lines = []
unstable_lines = []
stable_particles = []
unstable_particles = []

for stable_tubes in manifold_data['stable_tubes']:
    for xs in stable_tubes:
        line, = ax.plot([], [], color="#238443", linewidth=3, alpha=0.7)
        line.set_data([], [])
        stable_lines.append((line, xs))
        particle, = ax.plot([], [], 'o', color="#238443", markersize=12, alpha=0.9)
        stable_particles.append((particle, xs))

for unstable_tubes in manifold_data['unstable_tubes']:
    for xs in unstable_tubes:
        line, = ax.plot([], [], color="#c43c1a", linewidth=3, alpha=0.7)
        line.set_data([], [])
        unstable_lines.append((line, xs))
        particle, = ax.plot([], [], 'o', color="#c43c1a", markersize=12, alpha=0.9)
        unstable_particles.append((particle, xs))

text_display = ax.text(0.5, 0.95, '', transform=ax.transAxes,
                       fontsize=28, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       fontweight='bold')

n_steps_per_manifold = 150
pause_frames = 30
total_stable_frames = n_steps_per_manifold + pause_frames
total_frames = total_stable_frames + pause_frames + n_steps_per_manifold

def animate(frame):
    all_artists = []

    if frame < n_steps_per_manifold:
        # Animate stable manifold
        progress = frame / n_steps_per_manifold
        for line, xs in stable_lines:
            n_points = int(xs.shape[1] * progress)
            if n_points > 0:
                line.set_data(xs[0, :n_points], xs[1, :n_points])
                all_artists.append(line)

        for particle, xs in stable_particles:
            n_points = int(xs.shape[1] * progress)
            if n_points > 0:
                particle.set_data([xs[0, n_points-1]], [xs[1, n_points-1]])
                all_artists.append(particle)

        text_display.set_text(f'Stable Manifold\nProgress: {progress*100:.1f}%')
        all_artists.append(text_display)

    elif frame < total_stable_frames:
        # Pause after stable
        for line, xs in stable_lines:
            line.set_data(xs[0, :], xs[1, :])
            all_artists.append(line)

        for particle, xs in stable_particles:
            particle.set_data([xs[0, -1]], [xs[1, -1]])
            all_artists.append(particle)

        text_display.set_text('Stable Manifold\n(Complete)')
        all_artists.append(text_display)

    elif frame < total_stable_frames + pause_frames:
        # Pause before unstable
        for line, xs in stable_lines:
            line.set_data(xs[0, :], xs[1, :])
            all_artists.append(line)

        for particle, xs in stable_particles:
            particle.set_data([xs[0, -1]], [xs[1, -1]])
            all_artists.append(particle)

        text_display.set_text('Preparing Unstable Manifold...')
        all_artists.append(text_display)

    else:
        # Animate unstable manifold
        progress = (frame - total_stable_frames - pause_frames) / n_steps_per_manifold
        for line, xs in stable_lines:
            line.set_data(xs[0, :], xs[1, :])
            all_artists.append(line)

        for particle, xs in stable_particles:
            particle.set_data([xs[0, -1]], [xs[1, -1]])
            all_artists.append(particle)

        for line, xs in unstable_lines:
            n_points = int(xs.shape[1] * progress)
            if n_points > 0:
                line.set_data(xs[0, :n_points], xs[1, :n_points])
                all_artists.append(line)

        for particle, xs in unstable_particles:
            n_points = int(xs.shape[1] * progress)
            if n_points > 0:
                particle.set_data([xs[0, n_points-1]], [xs[1, n_points-1]])
                all_artists.append(particle)

        text_display.set_text(f'Unstable Manifold\nProgress: {progress*100:.1f}%')
        all_artists.append(text_display)

    return all_artists

anim = animation.FuncAnimation(
    fig, animate, frames=total_frames, interval=50, blit=True, repeat=False
)

output_path = r'c:\Users\cvest\Claudio\Princeton\4th Year\MAE 541\Final Project\Figures\manifold_propagation.mp4'
writer = animation.FFMpegWriter(fps=30, bitrate=1800)
anim.save(output_path, writer=writer, dpi=100)
print(f"Animation saved to {output_path}")

plt.close()
