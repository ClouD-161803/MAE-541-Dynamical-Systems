import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from code_manifold2_python import (
    compute_periodic_orbit,
    dPdt_rk8,
    find_lagrange_point,
)


STATE_INDEX = {
    "x": 0,
    "y": 1,
    "vx": 2,
    "vy": 3,
}

PROJECTION = ("x", "y", "vy")


def cr3bp_parameters():
    G = 1
    m = 0.025
    M = 1 - m
    R = 1

    xbary = (M * 0 + m * R) / (m + M)
    xsecondary = R - xbary
    xprimary = -xbary
    omega = np.sqrt(G * M / (xsecondary * (xsecondary - xprimary) ** 2))

    return {
        "G": G,
        "m": m,
        "M": M,
        "R": R,
        "Gm": G * m,
        "GM": G * M,
        "xsecondary": xsecondary,
        "xprimary": xprimary,
        "omega": omega,
    }


def l1_position(params):
    y = 0
    Gm = params["Gm"]
    GM = params["GM"]
    omega = params["omega"]
    xsecondary = params["xsecondary"]
    xprimary = params["xprimary"]

    def force_balance(x):
        return (
            -x * omega**2
            - Gm * -0.5 * ((x - xsecondary) ** 2 + y**2) ** (-3 / 2) * 2 * (x - xsecondary)
            - GM * -0.5 * ((x - xprimary) ** 2 + y**2) ** (-3 / 2) * 2 * (x - xprimary)
        )

    return find_lagrange_point(0.78, force_balance)


def propagate_state(P0, T, Nsteps, params):
    ts = np.linspace(0, T, Nsteps)
    dt = ts[1] - ts[0]
    P = P0.copy()

    for t in ts[1:]:
        dPdt = dPdt_rk8(
            t,
            P,
            params["Gm"],
            params["GM"],
            params["xsecondary"],
            params["xprimary"],
            params["omega"],
            dt,
        )
        P = P + dPdt * dt

    return P


def monodromy_matrix(P0, T, Nsteps, params, fd_step=1e-7):
    xT = propagate_state(P0, T, Nsteps, params)
    columns = []

    for i in range(4):
        dP0 = np.zeros(4)
        dP0[i] = fd_step
        xT_perturbed = propagate_state(P0 + dP0, T, Nsteps, params)
        columns.append((xT_perturbed - xT) / fd_step)

    return np.column_stack(columns)


def real_hyperbolic_eigendirections(M):
    eigenvalues, eigenvectors = np.linalg.eig(M)
    real_candidates = np.where(np.abs(np.imag(eigenvalues)) < 1e-6)[0]

    if len(real_candidates) < 2:
        raise ValueError("Could not isolate real stable and unstable eigendirections.")

    magnitudes = np.abs(np.real(eigenvalues[real_candidates]))
    stable_index = real_candidates[np.argmin(magnitudes)]
    unstable_index = real_candidates[np.argmax(magnitudes)]

    stable = np.real(eigenvectors[:, stable_index])
    unstable = np.real(eigenvectors[:, unstable_index])

    stable = stable / np.linalg.norm(stable)
    unstable = unstable / np.linalg.norm(unstable)

    return stable, unstable


def local_eigendirections(xsorb, Torb, Nsteps, params, phase_count, visible_fraction):
    max_index = xsorb.shape[1] - 2
    stop_index = int(np.floor(max_index * visible_fraction))
    indices = np.unique(np.round(np.linspace(0, stop_index, phase_count)).astype(int))

    base_states = xsorb[:, indices].T
    stable_dirs = []
    unstable_dirs = []

    previous_stable = None
    previous_unstable = None

    for state in base_states:
        M = monodromy_matrix(state, Torb, Nsteps, params)
        stable, unstable = real_hyperbolic_eigendirections(M)

        if previous_stable is not None and np.dot(stable, previous_stable) < 0:
            stable = -stable
        if previous_unstable is not None and np.dot(unstable, previous_unstable) < 0:
            unstable = -unstable

        stable_dirs.append(stable)
        unstable_dirs.append(unstable)
        previous_stable = stable
        previous_unstable = unstable

    return base_states, np.array(stable_dirs), np.array(unstable_dirs)


def project(states, projection=PROJECTION):
    return tuple(states[..., STATE_INDEX[name]] for name in projection)


def eigendirection_surface(base_states, directions, local_width, offset_count):
    offsets = np.linspace(-local_width, local_width, offset_count)
    surface_states = base_states[None, :, :] + offsets[:, None, None] * directions[None, :, :]
    surfacecolor = np.tile(np.abs(offsets / local_width)[:, None], (1, base_states.shape[0]))
    return (*project(surface_states), surfacecolor)


def orbit_radius(xsorb, l1_x):
    xy_offsets = xsorb[:2, :].T - np.array([l1_x, 0])
    return np.max(np.linalg.norm(xy_offsets, axis=1))


def cross_section_patch(base_state, stable_dir, unstable_dir, local_width, grid_count=9):
    u = np.linspace(-local_width, local_width, grid_count)
    v = np.linspace(-local_width, local_width, grid_count)
    U, V = np.meshgrid(u, v)
    states = base_state[None, None, :] + U[..., None] * stable_dir + V[..., None] * unstable_dir
    surfacecolor = np.zeros_like(U)
    return (*project(states), surfacecolor)


def line_along_direction(base_state, direction, local_width, count=25):
    offsets = np.linspace(-local_width, local_width, count)
    states = base_state[None, :] + offsets[:, None] * direction[None, :]
    return project(states)


def plot_config(args):
    return {
        "toImageButtonOptions": {
            "format": "png",
            "filename": "l1_eigendirection_surfaces",
            "height": args.height,
            "width": args.width,
            "scale": args.export_scale,
        }
    }


def add_surface(fig, x, y, z, surfacecolor, colorscale, opacity):
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=surfacecolor,
            colorscale=colorscale,
            cmin=0,
            cmax=1,
            opacity=opacity,
            showscale=False,
            contours=dict(
                x=dict(show=True, width=1, color="rgba(20,20,20,0.12)"),
                y=dict(show=True, width=1, color="rgba(20,20,20,0.12)"),
                z=dict(show=True, width=1, color="rgba(20,20,20,0.10)"),
            ),
            lighting=dict(
                ambient=0.42,
                diffuse=0.78,
                fresnel=0.15,
                specular=0.12,
                roughness=0.72,
            ),
            lightposition=dict(x=100, y=120, z=800),
            hoverinfo="skip",
        )
    )


def build_figure(args):
    params = cr3bp_parameters()

    Pinit = np.array([args.orbit_x0, 0, 0, args.orbit_vy0])
    P_periodic, T_periodic, xsorb = compute_periodic_orbit(
        Pinit,
        args.period_guess,
        args.steps,
        params["Gm"],
        params["GM"],
        params["xsecondary"],
        params["xprimary"],
        params["omega"],
    )

    base_states, stable_dirs, unstable_dirs = local_eigendirections(
        xsorb,
        T_periodic,
        args.steps,
        params,
        args.phase_count,
        args.visible_fraction,
    )
    L1_x = l1_position(params)
    local_width = args.local_width
    if local_width is None:
        local_width = args.width_fraction * orbit_radius(xsorb, L1_x)

    stable_x, stable_y, stable_z, stable_color = eigendirection_surface(
        base_states,
        stable_dirs,
        local_width,
        args.offset_count,
    )
    unstable_x, unstable_y, unstable_z, unstable_color = eigendirection_surface(
        base_states,
        unstable_dirs,
        local_width,
        args.offset_count,
    )

    fig = go.Figure()
    unstable_colorscale = [
        [0.00, "#fff176"],
        [0.38, "#fdae32"],
        [1.00, "#c43c1a"],
    ]
    stable_colorscale = [
        [0.00, "#fff176"],
        [0.42, "#b9df57"],
        [1.00, "#238443"],
    ]
    add_surface(fig, unstable_x, unstable_y, unstable_z, unstable_color, unstable_colorscale, 0.78)
    add_surface(fig, stable_x, stable_y, stable_z, stable_color, stable_colorscale, 0.72)

    orbit_x, orbit_y, orbit_z = project(xsorb.T)
    fig.add_trace(
        go.Scatter3d(
            x=orbit_x,
            y=orbit_y,
            z=orbit_z,
            mode="lines",
            line=dict(color="black", width=8),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[L1_x],
            y=[0],
            z=[0],
            mode="markers",
            marker=dict(color="black", size=4),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    cut_state = base_states[-1]
    cut_stable = stable_dirs[-1]
    cut_unstable = unstable_dirs[-1]
    x, y, z, color = cross_section_patch(cut_state, cut_stable, cut_unstable, local_width)
    add_surface(fig, x, y, z, color, [[0, "rgba(230,230,230,0.65)"], [1, "rgba(230,230,230,0.65)"]], 0.42)

    for direction, color in [(cut_stable, "#76b82a"), (cut_unstable, "#d95f02")]:
        x, y, z = line_along_direction(cut_state, direction, local_width * 1.15)
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color=color, width=5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        autosize=False,
        width=args.width,
        height=args.height,
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="", showgrid=False, showticklabels=False, showbackground=False, zeroline=False),
            yaxis=dict(title="", showgrid=False, showticklabels=False, showbackground=False, zeroline=False),
            zaxis=dict(title="", showgrid=False, showticklabels=False, showbackground=False, zeroline=False),
            camera=dict(eye=dict(x=1.55, y=-1.8, z=1.08)),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="white",
    )

    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--steps", type=int, default=2**8)
    parser.add_argument("--phase-count", type=int, default=72)
    parser.add_argument("--offset-count", type=int, default=23)
    parser.add_argument("--visible-fraction", type=float, default=0.75)
    parser.add_argument("--local-width", type=float, default=None)
    parser.add_argument("--width-fraction", type=float, default=0.9)
    parser.add_argument("--orbit-x0", type=float, default=0.7712)
    parser.add_argument("--orbit-vy0", type=float, default=0.143)
    parser.add_argument("--period-guess", type=float, default=2.65)
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--export-scale", type=int, default=4)
    parser.add_argument("--html", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    fig = build_figure(args)

    output_path = args.html
    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "Figures" / "l1_eigendirection_surfaces.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn", config=plot_config(args))
    print(f"HTML saved to {output_path}")

    if not args.no_show:
        fig.show(config=plot_config(args))


if __name__ == "__main__":
    main()
