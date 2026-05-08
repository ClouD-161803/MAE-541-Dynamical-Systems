import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

"""
Three-Layer Architecture for Orbit and Manifold Computation:

LAYER 1 - Pure Computation (no plotting, returns data):
  - compute_reference_trajectory()
  - compute_periodic_orbit()
  - compute_periodic_family()
  - compute_energy_matched_orbit()
  - compute_manifold_tubes()

LAYER 2 - 2D Plotting Wrappers (call Layer 1 + plot):
  - compute_initial_reference_trajectory(ax)
  - find_periodic_orbit(ax)
  - continue_periodic_orbits(ax)
  - find_energy_matched_orbit(ax)
  - compute_manifolds(ax)
  - plot_manifold_tubes(ax, manifold_data)

LAYER 3 - Plotting Utilities (basic matplotlib helpers):
  - scatter_point(ax, ...)
  - plot_trajectory(ax, ...)

Usage for 3D or custom plotting:
  1. Call Layer 1 computation functions to get data
  2. Create your own plotting functions
  3. Pass the returned data to your plotter

Example:
  xs = compute_reference_trajectory(...)
  manifold_data = compute_manifold_tubes(...)
  # Now use xs and manifold_data with your 3D plotting tool
"""


def Hamiltonian(t, P):
    x = P[0]
    y = P[1]

    G = 1
    m = 0.025
    M = 1 - m
    R = 1

    xbary = (M * 0 + m * R) / (m + M)
    xsecondary = R - xbary
    xprimary = -xbary

    omega = np.sqrt(G * M / (xsecondary * (xsecondary - xprimary) ** 2))

    dHdx = -1 * x * omega ** 2 - G * m * -0.5 * ((x - xsecondary) ** 2 + y ** 2) ** (-3 / 2) * 2 * (x - xsecondary) - G * M * -0.5 * ((x - xprimary) ** 2 + y ** 2) ** (-3 / 2) * 2 * (x - xprimary)
    dHdy = -1 * y * omega ** 2 - G * m * -0.5 * ((x - xsecondary) ** 2 + y ** 2) ** (-3 / 2) * 2 * y - G * M * -0.5 * ((x - xprimary) ** 2 + y ** 2) ** (-3 / 2) * 2 * y

    dPdt = np.array([-dHdy, dHdx])
    return dPdt


def dPdt_euler(t, P, Gm, GM, xsecondary, xprimary, omega):
    x = P[0]
    y = P[1]
    xp = P[2]
    yp = P[3]

    dPdt = np.array([xp, yp, 2 * omega * yp + omega ** 2 * x - Gm * (x - xsecondary) / ((x - xsecondary) ** 2 + y ** 2) ** (3/2) - GM * (x - xprimary) / ((x - xprimary) ** 2 + y ** 2) ** (3/2), -2 * omega * xp + omega ** 2 * y - Gm * y / ((x - xsecondary) ** 2 + y ** 2) ** (3/2) - GM * y / ((x - xprimary) ** 2 + y ** 2) ** (3/2)])
    return dPdt


def kfromP(t, P1, Gm, GM, xsecondary, xprimary, omega):
    k = np.zeros(4,)
    k[0] = P1[2]
    k[1] = P1[3]
    k[2] = 2 * omega * P1[3] + omega ** 2 * P1[0] - Gm * (P1[0] - xsecondary) / ((P1[0] - xsecondary) ** 2 + P1[1] ** 2) ** (3/2) - GM * (P1[0] - xprimary) / ((P1[0] - xprimary) ** 2 + P1[1] ** 2) ** (3/2)
    k[3] = -2 * omega * P1[2] + omega ** 2 * P1[1] - Gm * P1[1] / ((P1[0] - xsecondary) ** 2 + P1[1] ** 2) ** (3/2) - GM * P1[1] / ((P1[0] - xprimary) ** 2 + P1[1] ** 2) ** (3/2)
    return k


def dPdt_rk8(t, P, Gm, GM, xsecondary, xprimary, omega, dt):
    P0 = P
    k1 = kfromP(t, P0, Gm, GM, xsecondary, xprimary, omega)

    P1 = P0 + k1 * 4 / 27 * dt
    k2 = kfromP(t, P1, Gm, GM, xsecondary, xprimary, omega)

    P1 = P0 + (dt / 18) * (k1 + 3 * k2)
    k3 = kfromP(t, P1, Gm, GM, xsecondary, xprimary, omega)

    P1 = P0 + (dt / 12) * (k1 + 3 * k3)
    k4 = kfromP(t, P1, Gm, GM, xsecondary, xprimary, omega)

    P1 = P0 + (dt / 8) * (k1 + 3 * k4)
    k5 = kfromP(t, P1, Gm, GM, xsecondary, xprimary, omega)

    P1 = P0 + (dt / 54) * (13 * k1 - 27 * k3 + 42 * k4 + 8 * k5)
    k6 = kfromP(t, P1, Gm, GM, xsecondary, xprimary, omega)

    P1 = P0 + (dt / 4320) * (389 * k1 - 54 * k3 + 966 * k4 - 824 * k5 + 243 * k6)
    k7 = kfromP(t, P1, Gm, GM, xsecondary, xprimary, omega)

    P1 = P0 + (dt / 20) * (-231 * k1 + 81 * k3 - 1164 * k4 + 656 * k5 - 122 * k6 + 800 * k7)
    k8 = kfromP(t, P1, Gm, GM, xsecondary, xprimary, omega)

    P1 = P0 + (dt / 288) * (-127 * k1 + 18 * k3 - 678 * k4 + 456 * k5 - 9 * k6 + 576 * k7 + 4 * k8)
    k9 = kfromP(t, P1, Gm, GM, xsecondary, xprimary, omega)

    P1 = P0 + (dt / 820) * (1481 * k1 - 81 * k3 + 7104 * k4 - 3376 * k5 + 72 * k6 - 5040 * k7 - 60 * k8 + 720 * k9)
    k10 = kfromP(t, P1, Gm, GM, xsecondary, xprimary, omega)

    dPdt = (41 * k1 + 27 * k4 + 272 * k5 + 27 * k6 + 216 * k7 + 216 * k9 + 41 * k10) / 840
    return dPdt


def scatter_point(ax, x, y, size=100, color=None, marker='.'):
    if color is None:
        color = [0, 0, 0]
    ax.scatter(x, y, s=size, c=[color], marker=marker)


def plot_trajectory(ax, xs, color, linewidth=1, alpha=1.0):
    ax.plot(xs[0, :], xs[1, :], color=color, linewidth=linewidth, alpha=alpha)


def integrate_rk8_trajectory(P, T, Nsteps, Gm, GM, xsecondary, xprimary, omega, return_full=False):
    ts = np.linspace(0, T, Nsteps)
    dt = ts[1] - ts[0]
    xs = np.nan * np.ones((2, len(ts)))
    xs[:, 0] = P[0:2]
    P_curr = P.copy()

    for i, t in enumerate(ts[1:], 1):
        dPdt = dPdt_rk8(t, P_curr, Gm, GM, xsecondary, xprimary, omega, dt)
        P_curr = P_curr + dPdt * dt
        xs[:, i] = P_curr[0:2]

    if return_full:
        return xs, P_curr
    return xs


def integrate_rk8_full_state(P, T, Nsteps, Gm, GM, xsecondary, xprimary, omega):
    ts = np.linspace(0, T, Nsteps)
    dt = ts[1] - ts[0]
    xs = np.nan * np.ones((4, len(ts)))
    xs[:, 0] = P
    P_curr = P.copy()

    for i, t in enumerate(ts[1:], 1):
        dPdt = dPdt_rk8(t, P_curr, Gm, GM, xsecondary, xprimary, omega, dt)
        P_curr = P_curr + dPdt * dt
        xs[:, i] = P_curr

    return xs, P_curr


def integrate_verlet_inertial(X, V, T, dt, xsecondary, xprimary, xsecondarys, xprimarys, G, m, M, omega, ts):
    rotangle = omega * ts
    A = -G * m * (X - xsecondarys[:, 0]) / np.linalg.norm(X - xsecondarys[:, 0])**3 - G * M * (X - xprimarys[:, 0]) / np.linalg.norm(X - xprimarys[:, 0])**3

    xs = np.empty((2, len(ts)))
    xs[:, 0] = X

    for i, t in enumerate(ts[1:], 1):
        xsecondarycurr = xsecondarys[:, i]
        xprimarycurr = xprimarys[:, i]

        X = X + V * dt + 0.5 * A * dt**2
        Aold = A
        A = -G * m * (X - xsecondarycurr) / np.linalg.norm(X - xsecondarycurr)**3 - G * M * (X - xprimarycurr) / np.linalg.norm(X - xprimarycurr)**3
        V = V + 0.5 * (A + Aold) * dt

        xs[:, i] = X

    normX = np.sqrt(xs[0, :]**2 + xs[1, :]**2)
    angX = np.arctan2(xs[1, :], xs[0, :])
    angXcorot = angX - rotangle
    xscorot = np.array([normX * np.cos(angXcorot), normX * np.sin(angXcorot)])

    return xs, xscorot


def find_lagrange_point(x0, f, niter=1000, tol=1e-6, h=1e-9):
    xs = np.zeros(niter)
    xs[0] = x0

    ii = 0
    for ii in range(niter):
        dfdx = (f(xs[ii] + h) - f(xs[ii] - h)) / (2 * h)
        xs[ii + 1] = xs[ii] - f(xs[ii]) / dfdx
        if abs(f(xs[ii + 1])) < tol:
            break

    return xs[ii + 1]


def deltatoperiodic(Pfree, Pinit, Gm, GM, xsecondary, xprimary, omega, Nsteps):
    P = np.array([Pinit[0], 0, 0, Pfree[0]])
    T = Pfree[1]

    P0 = P.copy()

    dt = T / Nsteps
    ts = np.arange(dt, T+dt, dt)
    for t in ts:
        dPdt = dPdt_rk8(t, P, Gm, GM, xsecondary, xprimary, omega, dt)
        P = P + dPdt * dt

    deltaP = P - P0
    return deltaP


def deltatoperiodicenergy(Pfree, Pinit, Gm, GM, xsecondary, xprimary, omega, Nsteps, Vbarrier):
    P = np.array([Pfree[0], 0, 0, Pfree[1]])
    P0 = P.copy()

    T = Pfree[2]
    dt = T / Nsteps
    ts = np.arange(dt, T+dt, dt)

    for t in ts:
        dPdt = dPdt_rk8(t, P, Gm, GM, xsecondary, xprimary, omega, dt)
        P = P + dPdt * dt

    CJ = omega**2 * (P0[0]**2 + P0[1]**2) + 2 * (Gm / np.sqrt((P0[0] - xsecondary)**2 + P0[1]**2) + GM / np.sqrt((P0[0] - xprimary)**2 + P0[1]**2)) - (P0[2]**2 + P0[3]**2)
    V = -CJ / 2

    deltaP = np.append(P - P0, V - Vbarrier)
    return deltaP


def compute_reference_trajectory(Pinit, T, Nsteps, Gm, GM, xsecondary, xprimary, omega):
    xs, _ = integrate_rk8_full_state(Pinit, T, Nsteps, Gm, GM, xsecondary, xprimary, omega)
    return xs


def compute_periodic_orbit(Pinit, T, Nsteps, Gm, GM, xsecondary, xprimary, omega):
    Pinitfree = np.array([Pinit[3], T])
    f = lambda Pfree: deltatoperiodic(Pfree, Pinit, Gm, GM, xsecondary, xprimary, omega, Nsteps)
    sol = optimize.root(f, Pinitfree, method='lm')
    Pfree = sol.x

    P = np.array([Pinit[0], 0, 0, Pfree[0]])
    T_opt = Pfree[1]

    xs, _ = integrate_rk8_full_state(P, T_opt, Nsteps, Gm, GM, xsecondary, xprimary, omega)

    return P, T_opt, xs


def compute_periodic_family(Pinit, T, ds, max_iterations, Nsteps, Gm, GM, xsecondary, xprimary, omega):
    Ps = np.nan * np.zeros((4, max_iterations))
    Ts = np.nan * np.zeros(max_iterations)
    CJs = np.nan * np.zeros(max_iterations)
    initx = np.nan * np.zeros(max_iterations)
    all_xs = []

    ii = 0
    while Pinit[0] > 0.76:
        if ii >= max_iterations:
            break

        Pinitfree = np.array([Pinit[3], T])
        f = lambda Pfree: deltatoperiodic(Pfree, Pinit, Gm, GM, xsecondary, xprimary, omega, Nsteps)
        sol = optimize.root(f, Pinitfree, method='lm')
        Pfree = sol.x

        if sol.success == False:
            ii += 1
            continue

        P = np.array([Pinit[0], 0, 0, Pfree[0]])
        T = Pfree[1]

        Pinit = P.copy()
        Pinit[0] = Pinit[0] - ds

        Ps[:, ii] = P
        Ts[ii] = T

        xs, _ = integrate_rk8_full_state(P, T, Nsteps, Gm, GM, xsecondary, xprimary, omega)
        all_xs.append(xs)

        initx[ii] = xs[0, 0]
        CJ = omega**2 * (xs[0, :]**2 + xs[1, :]**2) + 2 * (Gm / np.sqrt((xs[0, :] - xsecondary)**2 + xs[1, :]**2) + GM / np.sqrt((xs[0, :] - xprimary)**2 + xs[1, :]**2)) - (xs[2, :]**2 + xs[3, :]**2)
        CJs[ii] = np.nanmean(CJ[~np.isnan(CJ)])

        ii += 1

    return Ps[:, :ii], Ts[:ii], CJs[:ii], initx[:ii], all_xs


def compute_energy_matched_orbit(Ps, Ts, CJs, Vbarrier, Nsteps, Gm, GM, xsecondary, xprimary, omega):
    crit = CJs + 2 * Vbarrier
    Iorbit = np.where(np.diff(np.sign(crit)))[0][0]
    Pcross = Ps[:, Iorbit]
    Tcross = Ts[Iorbit]

    Pinitfree = [Pcross[0], Pcross[3], Tcross]
    f = lambda Pfree: deltatoperiodicenergy(Pfree, Pcross, Gm, GM, xsecondary, xprimary, omega, Nsteps, Vbarrier)
    sol = optimize.root(f, Pinitfree, method='lm')
    Pfree = sol.x

    Popt = [Pfree[0], 0, 0, Pfree[1]]
    Topt = Pfree[2]

    xs, _ = integrate_rk8_full_state(Popt, Topt, Nsteps, Gm, GM, xsecondary, xprimary, omega)

    return np.array(Popt), Topt, xs


def compute_manifold_tubes(xsorb, Torb, Nsteps, Gm, GM, xsecondary, xprimary, omega, xsecondary_val):
    manifold_data = {
        'base_points': [],
        'unstable_tubes': [],
        'stable_tubes': [],
        'eigenvalues': [],
        'eigenvectors': []
    }

    Iall = range(0, xsorb.shape[1] - 1)
    ipoints = Iall[::2**3]

    for ipoint in ipoints:
        P0 = xsorb[:, ipoint]
        dt = Torb / Nsteps
        ts = np.arange(0, Torb + dt, dt)

        P = P0.copy()
        for t in ts[1:]:
            dPdt = dPdt_rk8(t, P, Gm, GM, xsecondary_val, xprimary, omega, dt)
            P = P + dPdt * dt
        xT = P.copy()

        M = []
        for icol in range(4):
            dP0 = np.zeros(4)
            scl = 1e-12
            dP0[icol] = -1 * scl

            P = P0 + dP0
            for t in ts[1:]:
                dPdt = dPdt_rk8(t, P, Gm, GM, xsecondary_val, xprimary, omega, dt)
                P = P + dPdt * dt
            xTpert = P.copy()
            dxT = (xTpert - xT) / scl

            M.append(dxT)

        D, V = np.linalg.eig(M)

        Imin = np.argmin(np.real(D))
        Vmin = V[:, Imin]
        Imax = np.argmax(np.real(D))
        Vmax = V[:, Imax]

        if np.imag(D[Imin]) != 0 or np.imag(D[Imax]) != 0:
            continue

        Vmin = np.real(Vmin)
        Vmax = np.real(Vmax)

        manifold_data['base_points'].append(P0)
        manifold_data['eigenvalues'].append((D[Imin], D[Imax]))
        manifold_data['eigenvectors'].append((Vmin, Vmax))

        unstable_tubes = []
        T_manifold = 5
        dt_manifold = 1e-3
        ts_manifold = np.arange(0, T_manifold + dt_manifold, dt_manifold)
        dist = 1e-4

        for isign in [-1, 1]:
            dt = abs(dt_manifold)
            P = P0 + dist * Vmin * isign
            xs = np.nan * np.zeros((4, len(ts_manifold)))

            for i, t in enumerate(ts_manifold):
                dPdt = dPdt_rk8(t, P, Gm, GM, xsecondary_val, xprimary, omega, dt)
                P = P + dPdt * dt
                xs[:, i] = P

                if np.linalg.norm(P[:2] - np.array([xsecondary_val, 0])) < 0.01:
                    break
                if P[0] - xsecondary_val > 0:
                    break

            unstable_tubes.append(xs)

        manifold_data['unstable_tubes'].append(unstable_tubes)

        stable_tubes = []
        for isign in [-1, 1]:
            dt = -abs(dt_manifold)
            P = P0 + dist * Vmax * isign
            xs = np.nan * np.zeros((4, len(ts_manifold)))

            for i, t in enumerate(ts_manifold):
                dPdt = dPdt_rk8(t, P, Gm, GM, xsecondary_val, xprimary, omega, dt)
                P = P + dPdt * dt
                xs[:, i] = P

                if np.linalg.norm(P[:2] - np.array([xsecondary_val, 0])) < 0.01:
                    break
                if P[0] - xsecondary_val > 0:
                    break

            stable_tubes.append(xs)

        manifold_data['stable_tubes'].append(stable_tubes)

    return manifold_data


def compute_initial_reference_trajectory(Pinit, T, Nsteps, Gm, GM, xsecondary, xprimary, omega, ax):
    xs = compute_reference_trajectory(Pinit, T, Nsteps, Gm, GM, xsecondary, xprimary, omega)
    plot_trajectory(ax, xs, color=[0.5, 0.5, 0.5], linewidth=1)
    scatter_point(ax, xs[0, 0], xs[1, 0], size=50, color=[0.5, 0.5, 0.5])
    scatter_point(ax, xs[0, -1], xs[1, -1], size=200, color=[0.5, 0.5, 0.5])
    return xs


def find_periodic_orbit(Pinit, T, Nsteps, Gm, GM, xsecondary, xprimary, omega, ax):
    P, T_opt, xs = compute_periodic_orbit(Pinit, T, Nsteps, Gm, GM, xsecondary, xprimary, omega)
    plot_trajectory(ax, xs, color=[0, 0, 1], linewidth=1)
    scatter_point(ax, xs[0, -1], xs[1, -1], size=200, color=[0, 0, 1])

    return P, T_opt


def continue_periodic_orbits(Pinit, T, ds, max_iterations, Nsteps, Gm, GM, xsecondary, xprimary, omega, ax):
    Ps, Ts, CJs, initx, all_xs = compute_periodic_family(Pinit, T, ds, max_iterations, Nsteps, Gm, GM, xsecondary, xprimary, omega)

    for xs in all_xs:
        plot_trajectory(ax, xs, color=[0, 0, 1], linewidth=1)
        scatter_point(ax, xs[0, -1], xs[1, -1], size=200, color=[0, 0, 1])

    return Ps, Ts, CJs, initx


def find_energy_matched_orbit(Ps, Ts, CJs, Vbarrier, Nsteps, Gm, GM, xsecondary, xprimary, omega, ax):
    Popt, Topt, xs = compute_energy_matched_orbit(Ps, Ts, CJs, Vbarrier, Nsteps, Gm, GM, xsecondary, xprimary, omega)
    plot_trajectory(ax, xs, color=[0, 0, 0], linewidth=1)
    scatter_point(ax, xs[0, -1], xs[1, -1], size=200, color=[0, 0, 0])

    return Popt, Topt, xs


def plot_manifold_tubes(ax, manifold_data):
    for base_point in manifold_data['base_points']:
        scatter_point(ax, base_point[0], base_point[1], size=100, color=[0, 0, 0])

    for unstable_tubes in manifold_data['unstable_tubes']:
        for xs in unstable_tubes:
            plot_trajectory(ax, xs, color=[1, 0, 0], linewidth=1, alpha=0.25)

    for stable_tubes in manifold_data['stable_tubes']:
        for xs in stable_tubes:
            plot_trajectory(ax, xs, color=[0, 0, 1], linewidth=1, alpha=0.25)


def compute_manifolds(xsorb, Torb, Nsteps, Gm, GM, xsecondary, xprimary, omega, ax):
    manifold_data = compute_manifold_tubes(xsorb, Torb, Nsteps, Gm, GM, xsecondary, xprimary, omega, xsecondary)
    plot_manifold_tubes(ax, manifold_data)


def run_demo():
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

    fig, ax = plt.subplots(figsize=(8, 8))

    scatter_point(ax, xprimary, 0, size=300, color=[0.75, 0.75, 0.75])
    scatter_point(ax, xsecondary, 0, size=100, color=[0.75, 0.75, 0.75])
    ax.set_aspect('equal')
    ax.set_xlim(tuple(np.array([-1, 1]) * 1.5))
    ax.set_ylim(tuple(np.array([-1, 1]) * 1.5))

    xx, yy = np.meshgrid(np.arange(-2, 2.01, 0.01), np.arange(-2, 2.01, 0.01))
    Vc = -0.5 * (xx**2 + yy**2) * omega**2
    Vg = -G * m / np.sqrt((xx - xsecondary)**2 + (yy - 0)**2) - G * M / np.sqrt((xx - xprimary)**2 + (yy - 0)**2)
    Veff = Vc + Vg
    ax.contour(xx, yy, Veff, [Vbarrier])

    y = 0
    f = lambda x: -1 * x * omega**2 - G * m * -1 / 2 * ((x - xsecondary)**2 + y**2)**(-3/2) * 2 * (x - xsecondary) - G * M * -1 / 2 * ((x - xprimary)**2 + y**2)**(-3/2) * 2 * (x - xprimary)

    L1_x = find_lagrange_point(0.78, f)
    L2_x = find_lagrange_point(1.19, f)
    L3_x = find_lagrange_point(-1, f)
    L4 = R * np.array([(xsecondary + xprimary) / 2, np.sqrt(3) / 2])
    L5 = R * np.array([(xsecondary + xprimary) / 2, -np.sqrt(3) / 2])

    scatter_point(ax, L1_x, 0, size=100, color=[0, 0, 0])
    scatter_point(ax, L2_x, 0, size=100, color=[0, 0, 0])
    scatter_point(ax, L3_x, 0, size=100, color=[0, 0, 0])
    scatter_point(ax, L4[0], L4[1], size=100, color=[0, 0, 0])
    scatter_point(ax, L5[0], L5[1], size=100, color=[0, 0, 0])

    L1 = [L1_x, 0]

    Pinit = np.array([0.77, 0, 0, 0.1551])
    Nsteps = 2**8
    T = 2.65

    compute_initial_reference_trajectory(Pinit, T, Nsteps, Gm, GM, xsecondary, xprimary, omega, ax)
    ax.set_xlim(tuple(np.array([-1, 1]) * 0.25 + L1[0]))
    ax.set_ylim(tuple(np.array([-1, 1]) * 0.25))

    P_periodic, T_periodic = find_periodic_orbit(Pinit, T, Nsteps, Gm, GM, xsecondary, xprimary, omega, ax)

    Ps, Ts, CJs, initx = continue_periodic_orbits(Pinit.copy(), T_periodic, 1e-4, 2000, Nsteps, Gm, GM, xsecondary, xprimary, omega, ax)

    Popt, Topt, xsorb = find_energy_matched_orbit(Ps, Ts, CJs, Vbarrier, Nsteps, Gm, GM, xsecondary, xprimary, omega, ax)

    compute_manifolds(xsorb, Topt, Nsteps, Gm, GM, xsecondary, xprimary, omega, ax)

    x_shoot = np.array([0.65, -0.15])
    angleshoot = 0.875
    vshoot = 1
    dx = vshoot * np.array([np.cos(angleshoot), np.sin(angleshoot)])
    CJlim = -2 * Vbarrier
    CJpot = omega**2 * (x_shoot[0]**2 + x_shoot[1]**2) + 2 * (Gm / np.sqrt((x_shoot[0] - xsecondary)**2 + (x_shoot[1] - 0)**2) + GM / np.sqrt((x_shoot[0] - xprimary)**2 + (x_shoot[1] - 0)**2))
    deltaJ = CJpot - CJlim

    dx2 = dx**2
    dx2 = dx2[0] + dx2[1]
    dxfac = (deltaJ / dx2)**0.5
    dx = dx * dxfac
    P_shoot = np.concatenate((x_shoot, dx))

    T_shoot = 15
    dt_shoot = 1e-3
    ts_shoot = np.arange(0, T_shoot, dt_shoot)
    xs_shoot = np.nan * np.ones((4, len(ts_shoot)))
    xs_shoot[:, 0] = P_shoot[0:4]
    P_curr = P_shoot.copy()

    for i, t in enumerate(ts_shoot[1:], 1):
        dPdt = dPdt_rk8(t, P_curr, Gm, GM, xsecondary, xprimary, omega, dt_shoot)
        P_curr = P_curr + dPdt * dt_shoot
        xs_shoot[:, i] = P_curr[0:4]

    plot_trajectory(ax, xs_shoot, color=[0, 0, 0], linewidth=1)
    scatter_point(ax, xs_shoot[0, -1], xs_shoot[1, -1], size=200, color=[0, 0, 0])

    ax.set_xlim(tuple(np.array([-1, 1]) * 0.25 + L1[0]))
    ax.set_ylim(tuple(np.array([-1, 1]) * 0.25))

    plt.show()


if __name__ == "__main__":
    run_demo()
