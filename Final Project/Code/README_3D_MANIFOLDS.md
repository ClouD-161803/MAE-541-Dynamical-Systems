# 3D Manifold Visualization Guide

## Overview

`plot_manifolds_3d.py` creates an interactive 3D visualization of stable and unstable manifolds in the (x, y, vy) phase space, overlaid with the effective potential surface for the restricted 3-body problem at C = 3.17.

## What's Displayed

### Geometric Elements

- **Effective Potential Surface**: Translucent Magma-colored surface (highly transparent at 12% opacity)
- **Hill Curve (Zero Velocity Curve)**: Orange/gold line showing the boundary of allowed motion at C = 3.17
- **Lyapunov Orbit**: Bright green line - the periodic orbit used for manifold computation
- **Primary Body**: Light blue circle (larger) - the sun
- **Secondary Body**: White dot (smaller) - the planet

### Manifold Tubes

- **Unstable Manifold**: Red/orange colored trajectories that diverge forward in time
- **Stable Manifold**: Green/yellow colored trajectories that converge backward in time

### Axes

- **X-axis**: Position along the x-axis (primary-secondary line)
- **Y-axis**: Position perpendicular to the primary-secondary line
- **Z-axis (vy)**: Velocity component perpendicular to the orbital plane (out-of-plane motion)

## Usage

### Basic Execution

```bash
cd "Final Project/Code"
python plot_manifolds_3d.py
```

### Computation Time

- **First run**: ~5-10 minutes (computes periodic family + manifolds)
- **Periodic family**: ~2 minutes (finds 101 periodic orbits)
- **Manifold computation**: ~3-8 minutes (computes Jacobian + integrates tubes)

### Output

- **File**: `Figures/manifolds_3d_C_3.17.html` (interactive Plotly visualization)
- Can be opened in any web browser and rotated/zoomed interactively

## How It Uses the Refactored Code

### Layer 1 (Pure Computation Functions)

```python
from code_manifold2_python import (
    compute_periodic_family,           # Returns orbit family data
    compute_energy_matched_orbit,      # Returns matched orbit + trajectory
    compute_manifold_tubes             # Returns manifold data structure
)
```

### Returned Data Structures

**Periodic Family:**

```python
Ps, Ts, CJs, initx, all_xs = compute_periodic_family(...)
# Ps: Initial states for each orbit
# Ts: Period for each orbit  
# CJs: Jacobi constant for each orbit
# initx: Initial x-position
# all_xs: All trajectory data (4D: x,y,vx,vy)
```

**Manifold Data:**

```python
manifold_data = compute_manifold_tubes(...)
# Returns dict with:
# - 'base_points': Points on periodic orbit
# - 'unstable_tubes': Forward-time trajectories (red/orange)
# - 'stable_tubes': Backward-time trajectories (green/yellow)
# - 'eigenvalues': Monodromy matrix eigenvalues
# - 'eigenvectors': Corresponding eigenvectors
```

Each manifold tube is a (4, N) array containing full 4D phase space data.

## Customization

### Energy Level

Change `C_jacobi` at the top of the script:

```python
C_jacobi = 3.17  # Change this value
E = -C_jacobi / 2
```

### Visual Parameters

```python
# Potential surface opacity (lower = more transparent)
opacity=0.12,

# Grid resolution (lower = faster, lower quality)
resolution = 250

# Number of integration steps (lower = faster, lower quality)
Nsteps = 2**7

# Number of manifold points (lower = faster)
# Controlled by compute_manifold_tubes sampling
```

### Colormap

- **Potential surface**: Change `colorscale='Magma_r'` to another Plotly colorscale
- **Manifold colors**: Modify `unstable_colors` and `stable_colors` lists

### Camera View

```python
camera=dict(
    eye=dict(x=1.5, y=1.5, z=1.2),  # Viewing angle
    up=dict(x=0, y=0, z=1)           # Up direction
)
```

## Interpreting the Visualization

1. **The manifold structure shows**:
   - How trajectories escape from the unstable periodic orbit (red tubes)
   - How trajectories approach the stable periodic orbit (green tubes)
   - The 3D nature of dynamics including out-of-plane motion (vy)

2. **Energy matching**: All elements are at the same Jacobi constant C = 3.17, so the Hill curve, periodic orbit, and manifolds are energetically consistent

3. **The effective potential surface** provides context for the allowed motions—trajectories stay within regions where the potential is below the energy level

## Dependencies

- numpy
- scipy
- plotly
- scikit-image

All are included in the mae341 conda environment.

## For Your Own Plotting

You can use the Layer 1 computation functions to get data and plot with any tool:

```python
from code_manifold2_python import (
    compute_periodic_family,
    compute_energy_matched_orbit,
    compute_manifold_tubes
)

# Get computation results (no plotting)
Ps, Ts, CJs, initx, all_xs = compute_periodic_family(...)
Popt, Topt, xsorb = compute_energy_matched_orbit(...)
manifold_data = compute_manifold_tubes(...)

# Now plot with plotly, mayavi, matplotlib, or any other tool
# All data is in standard numpy arrays in 4D phase space
```

This separation allows you to:

- Use different plotting libraries
- Combine with other visualizations
- Export data for analysis
- Compute manifolds once, plot many times
