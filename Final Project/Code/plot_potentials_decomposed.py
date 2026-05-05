import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

mu = 0.05
resolution = 400

x = np.linspace(-1.5, 1.5, resolution)
y = np.linspace(-1.5, 1.5, resolution)
X, Y = np.meshgrid(x, y)
X = X.astype(float)
Y = Y.astype(float)

r1 = np.sqrt((X - (-mu))**2 + Y**2)
r2 = np.sqrt((X - (1 - mu))**2 + Y**2)

V_grav = -((1 - mu) / r1) - (mu / r2)
V_cent = -0.5 * (X**2 + Y**2)
U_eff = V_grav + V_cent

z_min = -3.0
V_grav_clipped = np.clip(V_grav, z_min, np.max(V_grav))
V_cent_clipped = np.clip(V_cent, z_min, np.max(V_cent))
U_eff_clipped = np.clip(U_eff, z_min, np.max(U_eff))

fig = make_subplots(
    rows=1, cols=3,
    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
    horizontal_spacing=0.02
)

fig.add_trace(
    go.Surface(
        z=V_grav_clipped,
        x=X,
        y=Y,
        colorscale='Blues',
        cmin=z_min,
        cmax=-1.5,
        opacity=0.85,
        contours=dict(
            x=dict(show=True, width=1, color='rgba(150,150,150,0.15)'),
            y=dict(show=True, width=1, color='rgba(150,150,150,0.15)')
        ),
        lighting=dict(ambient=0.4, diffuse=0.8, fresnel=0.2, specular=0.1, roughness=1.0),
        lightposition=dict(x=100, y=100, z=1000),
        showscale=False
    ),
    row=1, col=1
)

fig.add_trace(
    go.Surface(
        z=V_cent_clipped,
        x=X,
        y=Y,
        colorscale='Reds',
        cmin=z_min,
        cmax=-1.5,
        opacity=0.85,
        contours=dict(
            x=dict(show=True, width=1, color='rgba(150,150,150,0.15)'),
            y=dict(show=True, width=1, color='rgba(150,150,150,0.15)')
        ),
        lighting=dict(ambient=0.4, diffuse=0.8, fresnel=0.2, specular=0.1, roughness=1.0),
        lightposition=dict(x=100, y=100, z=1000),
        showscale=False
    ),
    row=1, col=2
)

fig.add_trace(
    go.Surface(
        z=U_eff_clipped,
        x=X,
        y=Y,
        colorscale='Magma_r',
        cmin=z_min,
        cmax=-1.5,
        opacity=0.85,
        contours=dict(
            x=dict(show=True, width=1, color='rgba(150,150,150,0.15)'),
            y=dict(show=True, width=1, color='rgba(150,150,150,0.15)')
        ),
        lighting=dict(ambient=0.4, diffuse=0.8, fresnel=0.2, specular=0.1, roughness=1.0),
        lightposition=dict(x=100, y=100, z=1000),
        showscale=False
    ),
    row=1, col=3
)

fig.update_layout(
    autosize=True,
    width=1800,
    height=600,
    margin=dict(l=0, r=0, b=0, t=0),
    showlegend=False
)

for i in range(1, 4):
    fig.update_scenes(
        xaxis=dict(title='', showgrid=False, showticklabels=False, showbackground=False),
        yaxis=dict(title='', showgrid=False, showticklabels=False, showbackground=False),
        zaxis=dict(title='', showgrid=False, showticklabels=False, showbackground=False),
        camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        bgcolor='rgba(0,0,0,0)',
        row=1, col=i
    )

script_dir = Path(__file__).parent
figures_dir = script_dir.parent / 'Figures'
figures_dir.mkdir(parents=True, exist_ok=True)
output_path = figures_dir / 'potentials_decomposed.png'
fig.write_image(str(output_path), scale=4)
print(f"PNG saved to {output_path}")
