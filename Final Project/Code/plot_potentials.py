import numpy as np
import plotly.graph_objects as go

mu = 0.05
resolution = 400

x = np.linspace(-1.5, 1.5, resolution)
y = np.linspace(-1.5, 1.5, resolution)
X, Y = np.meshgrid(x, y)

r1 = np.sqrt((X - (-mu))**2 + Y**2)
r2 = np.sqrt((X - (1 - mu))**2 + Y**2)

V_grav = -((1 - mu) / r1) - (mu / r2)
V_cent = -0.5 * (X**2 + Y**2)
U_eff = V_grav + V_cent

z_min = -3.0
U_eff_clipped = np.clip(U_eff, z_min, np.max(U_eff))

fig = go.Figure(data=[go.Surface(
    z=U_eff_clipped,
    x=X,
    y=Y,
    colorscale='Magma_r',
    cmin=z_min,
    cmax=-1.5,
    opacity=0.85,
    contours=dict(
        x=dict(show=True, width=1, color='rgba(0,0,0,0.15)'),
        y=dict(show=True, width=1, color='rgba(0,0,0,0.15)')
    ),
    lighting=dict(
        ambient=0.4,
        diffuse=0.8,
        fresnel=0.2,
        specular=0.2,
        roughness=0.5
    ),
    lightposition=dict(x=100, y=100, z=1000)
)])

fig.update_layout(
    title='Effective Potential of the Restricted 3-Body Problem',
    autosize=False,
    width=900,
    height=800,
    margin=dict(l=0, r=0, b=0, t=40),
    scene=dict(
        xaxis_title='X Axis',
        yaxis_title='Y Axis',
        zaxis_title='Energy / Potential',
        camera=dict(
            eye=dict(x=1.5, y=-1.5, z=1.2)
        )
    )
)

fig.show()