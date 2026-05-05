import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from scipy.optimize import fsolve
from PIL import Image, ImageChops

mu = 0.012150582
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
U_eff_clipped = np.clip(U_eff, z_min, np.max(U_eff))

fig = go.Figure()

fig.add_trace(go.Surface(
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
))

def potential_gradient_x(x_val):
    r1_val = np.sqrt((x_val + mu)**2)
    r2_val = np.sqrt((x_val - (1 - mu))**2)
    return x_val - (1 - mu) / r1_val**3 * (x_val + mu) - mu / r2_val**3 * (x_val - (1 - mu))

L1 = fsolve(potential_gradient_x, 0.8)[0]
L2 = fsolve(potential_gradient_x, 1.2)[0]
L3 = fsolve(potential_gradient_x, -1.0)[0]

L4_x = 0.5 - mu
L4_y = np.sqrt(3) / 2
L5_x = 0.5 - mu
L5_y = -np.sqrt(3) / 2

lagrange_points = [
    (L1, 0, 'L1'),
    (L2, 0, 'L2'),
    (L3, 0, 'L3'),
    (L4_x, L4_y, 'L4'),
    (L5_x, L5_y, 'L5')
]

U_values = []
for lx, ly, _ in lagrange_points:
    r1_val = np.sqrt((lx + mu)**2 + ly**2)
    r2_val = np.sqrt((lx - (1 - mu))**2 + ly**2)
    u_val = -((1 - mu) / r1_val) - (mu / r2_val) - 0.5 * (lx**2 + ly**2)
    U_values.append(u_val)

for (lx, ly, name), u_val in zip(lagrange_points, U_values):
    fig.add_trace(go.Scatter3d(
        x=[lx],
        y=[ly],
        z=[u_val + 0.2],
        mode='markers',
        marker=dict(size=1, color='#FFA500', symbol='circle'),
        showlegend=False,
        hoverinfo='skip'
    ))

fig.add_trace(go.Scatter3d(
    x=[-mu],
    y=[0],
    z=[-1.5],
    mode='markers',
    marker=dict(size=10, color='lightblue', symbol='circle'),
    showlegend=False,
    hoverinfo='skip'
))

fig.add_trace(go.Scatter3d(
    x=[1 - mu],
    y=[0],
    z=[-1.5],
    mode='markers',
    marker=dict(size=2.7, color='white', symbol='circle'),
    showlegend=False,
    hoverinfo='skip'
))

fig.update_layout(
    autosize=False,
    width=1350,
    height=1350,
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        xaxis=dict(title='', showgrid=False, showticklabels=False, showbackground=False),
        yaxis=dict(title='', showgrid=False, showticklabels=False, showbackground=False),
        zaxis=dict(title='', showgrid=False, showticklabels=False, showbackground=False),
        camera=dict(eye=dict(x=0, y=0, z=3.0), up=dict(x=0, y=1, z=0)),
        bgcolor='rgba(0,0,0,0)'
    )
)

script_dir = Path(__file__).parent
figures_dir = script_dir.parent / 'Figures'
figures_dir.mkdir(parents=True, exist_ok=True)
output_path = figures_dir / 'lagrange_points_topdown.png'
fig.write_image(str(output_path), scale=6)

img = Image.open(output_path).convert('RGB')
bg = Image.new('RGB', img.size, 'white')
diff = ImageChops.difference(img, bg)
bbox = diff.getbbox()
if bbox:
    cropped = img.crop(bbox)
    cropped.save(output_path)
    print(f"PNG saved and cropped to {output_path}")
else:
    print(f"PNG saved to {output_path}")
