import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from scipy.ndimage import binary_erosion
from skimage import measure
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

E = -1.585

fig.add_trace(go.Surface(
    z=np.full_like(X, E),
    x=X,
    y=Y,
    colorscale=[[0, 'rgba(255, 200, 100, 0.1)'], [1, 'rgba(255, 200, 100, 0.1)']],
    showscale=False,
    hoverinfo='skip',
    lighting=dict(ambient=0.2, diffuse=0.3)
))

contours = measure.find_contours(U_eff_clipped, E)

for contour in contours:
    if len(contour) > 10:
        i_indices = contour[:, 0]
        j_indices = contour[:, 1]
        
        contour_x = np.interp(j_indices, np.arange(len(x)), x)
        contour_y = np.interp(i_indices, np.arange(len(y)), y)
        contour_z = np.full_like(contour_x, E)

        fig.add_trace(go.Scatter3d(
            x=contour_x,
            y=contour_y,
            z=contour_z,
            mode='lines',
            line=dict(color='rgba(255, 165, 0, 0.7)', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))

fig.add_trace(go.Scatter3d(
    x=[-mu],
    y=[0],
    z=[E + 0.3],
    mode='markers',
    marker=dict(size=10, color='lightblue', symbol='circle'),
    showlegend=False,
    hoverinfo='skip'
))

fig.add_trace(go.Scatter3d(
    x=[1 - mu],
    y=[0],
    z=[E + 0.3],
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

C = -2 * E
output_path = figures_dir / f'hill_curve_{C:.2f}.png'
html_output_path = figures_dir / f'hill_curve_{C:.2f}.html'
fig.write_html(str(html_output_path), include_plotlyjs='cdn')
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
