import sys
from pathlib import Path

print("Testing imports...")
sys.path.insert(0, str(Path(__file__).parent))

try:
    print("Importing code_manifold2_python...")
    from code_manifold2_python import (
        compute_periodic_family,
        compute_energy_matched_orbit,
        compute_manifold_tubes
    )
    print("✓ Imports successful!")

    print("Importing plotly...")
    import plotly.graph_objects as go
    print("✓ Plotly imported!")

    print("Creating test figure...")
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[1,2,3], y=[4,5,6], z=[7,8,9], mode='markers'))
    print("✓ Figure created!")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
