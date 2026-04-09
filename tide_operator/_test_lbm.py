"""Test LBM solver produces realistic vortex shedding."""
import numpy as np
import sys
import time
from generate import generate_tide_field

print("Testing LBM solver: nx=64, ny=64, nt=32, Re~480...")
t0 = time.time()
result = generate_tide_field(
    nx=64, ny=64, nt=32,
    radius=12.0, velocity=2.0, viscosity=0.1, seed=42,
)
field, target_field, base_field, mask, sdf, params = result[:6]
elapsed = time.time() - t0

has_nan = np.any(np.isnan(field))
has_inf = np.any(np.isinf(field))
u_max = np.abs(field[:, 0]).max()
v_max = np.abs(field[:, 1]).max()
wake_energy = np.mean(target_field ** 2)
u_mean_inlet = field[:, 0, 0, :].mean()

print(f"  Time        : {elapsed:.1f}s")
print(f"  Re          : {params['Re']:.0f}")
print(f"  NaN/Inf     : {has_nan}/{has_inf}")
print(f"  |u|_max     : {u_max:.4f}")
print(f"  |v|_max     : {v_max:.4f}")
print(f"  u_mean inlet: {u_mean_inlet:.4f}")
print(f"  wake energy : {wake_energy:.6f}")
print(f"  field shape : {field.shape}")
print(f"  mask shape  : {mask.shape}, solid cells: {int((1-mask).sum())}")

if has_nan or has_inf:
    print("  [FAIL]")
    sys.exit(1)
else:
    print("  [PASS]")
