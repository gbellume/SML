import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
print(torch.__version__, torch.cuda.is_available())
import numpy as np
import matplotlib.pyplot as plt
from models import FNO, device, dtype

from earth_mars_transfer_helpers import *
from earth_mars_solver_functions import *

 

n_days = 2.5*365
n_arcs = 1
n_traj = 30
  




rng = np.random.default_rng(42)
base = calendar_to_epoch(2018, 7, 27) + 220 * 86400
deps_raw = base + rng.uniform(0, n_days * 86400, size=n_traj)
tofs_raw = np.ones(n_traj) * 200 

# deps, tofs, angles = filter_transfers(deps_raw, tofs_raw, min_angle=30.0)
print(f"Kept {len(deps_raw)} / {n_traj}")

soi, lamb, pert, corr, arc_lens = generate_dataset(
    deps_raw, tofs_raw, number_of_arcs=n_arcs
)

M, N = lamb.shape[0], lamb.shape[1] - 1

X_raw = lamb[:, :-1, :]
Y_raw = corr / arc_lens[:, None, None]
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

accel_mag = np.linalg.norm(Y_raw, axis=-1)  # (M, N)
s_arr = np.linspace(0, 1, accel_mag.shape[1])
epoch_days = (deps_raw - deps_raw.min()) / 86400.0

# Sort by epoch so the surface connects smoothly
order = np.argsort(epoch_days)
epoch_days = epoch_days[order]
accel_mag = accel_mag[order]

S, E = np.meshgrid(s_arr, epoch_days)
Z = np.log10(accel_mag + 1e-20)  # log10, guard against zeros

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(S, E, Z, cmap='viridis', alpha=0.9, edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, label='log₁₀(|a| [m/s²])')

ax.set_xlabel('Arc parameter s')
ax.set_ylabel('Start epoch [days from first]')
ax.set_zlabel('log₁₀(|a|) [m/s²]')
ax.set_title('Discretization invariance')
plt.tight_layout()
plt.savefig('discretization_invariance_3d.png', dpi=150)
plt.show()

plot_transfers(lamb, soi)
np.savez_compressed("dataset.npz",
    lamb=lamb, corr=corr, arc_lens=arc_lens,
    soi=soi, deps_raw=deps_raw, tofs_raw=tofs_raw,
    X_raw=X_raw, Y_raw=Y_raw,
    accel_mag=accel_mag, epoch_days=epoch_days, s_arr=s_arr)


# With n_arcs=1, Y_raw is (n_traj, 1, 3) — one thrust vector per trajectory
accel = Y_raw[:, 0, :]  # (n_traj, 3)
epoch_days = (deps_raw - deps_raw.min()) / 86400.0
order = np.argsort(epoch_days)

labels = ['Radial (R)', 'Along-track (S)', 'Cross-track (W)']

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i, (ax, label) in enumerate(zip(axes, labels)):
    ax.plot(epoch_days[order], accel[order, i], 'o-')
    ax.set_ylabel(f'a [{label}] [m/s²]')
    ax.grid(True)
axes[-1].set_xlabel('Start epoch [days from first]')
axes[0].set_title('Thrust components vs departure epoch (single arc)')
plt.tight_layout()
plt.savefig('thrust_vs_epoch.png', dpi=150)
plt.show()
print("Dataset saved to dataset.npz")