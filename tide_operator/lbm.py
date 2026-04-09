"""
Shared D2Q9 lattice Boltzmann utilities.

The numpy functions are used by the data generator, and the torch functions
are used by the training-time physics loss so both paths follow the exact same
discrete update.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - numpy-only workflows
    torch = None


CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36], dtype=np.float32)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)


def build_cylinder_geometry(nx: int, ny: int, radius: float, center_x=None, center_y=None):
    """Return fluid mask, SDF, and the cylinder center in grid coordinates."""
    if center_x is None:
        # Keep the cylinder safely away from the inlet while leaving more
        # downstream room for the wake to develop.
        center_x = max(radius + 4.0, 0.25 * nx)
    if center_y is None:
        center_y = 0.5 * (ny - 1)

    x = np.arange(nx, dtype=np.float32)
    y = np.arange(ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    sdf = r - float(radius)
    mask = (sdf >= 0.0).astype(np.float32)
    return mask, sdf.astype(np.float32), float(center_x), float(center_y)


def choose_lbm_parameters(velocity: float, viscosity: float, tau_min: float = 0.51):
    """
    Map physical parameters to a stable lattice parameterization.

    We keep the lattice inflow speed fixed and clamp tau to [tau_min, 1.8]
    so the solver stays stable even at high effective Re.  tau_min is kept
    slightly above 0.5 so we do not artificially suppress shedding.
    At tau_min=0.51 the effective Re ranges from ~384 (R=8) to ~768 (R=16)
    on a 64x64 grid, which is enough for clear von Karman shedding.
    """
    u_lbm = 0.08
    viscosity = max(float(viscosity), 1e-8)
    velocity = max(float(velocity), 1e-8)
    nu_lbm = u_lbm * viscosity / velocity
    tau = max(tau_min, 0.5 + 3.0 * nu_lbm)
    tau = min(tau, 1.8)
    nu_lbm_actual = (tau - 0.5) / 3.0
    tau_target = 0.5 + 3.0 * nu_lbm
    return dict(
        u_lbm=u_lbm,
        tau=float(tau),
        tau_target=float(tau_target),
        tau_clipped=bool(abs(tau - tau_target) > 1e-10),
        nu_lbm=float(nu_lbm_actual),
    )


def max_supported_physical_velocity(viscosity: float,
                                    tau_min: float = 0.51,
                                    u_lbm: float = 0.08) -> float:
    """
    Maximum physical velocity that can be represented without tau clipping.

    With the current nondimensionalization, clipping starts when

        0.5 + 3 * u_lbm * viscosity / velocity < tau_min

    so the no-clipping condition is ``velocity <= u_lbm * viscosity / nu_min``
    where ``nu_min = (tau_min - 0.5) / 3``.
    """
    viscosity = max(float(viscosity), 1e-8)
    nu_min = max((float(tau_min) - 0.5) / 3.0, 1e-8)
    return float(u_lbm * viscosity / nu_min)


def equilibrium_numpy(rho, ux, uy):
    """D2Q9 equilibrium distributions, returned as (9, nx, ny)."""
    rho = np.asarray(rho, dtype=np.float32)
    ux = np.asarray(ux, dtype=np.float32)
    uy = np.asarray(uy, dtype=np.float32)
    cu = 3.0 * (CX[:, None, None] * ux[None] + CY[:, None, None] * uy[None])
    u2 = ux[None] ** 2 + uy[None] ** 2
    return W[:, None, None] * rho[None] * (1.0 + cu + 0.5 * cu ** 2 - 1.5 * u2)


def macroscopic_numpy(f):
    """Return rho, ux, uy from populations shaped as (9, nx, ny)."""
    rho = f.sum(axis=0)
    rho_safe = np.maximum(rho, 1e-8)
    ux = (f * CX[:, None, None]).sum(axis=0) / rho_safe
    uy = (f * CY[:, None, None]).sum(axis=0) / rho_safe
    return rho, ux, uy


def populations_to_velocity_numpy(f, mask=None):
    """Convert populations (T, 9, nx, ny) or (9, nx, ny) to velocity fields."""
    arr = np.asarray(f, dtype=np.float32)
    squeezed = False
    if arr.ndim == 3:
        arr = arr[None]
        squeezed = True
    rho = arr.sum(axis=1)
    rho_safe = np.maximum(rho, 1e-8)
    ux = (arr * CX[None, :, None, None]).sum(axis=1) / rho_safe
    uy = (arr * CY[None, :, None, None]).sum(axis=1) / rho_safe
    vel = np.stack([ux, uy], axis=1)
    if mask is not None:
        vel *= mask[None, None]
    return vel[0] if squeezed else vel


def _stream_numpy(post):
    streamed = np.zeros_like(post)
    for i, (cx, cy) in enumerate(zip(CX, CY)):
        temp = np.roll(post[i], shift=int(cy), axis=1)
        if cx == 1:
            streamed[i, 1:, :] = temp[:-1, :]
        elif cx == -1:
            streamed[i, :-1, :] = temp[1:, :]
        else:
            streamed[i] = temp
    return streamed


def lbm_step_numpy(f, obstacle, tau: float, u_in: float):
    """One D2Q9 BGK step with proper inlet/outlet and far-field BCs.

    Inlet  : Zou-He style equilibrium at prescribed velocity ``u_in``.
    Outlet : zero-gradient copy from the second-to-last column, then
             populations are rescaled so that ``rho_outlet = 1`` to prevent
             the monotonic mass-accumulation that causes density drift and
             eventual blowup.
    Top/bot: zero-gradient (Neumann) far-field.
    """
    rho, ux, uy = macroscopic_numpy(f)
    ux = np.where(obstacle, 0.0, ux)
    uy = np.where(obstacle, 0.0, uy)

    feq = equilibrium_numpy(rho, ux, uy)
    post = f - (f - feq) / float(tau)
    streamed = _stream_numpy(post)
    streamed = np.where(obstacle[None], post[OPP], streamed)

    # --- Outlet (x = nx-1): zero-gradient + density correction -----------
    streamed[:, -1, :] = streamed[:, -2, :]
    rho_out = np.maximum(streamed[:, -1, :].sum(axis=0), 1e-6)
    streamed[:, -1, :] *= (1.0 / rho_out)[None, :]  # enforce rho = 1

    # --- Inlet (x = 0): equilibrium at prescribed velocity ---------------
    rho_in = np.maximum(streamed[:, 1, :].sum(axis=0), 1e-6)
    feq_in = equilibrium_numpy(
        rho_in[None, :],
        np.full((1, rho_in.shape[0]), float(u_in), dtype=np.float32),
        np.zeros((1, rho_in.shape[0]), dtype=np.float32),
    )
    streamed[:, 0, :] = feq_in[:, 0, :]

    # --- Top / bottom (y = 0, y = ny-1): zero-gradient far-field --------
    streamed[:, :, 0] = streamed[:, :, 1]
    streamed[:, :, -1] = streamed[:, :, -2]

    # Re-apply bounce-back (obstacle nodes must not be overwritten by BCs)
    streamed = np.where(obstacle[None], post[OPP], streamed)
    return streamed.astype(np.float32)


def lbm_multistep_numpy(f, obstacle, tau: float, u_in: float, n_steps: int):
    """Repeated numpy LBM stepping."""
    out = np.asarray(f, dtype=np.float32)
    for _ in range(int(n_steps)):
        out = lbm_step_numpy(out, obstacle, tau, u_in)
    return out


def initialize_populations(nx: int, ny: int, radius: float, velocity: float, viscosity: float, wake_strength: float = 1.0, seed=None):
    """Create the initial LBM state and geometry metadata."""
    rng = np.random.default_rng(seed)
    mask, sdf, cx, cy = build_cylinder_geometry(nx, ny, radius)
    obstacle = mask < 0.5
    lbm_params = choose_lbm_parameters(velocity, viscosity)

    rho0 = np.ones((nx, ny), dtype=np.float32)
    ux0 = np.full((nx, ny), lbm_params["u_lbm"], dtype=np.float32)
    uy0 = np.zeros((nx, ny), dtype=np.float32)

    y = np.arange(ny, dtype=np.float32)
    perturb = (1e-3 * float(wake_strength) * lbm_params["u_lbm"]) * np.sin(2.0 * np.pi * y / max(ny, 2))
    uy0 += perturb[None, :]
    uy0 += (2e-4 * float(wake_strength) * lbm_params["u_lbm"]) * rng.standard_normal((nx, ny)).astype(np.float32)

    ux0 *= mask
    uy0 *= mask
    f0 = equilibrium_numpy(rho0, ux0, uy0)
    return f0.astype(np.float32), mask.astype(np.float32), sdf.astype(np.float32), obstacle, lbm_params, (cx, cy)


def _torch_constants(device, dtype):
    if torch is None:
        raise ModuleNotFoundError("torch is required for torch LBM utilities")
    cx = torch.as_tensor(CX, device=device, dtype=dtype)
    cy = torch.as_tensor(CY, device=device, dtype=dtype)
    w = torch.as_tensor(W, device=device, dtype=dtype)
    opp = torch.as_tensor(OPP, device=device, dtype=torch.long)
    return cx, cy, w, opp


def equilibrium_torch(rho, ux, uy):
    """Torch D2Q9 equilibrium for rho/ux/uy shaped as (B, nx, ny)."""
    if torch is None:
        raise ModuleNotFoundError("torch is required for equilibrium_torch")
    cx, cy, w, _ = _torch_constants(rho.device, rho.dtype)
    cu = 3.0 * (cx[None, :, None, None] * ux[:, None] + cy[None, :, None, None] * uy[:, None])
    u2 = ux[:, None] ** 2 + uy[:, None] ** 2
    return w[None, :, None, None] * rho[:, None] * (1.0 + cu + 0.5 * cu ** 2 - 1.5 * u2)


def macroscopic_torch(f):
    """Return rho, ux, uy from populations shaped as (B, 9, nx, ny)."""
    if torch is None:
        raise ModuleNotFoundError("torch is required for macroscopic_torch")
    cx, cy, _, _ = _torch_constants(f.device, f.dtype)
    rho = f.sum(dim=1)
    rho_safe = rho.clamp_min(1e-8)
    ux = (f * cx[None, :, None, None]).sum(dim=1) / rho_safe
    uy = (f * cy[None, :, None, None]).sum(dim=1) / rho_safe
    return rho, ux, uy


def populations_to_velocity_torch(f, mask=None):
    """Convert torch populations (B, 9, nx, ny) to velocities (B, 2, nx, ny)."""
    if torch is None:
        raise ModuleNotFoundError("torch is required for populations_to_velocity_torch")
    rho, ux, uy = macroscopic_torch(f)
    vel = torch.stack([ux, uy], dim=1)
    if mask is not None:
        vel = vel * mask[:, None]
    return vel


def _stream_torch(post):
    if torch is None:
        raise ModuleNotFoundError("torch is required for _stream_torch")
    streamed = torch.zeros_like(post)
    for i, (cx, cy) in enumerate(zip(CX.tolist(), CY.tolist())):
        temp = torch.roll(post[:, i], shifts=int(cy), dims=-1)
        if cx == 1:
            streamed[:, i, 1:, :] = temp[:, :-1, :]
        elif cx == -1:
            streamed[:, i, :-1, :] = temp[:, 1:, :]
        else:
            streamed[:, i] = temp
    return streamed


def lbm_step_torch(f, obstacle, tau, u_in):
    """Torch equivalent of lbm_step_numpy for f shaped as (B, 9, nx, ny).

    Mirrors the numpy version: Zou-He inlet, density-corrected outlet,
    zero-gradient top/bottom.
    """
    if torch is None:
        raise ModuleNotFoundError("torch is required for lbm_step_torch")

    f = torch.nan_to_num(f, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
    rho, ux, uy = macroscopic_torch(f)
    rho = torch.nan_to_num(rho, nan=1.0, posinf=2.0, neginf=1.0).clamp(0.2, 2.5)
    ux = torch.nan_to_num(ux, nan=0.0, posinf=0.0, neginf=0.0)
    uy = torch.nan_to_num(uy, nan=0.0, posinf=0.0, neginf=0.0)
    speed = torch.sqrt(ux.pow(2) + uy.pow(2) + 1e-12)
    vel_scale = torch.clamp(0.20 / speed, max=1.0)
    ux = ux * vel_scale
    uy = uy * vel_scale
    ux = torch.where(obstacle, torch.zeros_like(ux), ux)
    uy = torch.where(obstacle, torch.zeros_like(uy), uy)

    feq = equilibrium_torch(rho, ux, uy)
    tau_b = tau[:, None, None, None]
    post = f - (f - feq) / tau_b
    streamed = _stream_torch(post)
    _, _, _, opp = _torch_constants(f.device, f.dtype)
    bounce = post.index_select(1, opp)
    streamed = torch.where(obstacle[:, None], bounce, streamed)

    # --- Inlet / outlet along x without in-place mutation ----------------
    rho_in = streamed[:, :, 1, :].sum(dim=1).clamp_min(1e-6)
    ux_in = u_in[:, None].expand_as(rho_in)
    uy_in = torch.zeros_like(ux_in)
    feq_in = equilibrium_torch(rho_in[:, None, :], ux_in[:, None, :], uy_in[:, None, :])
    left = feq_in[:, :, 0:1, :]
    right_src = streamed[:, :, -2:-1, :]
    rho_out = right_src.sum(dim=1, keepdim=True).clamp_min(1e-6)
    right = right_src / rho_out

    x_parts = [left]
    if streamed.shape[2] > 2:
        x_parts.append(streamed[:, :, 1:-1, :])
    x_parts.append(right)
    streamed_x = torch.cat(x_parts, dim=2)

    # --- Top / bottom: zero-gradient far-field, also without in-place ----
    bottom = streamed_x[:, :, :, 1:2]
    top = streamed_x[:, :, :, -2:-1]
    y_parts = [bottom]
    if streamed_x.shape[3] > 2:
        y_parts.append(streamed_x[:, :, :, 1:-1])
    y_parts.append(top)
    streamed = torch.cat(y_parts, dim=3)

    # Re-apply bounce-back
    streamed = torch.where(obstacle[:, None], bounce, streamed)
    streamed = torch.nan_to_num(streamed, nan=1e-6, posinf=10.0, neginf=1e-6).clamp_min(1e-6)
    return streamed


def lbm_multistep_torch(f, obstacle, tau, u_in, n_steps: int):
    """Repeated torch LBM stepping."""
    if torch is None:
        raise ModuleNotFoundError("torch is required for lbm_multistep_torch")
    out = f
    for _ in range(int(n_steps)):
        out = lbm_step_torch(out, obstacle, tau, u_in)
    return out
