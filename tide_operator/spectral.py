"""
Helpers for reconstructing real-valued time signals from stored temporal
Fourier modes.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - allows numpy-only workflows
    torch = None


def _mode_weights(mode_idx, nt: int):
    """Weight non-self-conjugate modes by 2 to account for mirrored FFT bins."""
    if isinstance(mode_idx, np.ndarray):
        self_conj = (mode_idx == 0) | ((nt % 2 == 0) & (mode_idx == nt // 2))
        return np.where(self_conj, 1.0, 2.0).astype(np.float32)

    self_conj = (mode_idx == 0) | ((nt % 2 == 0) & (mode_idx == nt // 2))
    return torch.where(
        self_conj,
        torch.ones_like(mode_idx, dtype=torch.float32),
        torch.full_like(mode_idx, 2.0, dtype=torch.float32),
    )


def reconstruct_from_modes_numpy(mode_coeffs, mode_idx, nt: int, times=None):
    """
    Reconstruct real-valued vector fields from a sparse set of positive FFT modes.

    Parameters
    ----------
    mode_coeffs : complex array, shape (n_modes, n_components, nx, ny)
    mode_idx    : int array, shape (n_modes,)
    nt          : total number of timesteps in the original sequence
    times       : optional iterable of timesteps to reconstruct

    Returns
    -------
    field : float32 array, shape (n_times, n_components, nx, ny)
    """
    coeffs = np.asarray(mode_coeffs)
    if coeffs.ndim != 4:
        raise ValueError("mode_coeffs must have shape (n_modes, n_components, nx, ny)")

    if times is None:
        times = np.arange(nt, dtype=np.float32)
    else:
        times = np.asarray(times, dtype=np.float32)

    mode_idx = np.asarray(mode_idx, dtype=np.int64)
    weights = _mode_weights(mode_idx, nt).astype(np.float32)

    angles = 2.0 * np.pi / float(nt) * times[:, None] * mode_idx[None, :].astype(np.float32)
    cos_term = np.cos(angles)[:, :, None, None, None]
    sin_term = np.sin(angles)[:, :, None, None, None]

    real = coeffs.real.astype(np.float32)[None, ...]
    imag = coeffs.imag.astype(np.float32)[None, ...]
    weighted = weights[None, :, None, None, None]

    field = weighted * (real * cos_term - imag * sin_term)
    return field.sum(axis=1).astype(np.float32)


def reconstruct_from_modes_torch(coeffs: torch.Tensor,
                                 mode_idx: torch.Tensor,
                                 nt: int,
                                 times=None) -> torch.Tensor:
    """
    Torch version of the sparse Fourier reconstruction.

    Parameters
    ----------
    coeffs   : (B, n_modes, n_components, 2, nx, ny)
    mode_idx : (B, n_modes)
    nt       : total number of timesteps
    times    : optional 1D list/tensor of timesteps

    Returns
    -------
    field : (B, n_times, n_components, nx, ny)
    """
    if torch is None:
        raise ModuleNotFoundError("torch is required for reconstruct_from_modes_torch")

    if coeffs.ndim != 6:
        raise ValueError("coeffs must have shape (B, n_modes, n_components, 2, nx, ny)")

    device = coeffs.device
    dtype = coeffs.dtype

    if times is None:
        times = torch.arange(nt, device=device, dtype=dtype)
    else:
        times = torch.as_tensor(times, device=device, dtype=dtype)

    mode_idx = mode_idx.to(device=device, dtype=dtype)
    weights = _mode_weights(mode_idx.to(torch.int64), nt).to(device=device, dtype=dtype)

    angles = 2.0 * torch.pi / float(nt) * mode_idx[:, None, :] * times[None, :, None]
    cos_term = torch.cos(angles)[:, :, :, None, None, None]
    sin_term = torch.sin(angles)[:, :, :, None, None, None]

    real = coeffs[:, None, :, :, 0, :, :]
    imag = coeffs[:, None, :, :, 1, :, :]
    weighted = weights[:, None, :, None, None, None]

    field = weighted * (real * cos_term - imag * sin_term)
    return field.sum(dim=2)


def reconstruct_query_modes_numpy(coeffs, mode_idx, nt: int, times=None):
    """
    Reconstruct local time signals from query-conditioned Fourier coefficients.

    Parameters
    ----------
    coeffs   : (Q, n_modes, n_components, 2) or (B, Q, n_modes, n_components, 2)
    mode_idx : (n_modes,) or (B, n_modes)
    nt       : total number of timesteps
    times    : optional iterable of timesteps

    Returns
    -------
    field : (..., n_times, n_components)
    """
    arr = np.asarray(coeffs, dtype=np.float32)
    squeezed = False
    if arr.ndim == 4:
        arr = arr[None]
        squeezed = True
    if arr.ndim != 5:
        raise ValueError("coeffs must have shape (Q, n_modes, n_components, 2) or (B, Q, n_modes, n_components, 2)")

    if times is None:
        times = np.arange(nt, dtype=np.float32)
    else:
        times = np.asarray(times, dtype=np.float32)

    mode_idx = np.asarray(mode_idx, dtype=np.int64)
    if mode_idx.ndim == 1:
        mode_idx = np.broadcast_to(mode_idx[None], (arr.shape[0], mode_idx.shape[0]))
    weights = _mode_weights(mode_idx, nt).astype(np.float32)

    angles = 2.0 * np.pi / float(nt) * mode_idx[:, None, :] * times[None, :, None]
    cos_term = np.cos(angles)[:, None, :, :, None]
    sin_term = np.sin(angles)[:, None, :, :, None]

    real = arr[..., 0][:, :, None, :, :]
    imag = arr[..., 1][:, :, None, :, :]
    weighted = weights[:, None, None, :, None]

    field = weighted * (real * cos_term - imag * sin_term)
    field = field.sum(axis=3)
    return field[0] if squeezed else field


def reconstruct_query_modes_torch(coeffs: torch.Tensor,
                                  mode_idx: torch.Tensor,
                                  nt: int,
                                  times=None) -> torch.Tensor:
    """
    Torch reconstruction for local query coefficients.

    Parameters
    ----------
    coeffs   : (B, Q, n_modes, n_components, 2)
    mode_idx : (B, n_modes) or (n_modes,)
    nt       : total number of timesteps
    times    : optional 1D list/tensor of timesteps

    Returns
    -------
    field : (B, Q, n_times, n_components)
    """
    if torch is None:
        raise ModuleNotFoundError("torch is required for reconstruct_query_modes_torch")
    if coeffs.ndim != 5:
        raise ValueError("coeffs must have shape (B, Q, n_modes, n_components, 2)")

    device = coeffs.device
    dtype = coeffs.dtype

    if times is None:
        times = torch.arange(nt, device=device, dtype=dtype)
    else:
        times = torch.as_tensor(times, device=device, dtype=dtype)

    if mode_idx.ndim == 1:
        mode_idx = mode_idx[None].expand(coeffs.shape[0], -1)
    mode_idx = mode_idx.to(device=device)
    weights = _mode_weights(mode_idx.to(torch.int64), nt).to(device=device, dtype=dtype)

    angles = 2.0 * torch.pi / float(nt) * mode_idx[:, None, :] * times[None, :, None]
    cos_term = torch.cos(angles)[:, None, :, :, None]
    sin_term = torch.sin(angles)[:, None, :, :, None]

    real = coeffs[..., 0][:, :, None, :, :]
    imag = coeffs[..., 1][:, :, None, :, :]
    weighted = weights[:, None, None, :, None]

    field = weighted * (real * cos_term - imag * sin_term)
    return field.sum(dim=3)


def reconstruct_query_mode_time_derivative_torch(coeffs: torch.Tensor,
                                                 mode_idx: torch.Tensor,
                                                 nt: int,
                                                 times=None,
                                                 dt=1.0) -> torch.Tensor:
    """
    Torch reconstruction of the time derivative for local query coefficients.

    Parameters
    ----------
    coeffs   : (B, Q, n_modes, n_components, 2)
    mode_idx : (B, n_modes) or (n_modes,)
    nt       : total number of stored timesteps
    times    : optional 1D list/tensor of timesteps
    dt       : physical time between stored frames (scalar or shape (B,))

    Returns
    -------
    dfield_dt : (B, Q, n_times, n_components)
    """
    if torch is None:
        raise ModuleNotFoundError("torch is required for reconstruct_query_mode_time_derivative_torch")
    if coeffs.ndim != 5:
        raise ValueError("coeffs must have shape (B, Q, n_modes, n_components, 2)")

    device = coeffs.device
    dtype = coeffs.dtype

    if times is None:
        times = torch.arange(nt, device=device, dtype=dtype)
    else:
        times = torch.as_tensor(times, device=device, dtype=dtype)

    if mode_idx.ndim == 1:
        mode_idx = mode_idx[None].expand(coeffs.shape[0], -1)
    mode_idx = mode_idx.to(device=device)
    mode_idx_float = mode_idx.to(dtype=dtype)
    weights = _mode_weights(mode_idx.to(torch.int64), nt).to(device=device, dtype=dtype)

    if not torch.is_tensor(dt):
        dt = torch.tensor(float(dt), device=device, dtype=dtype)
    else:
        dt = dt.to(device=device, dtype=dtype)
    if dt.ndim == 0:
        dt = dt.expand(coeffs.shape[0])

    angles = 2.0 * torch.pi / float(nt) * mode_idx_float[:, None, :] * times[None, :, None]
    freq = (2.0 * torch.pi / float(nt)) * mode_idx_float / dt[:, None]
    cos_term = torch.cos(angles)[:, None, :, :, None]
    sin_term = torch.sin(angles)[:, None, :, :, None]

    real = coeffs[..., 0][:, :, None, :, :]
    imag = coeffs[..., 1][:, :, None, :, :]
    weighted = weights[:, None, None, :, None]
    freq = freq[:, None, None, :, None]

    dfield_dt = weighted * freq * (-real * sin_term - imag * cos_term)
    return dfield_dt.sum(dim=3)
