"""Training loop for the query-conditioned Tide Fourier operator."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from equations import vorticity_transport_residual
from generate import SIM_FORMAT_VERSION, generate_dataset, load_simulation
from query_dataset import TideQueryDataset
from query_model import TideQueryOperator, apply_hard_constraints, extract_center_patch
from spectral import reconstruct_query_mode_time_derivative_torch, reconstruct_query_modes_torch


def _coeff_scale_tensor(coeff_scale, reference: torch.Tensor):
    scale = torch.as_tensor(coeff_scale, device=reference.device, dtype=reference.dtype)
    while scale.ndim < reference.ndim - 1:
        scale = scale.unsqueeze(0)
    return scale.unsqueeze(-1)


def _query_weights(query_wake, query_mask, wake_focus: float):
    return query_mask * (1.0 + wake_focus * query_wake.clamp_min(0.0))


def _weighted_reduce(values, weights):
    return (values * weights).sum() / (weights.sum() + 1e-8)


def _flatten_patch_queries(coeffs: torch.Tensor):
    batch, n_query, n_patch = coeffs.shape[:3]
    flat = coeffs.reshape(batch, n_query * n_patch, *coeffs.shape[3:])
    return flat, (batch, n_query, n_patch)


def _full_grid_query_xy(nx: int, ny: int, batch_size: int, device, dtype):
    x = torch.linspace(-1.0, 1.0, nx, device=device, dtype=dtype)
    y = torch.linspace(-1.0, 1.0, ny, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    return coords.unsqueeze(0).expand(batch_size, -1, -1)


def _physics_collocation_times(nt: int, n_samples: int, device, dtype):
    if n_samples <= 0 or n_samples >= nt:
        return torch.arange(nt, device=device, dtype=dtype)
    times = torch.linspace(0.0, float(nt - 1), steps=int(n_samples), device=device, dtype=dtype)
    return torch.unique(times.round(), sorted=True)


def _reshape_query_series(field: torch.Tensor, nx: int, ny: int) -> torch.Tensor:
    batch, n_query, n_time, n_comp = field.shape
    if n_query != nx * ny:
        raise ValueError(f"Expected {nx * ny} full-grid queries, got {n_query}.")
    return field.reshape(batch, nx, ny, n_time, n_comp).permute(0, 3, 4, 1, 2).contiguous()


def _central_diff_x(field: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    grad = torch.zeros_like(field)
    # 4th-order interior: (-f[i+2] + 8f[i+1] - 8f[i-1] + f[i-2]) / (12*dx)
    grad[..., 2:-2, :] = (
        -field[..., 4:, :] + 8.0 * field[..., 3:-1, :] - 8.0 * field[..., 1:-3, :] + field[..., :-4, :]
    ) / (12.0 * dx)
    # 2nd-order fallback for cells adjacent to boundaries
    grad[..., 1, :] = (field[..., 2, :] - field[..., 0, :]) / (2.0 * dx)
    grad[..., -2, :] = (field[..., -1, :] - field[..., -3, :]) / (2.0 * dx)
    return grad


def _central_diff_y(field: torch.Tensor, dy: float = 1.0) -> torch.Tensor:
    grad = torch.zeros_like(field)
    # 4th-order interior
    grad[..., :, 2:-2] = (
        -field[..., :, 4:] + 8.0 * field[..., :, 3:-1] - 8.0 * field[..., :, 1:-3] + field[..., :, :-4]
    ) / (12.0 * dy)
    # 2nd-order fallback
    grad[..., :, 1] = (field[..., :, 2] - field[..., :, 0]) / (2.0 * dy)
    grad[..., :, -2] = (field[..., :, -1] - field[..., :, -3]) / (2.0 * dy)
    return grad


def _second_diff_x(field: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    out = torch.zeros_like(field)
    # 4th-order interior: (-f[i+2] + 16f[i+1] - 30f[i] + 16f[i-1] - f[i-2]) / (12*dx^2)
    out[..., 2:-2, :] = (
        -field[..., 4:, :] + 16.0 * field[..., 3:-1, :] - 30.0 * field[..., 2:-2, :]
        + 16.0 * field[..., 1:-3, :] - field[..., :-4, :]
    ) / (12.0 * dx ** 2)
    # 2nd-order fallback
    out[..., 1, :] = (field[..., 2, :] - 2.0 * field[..., 1, :] + field[..., 0, :]) / (dx ** 2)
    out[..., -2, :] = (field[..., -1, :] - 2.0 * field[..., -2, :] + field[..., -3, :]) / (dx ** 2)
    return out


def _second_diff_y(field: torch.Tensor, dy: float = 1.0) -> torch.Tensor:
    out = torch.zeros_like(field)
    # 4th-order interior
    out[..., :, 2:-2] = (
        -field[..., :, 4:] + 16.0 * field[..., :, 3:-1] - 30.0 * field[..., :, 2:-2]
        + 16.0 * field[..., :, 1:-3] - field[..., :, :-4]
    ) / (12.0 * dy ** 2)
    # 2nd-order fallback
    out[..., :, 1] = (field[..., :, 2] - 2.0 * field[..., :, 1] + field[..., :, 0]) / (dy ** 2)
    out[..., :, -2] = (field[..., :, -1] - 2.0 * field[..., :, -2] + field[..., :, -3]) / (dy ** 2)
    return out


def _fluid_interior_mask(mask: torch.Tensor) -> torch.Tensor:
    valid = torch.zeros_like(mask, dtype=torch.bool)
    # Exclude 2 cells from each edge to match 4th-order stencil width
    valid[:, 2:-2, 2:-2] = (
        (mask[:, 2:-2, 2:-2] > 0.5)
        & (mask[:, :-4, 2:-2] > 0.5)
        & (mask[:, 4:, 2:-2] > 0.5)
        & (mask[:, 2:-2, :-4] > 0.5)
        & (mask[:, 2:-2, 4:] > 0.5)
        & (mask[:, 1:-3, 2:-2] > 0.5)
        & (mask[:, 3:-1, 2:-2] > 0.5)
        & (mask[:, 2:-2, 1:-3] > 0.5)
        & (mask[:, 2:-2, 3:-1] > 0.5)
    )
    return valid


def _predict_dense_center_coeffs(model,
                                 feature_map: torch.Tensor,
                                 sharp_map: torch.Tensor,
                                 global_feat: torch.Tensor,
                                 param_feat: torch.Tensor,
                                 mask: torch.Tensor,
                                 nx: int,
                                 ny: int,
                                 query_chunk_size: int) -> torch.Tensor:
    batch = feature_map.shape[0]
    coords = _full_grid_query_xy(nx, ny, batch, feature_map.device, feature_map.dtype)
    flat_mask = mask.reshape(batch, -1).to(feature_map.dtype)
    ix = torch.arange(nx, device=feature_map.device).repeat_interleave(ny)
    preds = []
    for start in range(0, coords.shape[1], int(query_chunk_size)):
        end = min(coords.shape[1], start + int(query_chunk_size))
        patch_pred = model.decode_queries(feature_map, sharp_map, global_feat, param_feat, coords[:, start:end])
        center = extract_center_patch(patch_pred, model.center_patch_index)
        center = center * flat_mask[:, start:end, None, None, None]
        inlet = (ix[start:end] == 0).to(center.dtype)[None, :, None, None, None]
        center = center * (1.0 - inlet)
        preds.append(center)
    return torch.cat(preds, dim=1)


def coefficient_loss(pred, target, weights, mode_weights=None, coeff_scale=None):
    diff = F.smooth_l1_loss(pred, target, reduction="none")
    # pred shape: (B, Q, n_modes, n_components, 2)
    # Normalize each mode by its energy scale so all modes contribute equally
    if coeff_scale is not None:
        scale = torch.as_tensor(coeff_scale, device=diff.device, dtype=diff.dtype)
        scale = scale.view(1, 1, scale.shape[0], scale.shape[1], 1).clamp(min=1e-6)
        diff = diff / scale
    if mode_weights is not None:
        mw = mode_weights.to(diff.device).view(1, 1, -1, 1, 1)
        diff = diff * mw
    per_query = diff.flatten(start_dim=2).mean(dim=2)
    return _weighted_reduce(per_query, weights)


def amplitude_loss(pred, target, weights, mode_weights=None, coeff_scale=None):
    pred_amp = torch.linalg.vector_norm(pred, dim=-1)
    target_amp = torch.linalg.vector_norm(target, dim=-1)
    diff = F.smooth_l1_loss(pred_amp, target_amp, reduction="none")
    # diff shape: (B, Q, n_modes, n_components)
    if coeff_scale is not None:
        scale = torch.as_tensor(coeff_scale, device=diff.device, dtype=diff.dtype)
        scale = scale.view(1, 1, scale.shape[0], scale.shape[1]).clamp(min=1e-6)
        diff = diff / scale
    if mode_weights is not None:
        mw = mode_weights.to(diff.device).view(1, 1, -1, 1)
        diff = diff * mw
    per_query = diff.flatten(start_dim=2).mean(dim=2)
    return _weighted_reduce(per_query, weights)


def reconstruction_loss(pred, target, mode_idx, nt: int, weights, query_chunk_size: int | None = None):
    batch, n_query = pred.shape[:2]
    if query_chunk_size is None or query_chunk_size <= 0:
        query_chunk_size = n_query

    total = pred.new_zeros(())
    total_weight = pred.new_zeros(())
    for start in range(0, n_query, int(query_chunk_size)):
        end = min(n_query, start + int(query_chunk_size))
        pred_chunk = pred[:, start:end]
        target_chunk = target[:, start:end]
        weight_chunk = weights[:, start:end]

        pred_recon = reconstruct_query_modes_torch(pred_chunk, mode_idx, nt=nt)
        target_recon = reconstruct_query_modes_torch(target_chunk, mode_idx, nt=nt)
        diff2 = (pred_recon - target_recon).pow(2).mean(dim=(2, 3))
        total = total + (diff2 * weight_chunk).sum()
        total_weight = total_weight + weight_chunk.sum()
    return total / (total_weight + 1e-8)


def vorticity_reconstruction_loss(pred,
                                  target,
                                  mode_idx,
                                  nt: int,
                                  weights,
                                  query_div_mask,
                                  coeff_scale,
                                  patch_index,
                                  nx: int,
                                  ny: int,
                                  query_chunk_size: int | None = None):
    batch, n_query, n_patch = pred.shape[:3]
    if query_chunk_size is None or query_chunk_size <= 0:
        query_chunk_size = n_query

    dx = 2.0 / max(nx - 1, 1)
    dy = 2.0 / max(ny - 1, 1)
    total = pred.new_zeros(())
    total_weight = pred.new_zeros(())

    for start in range(0, n_query, int(query_chunk_size)):
        end = min(n_query, start + int(query_chunk_size))
        scale_chunk = _coeff_scale_tensor(coeff_scale, pred[:, start:end])
        pred_chunk = pred[:, start:end] * scale_chunk
        target_chunk = target[:, start:end] * scale_chunk
        weight_chunk = weights[:, start:end] * query_div_mask[:, start:end]

        pred_flat, _ = _flatten_patch_queries(pred_chunk)
        target_flat, _ = _flatten_patch_queries(target_chunk)
        pred_recon = reconstruct_query_modes_torch(pred_flat, mode_idx, nt=nt)
        target_recon = reconstruct_query_modes_torch(target_flat, mode_idx, nt=nt)
        pred_recon = pred_recon.reshape(batch, end - start, n_patch, nt, pred.shape[4])
        target_recon = target_recon.reshape(batch, end - start, n_patch, nt, target.shape[4])

        pred_vort = (
            (pred_recon[:, :, patch_index["right"], :, 1] - pred_recon[:, :, patch_index["left"], :, 1]) / (2.0 * dx)
            - (pred_recon[:, :, patch_index["up"], :, 0] - pred_recon[:, :, patch_index["down"], :, 0]) / (2.0 * dy)
        )
        target_vort = (
            (target_recon[:, :, patch_index["right"], :, 1] - target_recon[:, :, patch_index["left"], :, 1]) / (2.0 * dx)
            - (target_recon[:, :, patch_index["up"], :, 0] - target_recon[:, :, patch_index["down"], :, 0]) / (2.0 * dy)
        )
        diff = (pred_vort - target_vort).pow(2).mean(dim=-1)
        total = total + (diff * weight_chunk).sum()
        total_weight = total_weight + weight_chunk.sum()
    return total / (total_weight + 1e-8)


def relative_l2(pred, target, query_mask, coeff_scale):
    scale = _coeff_scale_tensor(coeff_scale, pred)
    pred_raw = pred * scale
    target_raw = target * scale

    err = (pred_raw - target_raw).pow(2).sum(dim=(2, 3, 4))
    tgt2 = target_raw.pow(2).sum(dim=(2, 3, 4))

    num = (err * query_mask).sum()
    den = (tgt2 * query_mask).sum() + 1e-8
    return (num / den).sqrt().item() * 100.0


def _relative_l2_dense(pred_coeffs: np.ndarray, target_coeffs: np.ndarray, mask: np.ndarray) -> float:
    pred_arr = np.asarray(pred_coeffs)
    target_arr = np.asarray(target_coeffs)
    fluid = np.asarray(mask, dtype=np.float32)[None, None]
    num = np.sum(np.abs(pred_arr - target_arr) ** 2 * fluid)
    den = np.sum(np.abs(target_arr) ** 2 * fluid) + 1e-8
    return float(np.sqrt(num / den) * 100.0)


def _sample_param_vector(sample: dict) -> np.ndarray:
    params = sample["params"]
    return np.array(
        [params["velocity"], params["viscosity"], params["radius"]],
        dtype=np.float32,
    )


def compute_baselines(train_ds: TideQueryDataset, val_ds: TideQueryDataset) -> dict:
    train_samples = train_ds.samples
    val_samples = val_ds.samples
    n_modes = train_ds.n_modes

    train_coeffs = np.stack([sample["mode_coeffs"][:n_modes] for sample in train_samples], axis=0)
    train_params = np.stack([_sample_param_vector(sample) for sample in train_samples], axis=0)
    mean_coeffs = train_coeffs.mean(axis=0)
    param_scale = np.maximum(train_ds.param_std.astype(np.float32), 1e-6)

    zero_errs, mean_errs, nn_errs = [], [], []
    for sample in val_samples:
        target = sample["mode_coeffs"][:n_modes]
        mask = sample["mask"]
        zero_errs.append(_relative_l2_dense(np.zeros_like(target), target, mask))
        mean_errs.append(_relative_l2_dense(mean_coeffs, target, mask))

        p = _sample_param_vector(sample)
        dists = np.sum(((train_params - p[None]) / param_scale[None]) ** 2, axis=1)
        nn_coeffs = train_coeffs[int(np.argmin(dists))]
        nn_errs.append(_relative_l2_dense(nn_coeffs, target, mask))

    return dict(
        zero=float(np.mean(zero_errs)),
        train_mean=float(np.mean(mean_errs)),
        nearest_neighbor=float(np.mean(nn_errs)),
    )


def divergence_loss(pred, query_div_mask, weights, coeff_scale, patch_index, nx: int, ny: int):
    scale = _coeff_scale_tensor(coeff_scale, pred)
    pred_raw = pred * scale
    left = pred_raw[:, :, patch_index["left"]]
    right = pred_raw[:, :, patch_index["right"]]
    down = pred_raw[:, :, patch_index["down"]]
    up = pred_raw[:, :, patch_index["up"]]

    dx = 2.0 / max(nx - 1, 1)
    dy = 2.0 / max(ny - 1, 1)
    du_dx = (right[:, :, :, 0] - left[:, :, :, 0]) / (2.0 * dx)
    dv_dy = (up[:, :, :, 1] - down[:, :, :, 1]) / (2.0 * dy)
    div = du_dx + dv_dy
    per_query = div.pow(2).mean(dim=(2, 3))
    return _weighted_reduce(per_query, weights * query_div_mask)


def boundary_condition_loss(pred, query_ij, query_mask, weights, coeff_scale, patch_index, nx: int, ny: int):
    scale = _coeff_scale_tensor(coeff_scale, pred)
    pred_raw = pred * scale
    center = pred_raw[:, :, patch_index["center"]]
    left = pred_raw[:, :, patch_index["left"]]
    down = pred_raw[:, :, patch_index["down"]]
    up = pred_raw[:, :, patch_index["up"]]

    ix = query_ij[:, :, 0]
    iy = query_ij[:, :, 1]

    outlet_mask = query_mask * (ix == (nx - 1)).float()
    top_mask = query_mask * (iy == (ny - 1)).float()
    bottom_mask = query_mask * (iy == 0).float()

    zero = center.new_zeros(())
    losses = []
    if outlet_mask.sum() > 0:
        diff = F.smooth_l1_loss(center, left, reduction="none").mean(dim=(2, 3, 4))
        losses.append(_weighted_reduce(diff, weights * outlet_mask))
    if top_mask.sum() > 0:
        diff = F.smooth_l1_loss(center, down, reduction="none").mean(dim=(2, 3, 4))
        losses.append(_weighted_reduce(diff, weights * top_mask))
    if bottom_mask.sum() > 0:
        diff = F.smooth_l1_loss(center, up, reduction="none").mean(dim=(2, 3, 4))
        losses.append(_weighted_reduce(diff, weights * bottom_mask))
    return torch.stack(losses).mean() if losses else zero


def global_physics_loss(model,
                        feature_map: torch.Tensor,
                        sharp_map: torch.Tensor,
                        global_feat: torch.Tensor,
                        param_feat: torch.Tensor,
                        base_field: torch.Tensor,
                        mask: torch.Tensor,
                        params_raw: torch.Tensor,
                        frame_dt: torch.Tensor,
                        mode_idx: torch.Tensor,
                        nt: int,
                        coeff_scale,
                        nx: int,
                        ny: int,
                        query_chunk_size: int,
                        time_samples: int,
                        pde_div_weight: float,
                        pde_vort_weight: float):
    coeffs_norm = _predict_dense_center_coeffs(
        model,
        feature_map=feature_map,
        sharp_map=sharp_map,
        global_feat=global_feat,
        param_feat=param_feat,
        mask=mask,
        nx=nx,
        ny=ny,
        query_chunk_size=query_chunk_size,
    )
    scale = _coeff_scale_tensor(coeff_scale, coeffs_norm)
    coeffs_raw = coeffs_norm * scale

    times = _physics_collocation_times(nt, time_samples, coeffs_raw.device, coeffs_raw.dtype)
    pert_field = reconstruct_query_modes_torch(coeffs_raw, mode_idx, nt=nt, times=times)
    dpert_dt = reconstruct_query_mode_time_derivative_torch(
        coeffs_raw,
        mode_idx,
        nt=nt,
        times=times,
        dt=frame_dt,
    )

    pert_field = _reshape_query_series(pert_field, nx=nx, ny=ny)
    dpert_dt = _reshape_query_series(dpert_dt, nx=nx, ny=ny)
    full_field = pert_field + base_field[:, None]

    u = full_field[:, :, 0]
    v = full_field[:, :, 1]
    du_dt = dpert_dt[:, :, 0]
    dv_dt = dpert_dt[:, :, 1]

    du_dx = _central_diff_x(u, dx=1.0)
    du_dy = _central_diff_y(u, dy=1.0)
    dv_dx = _central_diff_x(v, dx=1.0)
    dv_dy = _central_diff_y(v, dy=1.0)
    div = du_dx + dv_dy

    omega = dv_dx - du_dy
    domega_dt = _central_diff_x(dv_dt, dx=1.0) - _central_diff_y(du_dt, dy=1.0)
    domega_dx = _central_diff_x(omega, dx=1.0)
    domega_dy = _central_diff_y(omega, dy=1.0)
    d2omega_dx2 = _second_diff_x(omega, dx=1.0)
    d2omega_dy2 = _second_diff_y(omega, dy=1.0)

    nu = params_raw[:, 1].to(full_field.dtype)[:, None, None, None]
    vort_res = vorticity_transport_residual(
        domega_dt,
        u,
        domega_dx,
        v,
        domega_dy,
        d2omega_dx2,
        d2omega_dy2,
        nu,
    )

    valid = _fluid_interior_mask(mask).to(full_field.dtype)[:, None].expand(-1, full_field.shape[1], -1, -1)
    div_loss = (div.pow(2) * valid).sum() / (valid.sum() + 1e-8)
    vort_loss = (vort_res.pow(2) * valid).sum() / (valid.sum() + 1e-8)
    total = float(pde_div_weight) * div_loss + float(pde_vort_weight) * vort_loss
    return total, dict(pde_div=float(div_loss.detach().item()), pde_vort=float(vort_loss.detach().item()))


def _inactive_physics_loss(name: str, lam: float, reference: torch.Tensor):
    if lam > 0.0:
        raise ValueError(
            f"{name}={lam} was requested, but this loss is not implemented for the current patch-query setup."
        )
    return reference.new_zeros(())


def _is_compatible_sim(path: Path, n_modes: int, n_ref_frames: int) -> bool:
    try:
        sample = load_simulation(path)
    except Exception:
        return False

    coeffs = sample.get("mode_coeffs")
    ref_field = sample.get("ref_field")
    wake_energy_map = sample.get("wake_energy_map")
    nt = int(sample.get("params", {}).get("nt", 0))
    expected_mode_idx = np.arange(1, n_modes + 1, dtype=np.int32)

    return (
        sample.get("schema_version") == SIM_FORMAT_VERSION
        and sample.get("mode_space") == "velocity"
        and isinstance(coeffs, np.ndarray)
        and coeffs.ndim == 4
        and coeffs.shape[0] >= n_modes
        and coeffs.shape[1] == 2
        and isinstance(ref_field, np.ndarray)
        and ref_field.ndim == 4
        and ref_field.shape[0] >= n_ref_frames
        and ref_field.shape[1] == 2
        and isinstance(wake_energy_map, np.ndarray)
        and wake_energy_map.ndim == 2
        and nt > 2 * n_modes + 2
        and np.array_equal(sample.get("mode_idx", np.array([], dtype=np.int32))[:n_modes], expected_mode_idx)
    )


def _split_paths(paths, seed: int):
    if len(paths) == 1:
        return list(paths), list(paths)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    n_val = max(1, int(0.2 * len(paths)))
    val_ids = order[:n_val]
    train_ids = order[n_val:]
    train_paths = [paths[i] for i in train_ids]
    val_paths = [paths[i] for i in val_ids]
    return train_paths, val_paths


def run_epoch(model, loader, optimizer, device, cfg, training: bool, epoch: int = 0):
    model.train(training)
    losses, errs = [], []
    loss_components = dict(spec=[], amp=[], recon=[], vort=[], mask=[], zero=[], div=[], bc=[], pde=[], pde_div=[], pde_vort=[], eq=[])
    grad_total_norms = []
    grad_layer_stats = {}

    with torch.set_grad_enabled(training):
        for batch in loader:
            geom = batch["geom"].to(device)
            params = batch["params"].to(device)
            params_raw = batch["params_raw"].to(device)
            query_xy = batch["query_xy"].to(device)
            query_ij = batch["query_ij"].to(device)
            query_mask = batch["query_mask"].to(device)
            query_wake = batch["query_wake"].to(device)
            query_div_mask = batch["query_div_mask"].to(device)
            patch_mask = batch["patch_mask"].to(device)
            patch_inlet = batch["patch_inlet"].to(device)
            target = batch["target"].to(device)
            mode_idx = batch["mode_idx"].to(device)
            base_field = batch["base_field"].to(device)
            frame_dt = batch["frame_dt"].to(device)
            full_mask = batch["mask"].to(device)
            nt = int(batch["nt"][0].item())

            weights = _query_weights(query_wake, query_mask, cfg["wake_focus"])
            feature_map, sharp_map, global_feat, param_feat = model.encode_context(geom, params)
            pred = model.decode_queries(feature_map, sharp_map, global_feat, param_feat, query_xy)
            pred = apply_hard_constraints(pred, patch_mask=patch_mask, patch_inlet=patch_inlet)
            target = apply_hard_constraints(target, patch_mask=patch_mask, patch_inlet=patch_inlet)
            pred_center = extract_center_patch(pred, cfg["patch_index"]["center"])
            target_center = extract_center_patch(target, cfg["patch_index"]["center"])

            norm_scale = cfg.get("normalize_mode_loss_scale")
            l_spec = coefficient_loss(pred_center, target_center, weights, cfg.get("mode_weights"), coeff_scale=norm_scale)
            l_amp = amplitude_loss(pred_center, target_center, weights, cfg.get("mode_weights"), coeff_scale=norm_scale)
            if cfg["lam_recon"] > 0.0:
                l_recon = reconstruction_loss(
                    pred_center,
                    target_center,
                    mode_idx,
                    nt=nt,
                    weights=weights,
                    query_chunk_size=cfg["recon_query_chunk_size"],
                )
            else:
                l_recon = pred.new_zeros(())
            if cfg["lam_vort"] > 0.0:
                l_vort = vorticity_reconstruction_loss(
                    pred,
                    target,
                    mode_idx,
                    nt=nt,
                    weights=weights,
                    query_div_mask=query_div_mask,
                    coeff_scale=cfg["coeff_scale"],
                    patch_index=cfg["patch_index"],
                    nx=cfg["nx"],
                    ny=cfg["ny"],
                    query_chunk_size=cfg["recon_query_chunk_size"],
                )
            else:
                l_vort = pred.new_zeros(())

            if cfg["lam_div"] > 0.0:
                l_div = divergence_loss(
                    pred,
                    query_div_mask=query_div_mask,
                    weights=query_mask,
                    coeff_scale=cfg["coeff_scale"],
                    patch_index=cfg["patch_index"],
                    nx=cfg["nx"],
                    ny=cfg["ny"],
                )
            else:
                l_div = pred.new_zeros(())
            if cfg["lam_bc"] > 0.0:
                l_bc = boundary_condition_loss(
                    pred,
                    query_ij=query_ij,
                    query_mask=query_mask,
                    weights=query_mask,
                    coeff_scale=cfg["coeff_scale"],
                    patch_index=cfg["patch_index"],
                    nx=cfg["nx"],
                    ny=cfg["ny"],
                )
            else:
                l_bc = pred.new_zeros(())
            # PDE loss with optional linear warm-start (annealing)
            pde_warmup = cfg.get("pde_warmup_epoch", 0)
            if pde_warmup > 0 and epoch < pde_warmup:
                effective_lam_pde = cfg["lam_pde"] * (epoch / pde_warmup)
            else:
                effective_lam_pde = cfg["lam_pde"]

            if effective_lam_pde > 0.0:
                physics_batch = min(int(cfg["pde_batch_size"]), geom.shape[0])
                l_pde, pde_terms = global_physics_loss(
                    model,
                    feature_map=feature_map[:physics_batch],
                    sharp_map=sharp_map[:physics_batch],
                    global_feat=global_feat[:physics_batch],
                    param_feat=param_feat[:physics_batch],
                    base_field=base_field[:physics_batch],
                    mask=full_mask[:physics_batch],
                    params_raw=params_raw[:physics_batch],
                    frame_dt=frame_dt[:physics_batch],
                    mode_idx=mode_idx[:physics_batch],
                    nt=nt,
                    coeff_scale=cfg["coeff_scale"],
                    nx=cfg["nx"],
                    ny=cfg["ny"],
                    query_chunk_size=cfg["pde_query_chunk_size"],
                    time_samples=cfg["pde_time_samples"],
                    pde_div_weight=cfg["pde_div_weight"],
                    pde_vort_weight=cfg["pde_vort_weight"],
                )
            else:
                l_pde = pred.new_zeros(())
                pde_terms = dict(pde_div=0.0, pde_vort=0.0)
            l_eq = _inactive_physics_loss("lam_eq", cfg["lam_eq"], pred)
            l_mask = pred.new_zeros(())
            l_zero = pred.new_zeros(())

            loss = (
                cfg["lam_spec"] * l_spec
                + cfg["lam_amp"] * l_amp
                + cfg["lam_recon"] * l_recon
                + cfg["lam_vort"] * l_vort
                + cfg["lam_div"] * l_div
                + cfg["lam_bc"] * l_bc
                + effective_lam_pde * l_pde
                + cfg["lam_eq"] * l_eq
            )
            loss = torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)

            if training:
                optimizer.zero_grad()
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip_norm"])
                grad_total_norms.append(float(total_norm))
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        group = name.rsplit(".", 1)[0]
                        grad_layer_stats.setdefault(group, []).append(float(param.grad.data.abs().max().item()))
                optimizer.step()

            losses.append(loss.item())
            errs.append(relative_l2(pred_center, target_center, query_mask, cfg["coeff_scale"]))
            loss_components["spec"].append(l_spec.item())
            loss_components["amp"].append(l_amp.item())
            loss_components["recon"].append(l_recon.item())
            loss_components["vort"].append(l_vort.item())
            loss_components["mask"].append(0.0)
            loss_components["zero"].append(0.0)
            loss_components["div"].append(l_div.item())
            loss_components["bc"].append(l_bc.item())
            loss_components["pde"].append(l_pde.item())
            loss_components["pde_div"].append(pde_terms["pde_div"])
            loss_components["pde_vort"].append(pde_terms["pde_vort"])
            loss_components["eq"].append(l_eq.item())

    avg_components = {k: float(np.mean(v)) if v else 0.0 for k, v in loss_components.items()}
    avg_grad = dict(
        total_norm=float(np.mean(grad_total_norms)) if grad_total_norms else 0.0,
        per_layer={k: float(np.mean(v)) for k, v in grad_layer_stats.items()},
    )
    return float(np.mean(losses)), float(np.mean(errs)), avg_components, avg_grad


def train(cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg["patch_size"] < 3 and (cfg["lam_div"] > 0.0 or cfg["lam_bc"] > 0.0 or cfg["lam_vort"] > 0.0):
        raise ValueError("patch_size must be at least 3 when lam_div, lam_bc, or lam_vort is enabled.")
    if cfg["patch_size"] < 5 and cfg["lam_pde"] > 0.0:
        raise ValueError("patch_size must be at least 5 when lam_pde is enabled.")
    print(f"\n{'=' * 60}")
    print("  Tide Query Operator - training")
    print(f"  Device      : {device}")
    print(f"  Simulations : {cfg['n_sims']}")
    print(f"  Grid        : {cfg['nx']}x{cfg['ny']}  nt={cfg['nt']}")
    print(f"  Modes       : {cfg['n_modes']}")
    print(f"  Queries     : {cfg['queries_per_sample']}")
    print(f"  Patch       : {cfg['patch_size']}x{cfg['patch_size']}")
    print(f"  Epochs      : {cfg['epochs']}")
    print(f"{'=' * 60}\n")

    data_dir = Path(cfg["data_dir"])
    existing = sorted(data_dir.glob("sim_*.pkl"))
    compatible = (
        not cfg["force_regen"]
        and len(existing) >= cfg["n_sims"]
        and all(_is_compatible_sim(p, cfg["n_modes"], cfg["n_ref_frames"]) for p in existing[: cfg["n_sims"]])
    )

    if compatible:
        print(f"Reusing {cfg['n_sims']} compatible simulations in {data_dir}/\n")
        paths = [str(p) for p in existing[: cfg["n_sims"]]]
    else:
        print("Generating simulations...")
        paths = generate_dataset(
            output_dir=str(data_dir),
            n_simulations=cfg["n_sims"],
            n_modes=cfg["n_modes"],
            n_ref_frames=cfg["n_ref_frames"],
            nx=cfg["nx"],
            ny=cfg["ny"],
            nt=cfg["nt"],
            radius=cfg["radius"],
            velocity=cfg["velocity"],
            viscosity=cfg["viscosity"],
            wake_strength=cfg["wake_strength"],
            noise=cfg["noise"],
            vel_range=cfg["vel_range"],
            visc_range=cfg["visc_range"],
            rad_range=cfg["rad_range"],
        )
        print()

    train_paths, val_paths = _split_paths(paths, seed=42)
    train_ds = TideQueryDataset(
        train_paths,
        n_modes=cfg["n_modes"],
        queries_per_sample=cfg["queries_per_sample"],
        patch_size=cfg["patch_size"],
        full_grid=False,
        wake_query_frac=cfg["wake_query_frac"],
        use_sdf=cfg["use_sdf"],
        seed=cfg["seed"],
    )
    stats = train_ds.normalization_state()
    val_ds = TideQueryDataset(
        val_paths,
        n_modes=cfg["n_modes"],
        queries_per_sample=None,
        patch_size=cfg["patch_size"],
        full_grid=True,
        wake_query_frac=cfg["wake_query_frac"],
        use_sdf=cfg["use_sdf"],
        stats=stats,
        seed=cfg["seed"] + 999,
    )
    baselines = compute_baselines(train_ds, val_ds)

    if len(paths) == 1:
        print("Single-simulation mode: train == val (overfit / pipeline check).\n")

    print(
        "Baseline val errors (%)  "
        f"zero={baselines['zero']:.2f}  "
        f"train-mean={baselines['train_mean']:.2f}  "
        f"nearest-neighbor={baselines['nearest_neighbor']:.2f}\n"
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["val_batch_size"], shuffle=False, num_workers=0)

    model = TideQueryOperator(
        n_modes=cfg["n_modes"],
        n_params=3,
        base_ch=cfg["base_ch"],
        in_channels=train_ds.geom_channels,
        n_components=2,
        patch_size=cfg["patch_size"],
        learnable_coord_embed=cfg.get("learnable_coord_embed", False),
        n_mode_groups=cfg.get("n_mode_groups", 1),
    ).to(device)
    print(f"Model parameters : {model.count_params():,}")
    if cfg.get("learnable_coord_embed", False):
        print(f"  Fourier embed  : LEARNABLE (L1 warm start at epoch {model.coord_embed.l1_start_epoch})")
    if cfg.get("n_mode_groups", 1) > 1:
        print(f"  Mode groups    : {model.n_mode_groups} groups, modes/group = {model._modes_per_group}")
    print()

    # Differential LR: coord_embed frequencies train at reduced rate to prevent
    # early collapse (data-space preconditioning from SML course material).
    embed_lr_scale = cfg.get("embed_lr_scale", 0.3)
    if cfg.get("learnable_coord_embed", False):
        embed_params = list(model.coord_embed.parameters())
        embed_ids = {id(p) for p in embed_params}
        other_params = [p for p in model.parameters() if id(p) not in embed_ids]
        optimizer = torch.optim.Adam([
            {"params": other_params},
            {"params": embed_params, "lr": cfg["lr"] * embed_lr_scale},
        ], lr=cfg["lr"], weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)

    if cfg.get("scheduler", "cosine") == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=cfg.get("plateau_patience", 30), min_lr=1e-6,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"], eta_min=1e-6,
        )

    # Per-mode loss weighting (NTK spectral bias counter-measure).
    # Linearly increasing weights: mode 1 gets weight 1.0, last mode gets mode_weight_max.
    mode_weight_max = cfg.get("mode_weight_max", 1.0)
    if mode_weight_max > 1.0:
        mode_weights = torch.linspace(1.0, mode_weight_max, cfg["n_modes"])
        print(f"Mode weights     : {mode_weights.tolist()}")
    else:
        mode_weights = None

    normalize_mode_loss_scale = train_ds.coeff_scale if cfg.get("normalize_mode_loss", False) else None
    if normalize_mode_loss_scale is not None:
        print(f"  Mode-normalized loss: ON (coeff_scale range {normalize_mode_loss_scale.min():.4f} - {normalize_mode_loss_scale.max():.4f})")

    loss_cfg = dict(
        lam_spec=cfg["lam_spec"],
        lam_amp=cfg["lam_amp"],
        lam_recon=cfg["lam_recon"],
        lam_vort=cfg["lam_vort"],
        lam_div=cfg["lam_div"],
        lam_bc=cfg["lam_bc"],
        lam_pde=cfg["lam_pde"],
        lam_eq=cfg["lam_eq"],
        wake_focus=cfg["wake_focus"],
        coeff_scale=train_ds.coeff_scale,
        normalize_mode_loss_scale=normalize_mode_loss_scale,
        patch_index=train_ds.patch_index,
        patch_size=train_ds.patch_size,
        nx=cfg["nx"],
        ny=cfg["ny"],
        recon_query_chunk_size=cfg["recon_query_chunk_size"],
        pde_batch_size=cfg["pde_batch_size"],
        pde_time_samples=cfg["pde_time_samples"],
        pde_query_chunk_size=cfg["pde_query_chunk_size"],
        pde_div_weight=cfg["pde_div_weight"],
        pde_vort_weight=cfg["pde_vort_weight"],
        pde_warmup_epoch=cfg.get("pde_warmup_epoch", 0),
        grad_clip_norm=cfg["grad_clip_norm"],
        mode_weights=mode_weights,
    )

    history = dict(
        train_loss=[],
        val_loss=[],
        train_err=[],
        val_err=[],
        val_evaluated=[],
        loss_components=[],
        grad_stats=[],
        weight_stats=[],
        normalization=stats,
        mode_idx=train_ds.mode_idx.copy(),
        baselines=baselines,
    )
    best_val_err = float("inf")
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'Train Err%':>11}  {'Val Err%':>9}")
    print("-" * 58)

    for epoch in range(1, cfg["epochs"] + 1):
        t_loss, t_err, t_comps, t_grad = run_epoch(model, train_loader, optimizer, device, loss_cfg, training=True, epoch=epoch)
        should_validate = (
            epoch == 1
            or epoch == cfg["epochs"]
            or epoch % max(int(cfg["val_every"]), 1) == 0
        )
        # Fourier embedding L1 penalty (added to training loss each epoch)
        if cfg.get("learnable_coord_embed", False) and hasattr(model.coord_embed, "l1_penalty"):
            l1_pen = model.coord_embed.l1_penalty(epoch)
            if isinstance(l1_pen, torch.Tensor) and l1_pen.item() > 0:
                optimizer.zero_grad()
                l1_pen.backward()
                optimizer.step()

        if should_validate:
            v_loss, v_err, _, _ = run_epoch(model, val_loader, optimizer, device, loss_cfg, training=False)
        else:
            v_loss, v_err = float("nan"), float("nan")

        if cfg.get("scheduler", "cosine") == "plateau":
            if should_validate:
                scheduler.step(v_err)
        else:
            scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_err"].append(t_err)
        history["val_err"].append(v_err)
        history["val_evaluated"].append(bool(should_validate))
        history["loss_components"].append(t_comps)

        history["grad_stats"].append(t_grad)

        weight_summary = {}
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                group = name.rsplit(".", 1)[0]
                data = param.data
                weight_summary[group] = dict(
                    mean=data.mean().item(),
                    std=data.std().item(),
                    absmax=data.abs().max().item(),
                )
        history["weight_stats"].append(weight_summary)

        if epoch % cfg["log_every"] == 0 or epoch == 1:
            if should_validate:
                print(f"{epoch:>6}  {t_loss:>12.6f}  {v_loss:>12.6f}  {t_err:>10.2f}%  {v_err:>8.2f}%")
            else:
                print(f"{epoch:>6}  {t_loss:>12.6f}  {'-':>12}  {t_err:>10.2f}%  {'-':>8}")

        if should_validate and v_err < best_val_err:
            best_val_err = v_err
            torch.save(model.state_dict(), out_dir / "model_best.pt")

    torch.save(model.state_dict(), out_dir / "model_final.pt")
    with open(out_dir / "history.pkl", "wb") as f:
        pickle.dump(history, f)
    with open(out_dir / "metadata.pkl", "wb") as f:
        pickle.dump(
            dict(
                config=cfg,
                normalization=stats,
                mode_idx=train_ds.mode_idx.copy(),
                baselines=baselines,
            ),
            f,
        )

    print(f"\n{'=' * 58}")
    print(f"  Best val error  : {best_val_err:.2f}%")
    print(f"  Saved to        : {out_dir}/")
    print(f"{'=' * 58}\n")

    return model, history, dict(
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_paths=train_paths,
        val_paths=val_paths,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train the query-conditioned Tide operator")
    parser.add_argument("--n_sims", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nt", type=int, default=256)
    parser.add_argument("--n_modes", type=int, default=8)
    parser.add_argument("--n_ref_frames", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=5)
    parser.add_argument("--base_ch", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--val_batch", type=int, default=1)
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--radius", type=float, default=12.0)
    parser.add_argument("--velocity", type=float, default=2.0)
    parser.add_argument("--viscosity", type=float, default=0.1)
    parser.add_argument("--wake_strength", type=float, default=1.5)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--lam_amp", type=float, default=0.25)
    parser.add_argument("--lam_recon", type=float, default=0.25)
    parser.add_argument("--lam_vort", type=float, default=0.0)
    parser.add_argument("--lam_div", type=float, default=0.05)
    parser.add_argument("--lam_bc", type=float, default=0.0)
    parser.add_argument("--lam_pde", type=float, default=0.0)
    parser.add_argument("--lam_eq", type=float, default=0.0)
    parser.add_argument("--wake_focus", type=float, default=4.0)
    parser.add_argument("--wake_query_frac", type=float, default=0.5)
    parser.add_argument("--recon_query_chunk_size", type=int, default=128)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--pde_batch_size", type=int, default=1)
    parser.add_argument("--pde_time_samples", type=int, default=8)
    parser.add_argument("--pde_query_chunk_size", type=int, default=1024)
    parser.add_argument("--pde_div_weight", type=float, default=1.0)
    parser.add_argument("--pde_vort_weight", type=float, default=1.0)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--out_dir", type=str, default="results/")
    parser.add_argument("--force_regen", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    return parser.parse_args()


DEFAULT_CONFIG = dict(
    n_sims=1,
    epochs=300,
    val_every=10,
    nx=64,
    ny=64,
    nt=256,
    n_modes=8,
    n_ref_frames=2,
    patch_size=5,
    base_ch=48,
    lr=1e-3,
    batch_size=4,
    val_batch_size=1,
    queries_per_sample=256,
    recon_query_chunk_size=128,
    radius=12.0,
    velocity=2.0,
    viscosity=0.1,
    wake_strength=1.5,
    noise=0.0,
    data_dir="data/",
    output_dir="results/",
    log_every=20,
    lam_spec=1.0,
    lam_amp=0.25,
    lam_recon=0.25,
    lam_vort=0.0,
    lam_div=0.05,
    lam_bc=0.0,
    lam_pde=0.0,
    lam_eq=0.0,
    wake_focus=4.0,
    wake_query_frac=0.5,
    grad_clip_norm=5.0,
    pde_batch_size=1,
    pde_time_samples=8,
    pde_query_chunk_size=1024,
    pde_div_weight=1.0,
    pde_vort_weight=1.0,
    force_regen=False,
    use_sdf=True,
    seed=0,
    vel_range=(1.0, 3.5),
    visc_range=(0.05, 0.25),
    rad_range=(6.0, 11.0),
)


if __name__ == "__main__":
    args = parse_args()
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(dict(
        n_sims=args.n_sims,
        epochs=args.epochs,
        val_every=args.val_every,
        nx=args.nx,
        ny=args.ny,
        nt=args.nt,
        n_modes=args.n_modes,
        n_ref_frames=args.n_ref_frames,
        patch_size=args.patch_size,
        base_ch=args.base_ch,
        lr=args.lr,
        batch_size=args.batch,
        val_batch_size=args.val_batch,
        queries_per_sample=args.queries,
        radius=args.radius,
        velocity=args.velocity,
        viscosity=args.viscosity,
        wake_strength=args.wake_strength,
        noise=args.noise,
        lam_amp=args.lam_amp,
        lam_recon=args.lam_recon,
        lam_vort=args.lam_vort,
        lam_div=args.lam_div,
        lam_bc=args.lam_bc,
        lam_pde=args.lam_pde,
        lam_eq=args.lam_eq,
        wake_focus=args.wake_focus,
        wake_query_frac=args.wake_query_frac,
        recon_query_chunk_size=args.recon_query_chunk_size,
        grad_clip_norm=args.grad_clip_norm,
        pde_batch_size=args.pde_batch_size,
        pde_time_samples=args.pde_time_samples,
        pde_query_chunk_size=args.pde_query_chunk_size,
        pde_div_weight=args.pde_div_weight,
        pde_vort_weight=args.pde_vort_weight,
        data_dir=args.data_dir,
        output_dir=args.out_dir,
        force_regen=args.force_regen,
        log_every=args.log_every,
    ))

    train(cfg)
