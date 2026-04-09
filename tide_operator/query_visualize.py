"""Diagnostic plots for the query-conditioned Tide Fourier operator."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from lbm import populations_to_velocity_numpy
from query_dataset import make_inference_inputs
from query_model import apply_hard_constraints, extract_center_patch
from spectral import reconstruct_from_modes_numpy


def _blue_wake_cmap():
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(color="black")
    return cmap


def _vorticity_cmap():
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="black")
    return cmap


def _style_light_axes(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("#b0b0b0")


def _wake_intensity(field_t, inflow):
    du = field_t[0] - inflow
    dv = field_t[1]
    return np.sqrt(np.maximum(du ** 2 + dv ** 2, 0.0))


def _wake_image(field_t, mask, inflow):
    wake = _wake_intensity(field_t, inflow)
    return np.ma.array(wake.T, mask=(mask <= 0.5).T)


def _vorticity(field_t):
    u = field_t[0]
    v = field_t[1]
    dv_dx = np.zeros_like(v)
    du_dy = np.zeros_like(u)
    dv_dx[1:-1, :] = 0.5 * (v[2:, :] - v[:-2, :])
    du_dy[:, 1:-1] = 0.5 * (u[:, 2:] - u[:, :-2])
    return dv_dx - du_dy


def _vorticity_image(field_t, mask):
    vort = _vorticity(field_t)
    return np.ma.array(vort.T, mask=(mask <= 0.5).T)


def _display_field_and_inflow(sim_data):
    if "target_field" in sim_data:
        return sim_data["target_field"], 0.0
    return sim_data["field"], float(sim_data["params"]["velocity"])


def _re_title(params):
    re_eff = params.get("Re", 0.0)
    re_tgt = params.get("Re_target", re_eff)
    if abs(re_eff - re_tgt) > 1e-6:
        return f"Re={re_eff:.0f} (target {re_tgt:.0f})"
    return f"Re={re_eff:.0f}"


def _reconstruct_display_field(sim_data, mode_coeffs, mode_idx, nt):
    """Reconstruct the display-space field from stored mode coefficients."""
    recon = reconstruct_from_modes_numpy(mode_coeffs, mode_idx, nt)
    if recon.shape[1] == 9 and sim_data.get("base_pop_field") is not None:
        full_pop = recon + sim_data["base_pop_field"][None]
        velocity_scale = float(sim_data["params"]["velocity"]) / max(float(sim_data["params"]["lbm_u_in"]), 1e-8)
        vel = populations_to_velocity_numpy(full_pop, mask=sim_data["mask"]) * velocity_scale
        return vel - sim_data["base_field"][None]
    return recon


def _masked_scalar_image(image, mask):
    return np.ma.array(image.T, mask=(mask <= 0.5).T)


def _vector_mode_amplitude(coeffs):
    return np.sqrt(np.sum(np.abs(coeffs) ** 2, axis=0))


def _predict_full_mode_map(model, sim_data, base_dataset, device: str, chunk_size: int = 1024):
    inputs = make_inference_inputs(
        sim_data,
        param_mean=base_dataset.param_mean,
        param_std=base_dataset.param_std,
        use_sdf=base_dataset.use_sdf,
    )
    geom = inputs["geom"].to(device)
    params = inputs["params"].to(device)
    query_xy = inputs["query_xy"].to(device)
    query_mask = inputs["query_mask"].to(device)
    query_ij = inputs["query_ij"].to(device)

    preds = []
    with torch.no_grad():
        for start in range(0, query_xy.shape[1], chunk_size):
            end = min(query_xy.shape[1], start + chunk_size)
            patch_pred = model(
                geom,
                params,
                query_xy[:, start:end],
                query_mask=query_mask[:, start:end],
            )
            center = extract_center_patch(patch_pred, model.center_patch_index)
            center = center * query_mask[:, start:end, None, None, None]
            inlet = (query_ij[start:end, 0] == 0).to(center.dtype)[None, :, None, None, None]
            center = center * (1.0 - inlet)
            preds.append(center.cpu())

    pred = torch.cat(preds, dim=1)[0].numpy()
    pred = base_dataset.denorm_coeffs(pred)

    nx = int(sim_data["params"]["nx"])
    ny = int(sim_data["params"]["ny"])
    pred = pred.reshape(nx, ny, pred.shape[1], pred.shape[2], pred.shape[3])
    pred = pred.transpose(2, 3, 4, 0, 1)
    return pred


def _model_reconstruction(model, dataset, sample_idx: int, device: str):
    model.eval()
    base_ds = dataset.dataset if hasattr(dataset, "dataset") else dataset
    real_idx = dataset.indices[sample_idx] if hasattr(dataset, "indices") else sample_idx
    sim = base_ds.samples[real_idx]

    pred_coeffs = _predict_full_mode_map(model, sim, base_ds, device)
    coeffs_complex = pred_coeffs[:, :, 0] + 1j * pred_coeffs[:, :, 1]
    nt = int(sim["params"]["nt"])
    mode_idx = np.asarray(sim["mode_idx"], dtype=np.int64)[:coeffs_complex.shape[0]]
    recon = _reconstruct_display_field(sim, coeffs_complex, mode_idx, nt)
    return recon, sim


def plot_simulation_overview(sim_path: str, save_path: str = None):
    with open(sim_path, "rb") as f:
        s = pickle.load(f)

    field, inflow = _display_field_and_inflow(s)
    mode_idx = s["mode_idx"]
    p = s["params"]
    nt, _, nx, ny = field.shape
    fft_field = np.fft.fft(field, axis=0) / nt
    recon = _reconstruct_display_field(s, s["mode_coeffs"], mode_idx, nt)
    mask = s["mask"]
    wake_cmap = _blue_wake_cmap()

    fig, axes = plt.subplots(3, 4, figsize=(17, 11))
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        _style_light_axes(ax)

    title = (
        f"Simulation | U={p['velocity']:.2f}  nu={p['viscosity']:.3f} "
        f"R={p['radius']:.1f}  {_re_title(p)}  nx={nx}  nt={nt}"
    )
    fig.suptitle(title, color="black", fontsize=13, y=1.01)

    t_snaps = [0, nt // 4, nt // 2, 3 * nt // 4]
    vmax = max(float(np.max(_wake_intensity(field[t], inflow))) for t in t_snaps) or 1.0
    for col, t in enumerate(t_snaps):
        ax = axes[0, col]
        im = ax.imshow(_wake_image(field[t], mask, inflow), cmap=wake_cmap, vmin=0.0, vmax=vmax,
                       origin="lower", interpolation="bilinear")
        ax.set_title(f"Wake intensity at t={t}", color="black", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")

    for col in range(4):
        ax = axes[1, col]
        if col < len(mode_idx):
            amp = _masked_scalar_image(_vector_mode_amplitude(fft_field[int(mode_idx[col])]), mask)
            im = ax.imshow(amp, cmap=wake_cmap, origin="lower", interpolation="bilinear")
            ax.set_title(f"|FFT| mode k={int(mode_idx[col])}", color="black", fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
        ax.axis("off")

    cx = min(nx - 1, nx // 2 + nx // 4)
    cy = ny // 2
    signal_u = field[:, 0, cx, cy]
    signal_v = field[:, 1, cx, cy]

    ax = axes[2, 0]
    ax.plot(signal_u, color="#1f77b4", linewidth=1.2, label="u")
    ax.plot(signal_v, color="#4c78a8", linewidth=1.2, linestyle="--", label="v")
    ax.set_title(f"Wake signal @ ({cx}, {cy})", color="black", fontsize=10)
    ax.set_xlabel("t", color="black", fontsize=9)
    ax.grid(True, color="#d0d0d0", linewidth=0.5)
    ax.legend(facecolor="white", edgecolor="#b0b0b0", fontsize=8)

    power = np.abs(fft_field[:, 0, cx, cy]) ** 2
    freqs = s["freqs"]
    pos = freqs >= 0
    ax = axes[2, 1]
    ax.stem(freqs[pos][:nt // 2], power[pos][:nt // 2], markerfmt="C0o", linefmt="C0-", basefmt="#d0d0d0")
    ax.set_title("Power spectrum of wake-u", color="black", fontsize=10)
    ax.set_xlabel("k", color="black", fontsize=9)
    ax.grid(True, color="#d0d0d0", linewidth=0.5)

    recon_u = recon[:, 0, cx, cy]
    ax = axes[2, 2]
    ax.plot(signal_u, color="#1f77b4", linewidth=1.5, label="Original wake-u")
    ax.plot(recon_u, color="#4c78a8", linewidth=1.2, linestyle="--", label="Recon wake-u")
    rel_err = np.linalg.norm(recon[:, :, cx, cy] - field[:, :, cx, cy]) / (
        np.linalg.norm(field[:, :, cx, cy]) + 1e-8
    ) * 100
    ax.set_title(f"Point recon err = {rel_err:.2f}%", color="black", fontsize=10)
    ax.grid(True, color="#d0d0d0", linewidth=0.5)
    ax.legend(facecolor="white", edgecolor="#b0b0b0", fontsize=8)

    ax = axes[2, 3]
    im = ax.imshow(s["sdf"].T, cmap="Greys", origin="lower", interpolation="bilinear")
    ax.set_title("Signed Distance Field", color="black", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved -> {save_path}")
    plt.close(fig)


def plot_training_history(history_path: str, save_path: str = None):
    with open(history_path, "rb") as f:
        h = pickle.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")
    for ax in axes:
        _style_light_axes(ax)

    epochs = range(1, len(h["train_loss"]) + 1)
    axes[0].semilogy(epochs, h["train_loss"], color="#1f77b4", label="Train")
    axes[0].semilogy(epochs, h["val_loss"], color="#4c78a8", label="Val")
    axes[0].set_title("Loss (log scale)", color="black")
    axes[0].set_xlabel("Epoch", color="black")
    axes[0].legend(facecolor="white", edgecolor="#b0b0b0")
    axes[0].grid(True, color="#d0d0d0", linewidth=0.5)

    axes[1].plot(epochs, h["val_err"], color="#1f77b4", label="Val error %")
    final_err = h["val_err"][-1]
    axes[1].set_title(f"Relative L2 Error (final: {final_err:.2f}%)", color="black")
    axes[1].set_xlabel("Epoch", color="black")
    axes[1].legend(facecolor="white", edgecolor="#b0b0b0")
    axes[1].grid(True, color="#d0d0d0", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved -> {save_path}")
    plt.close(fig)


def plot_prediction_vs_truth(model, dataset, sample_idx: int = 0,
                             device: str = "cpu", save_path: str = None):
    recon_model, sim = _model_reconstruction(model, dataset, sample_idx, device)
    field, inflow = _display_field_and_inflow(sim)
    mask = sim["mask"]
    nt = field.shape[0]
    t = nt // 4

    true_img = _wake_image(field[t], mask, inflow)
    pred_img = _wake_image(recon_model[t], mask, inflow)
    vmax = max(float(true_img.max()), float(pred_img.max()), 1e-8)
    true_vort = _vorticity_image(field[t], mask)
    pred_vort = _vorticity_image(recon_model[t], mask)
    vort_lim = max(float(np.max(np.abs(true_vort))), float(np.max(np.abs(pred_vort))), 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    wake_cmap = _blue_wake_cmap()
    vort_cmap = _vorticity_cmap()
    for ax in axes.ravel():
        _style_light_axes(ax)

    im0 = axes[0, 0].imshow(true_img, cmap=wake_cmap, vmin=0.0, vmax=vmax, origin="lower", interpolation="bilinear")
    axes[0, 0].set_title(f"True wake magnitude (t={t})", color="black", fontsize=10)
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(pred_img, cmap=wake_cmap, vmin=0.0, vmax=vmax, origin="lower", interpolation="bilinear")
    axes[0, 1].set_title("Pred wake magnitude", color="black", fontsize=10)
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(true_vort, cmap=vort_cmap, vmin=-vort_lim, vmax=vort_lim, origin="lower", interpolation="bilinear")
    axes[1, 0].set_title("True vorticity", color="black", fontsize=10)
    axes[1, 0].axis("off")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    im3 = axes[1, 1].imshow(pred_vort, cmap=vort_cmap, vmin=-vort_lim, vmax=vort_lim, origin="lower", interpolation="bilinear")
    axes[1, 1].set_title("Pred vorticity", color="black", fontsize=10)
    axes[1, 1].axis("off")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    rel = np.linalg.norm(recon_model - field) / (np.linalg.norm(field) + 1e-8) * 100
    p = sim["params"]
    fig.suptitle(
        f"Prediction vs Truth | U={p['velocity']:.2f}  nu={p['viscosity']:.3f}  "
        f"R={p['radius']:.1f}  {_re_title(p)}  err={rel:.2f}%",
        color="black",
        fontsize=12,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved -> {save_path}")
    plt.close(fig)


def animate_reconstruction(sim_path: str, model=None, dataset=None,
                           sample_idx: int = 0, device: str = "cpu",
                           save_path: str = None, fps: int = 20,
                           title_extra: str = ""):
    with open(sim_path, "rb") as f:
        s = pickle.load(f)

    field, inflow = _display_field_and_inflow(s)
    mode_idx = s["mode_idx"]
    nt = field.shape[0]
    recon_gt = _reconstruct_display_field(s, s["mode_coeffs"], mode_idx, nt)
    mask = s["mask"]

    show_model = model is not None and dataset is not None
    if show_model:
        recon_model, _ = _model_reconstruction(model, dataset, sample_idx, device)

    vmax = max(float(np.max(_wake_intensity(field[t], inflow))) for t in range(nt)) or 1.0
    n_panels = 3 if show_model else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels + 1, 5))
    fig.patch.set_facecolor("white")
    wake_cmap = _blue_wake_cmap()
    for ax in axes:
        ax.set_facecolor("white")
        ax.axis("off")

    im_kwargs = dict(cmap=wake_cmap, vmin=0.0, vmax=vmax,
                     origin="lower", interpolation="bilinear", animated=True)
    ims_list = [
        axes[0].imshow(_wake_image(field[0], mask, inflow), **im_kwargs),
        axes[1].imshow(_wake_image(recon_gt[0], mask, inflow), **im_kwargs),
    ]
    axes[0].set_title("True wake", color="black", pad=4)
    axes[1].set_title(f"GT recon ({len(mode_idx)} modes)", color="black", pad=4)

    if show_model:
        ims_list.append(axes[2].imshow(_wake_image(recon_model[0], mask, inflow), **im_kwargs))
        axes[2].set_title("Model recon", color="black", pad=4)

    p = s["params"]
    sup = f"U={p['velocity']:.2f}  nu={p['viscosity']:.3f}  R={p['radius']:.1f}  {_re_title(p)}"
    if title_extra:
        sup = f"{title_extra} | {sup}"
    time_title = fig.suptitle(f"{sup} | t = 0", color="black", fontsize=12)

    def update(t):
        ims_list[0].set_data(_wake_image(field[t], mask, inflow))
        ims_list[1].set_data(_wake_image(recon_gt[t], mask, inflow))
        if show_model:
            ims_list[2].set_data(_wake_image(recon_model[t], mask, inflow))
        time_title.set_text(f"{sup} | t = {t:>4}")
        return ims_list

    ani = animation.FuncAnimation(fig, update, frames=nt, interval=1000 // fps, blit=True)
    if save_path:
        ext = Path(save_path).suffix.lower()
        if ext == ".gif":
            ani.save(save_path, writer="pillow", fps=fps)
        else:
            ani.save(save_path, fps=fps)
        print(f"Animation saved -> {save_path}")
    plt.close(fig)
    return ani


def animate_reconstruction_from_data(sim_data: dict,
                                     model,
                                     train_dataset,
                                     device: str = "cpu",
                                     save_path: str = None,
                                     fps: int = 20,
                                     title_extra: str = ""):
    model.eval()
    base_ds = train_dataset.dataset if hasattr(train_dataset, "dataset") else train_dataset

    field, inflow = _display_field_and_inflow(sim_data)
    mode_idx = sim_data["mode_idx"]
    p = sim_data["params"]
    nt = field.shape[0]
    recon_gt = _reconstruct_display_field(sim_data, sim_data["mode_coeffs"], mode_idx, nt)
    mask = sim_data["mask"]

    pred_coeffs = _predict_full_mode_map(model, sim_data, base_ds, device)
    coeffs_complex = pred_coeffs[:, :, 0] + 1j * pred_coeffs[:, :, 1]
    recon_model = _reconstruct_display_field(sim_data, coeffs_complex, mode_idx, nt)
    rel_err = (
        np.linalg.norm((recon_model - field) * mask[None, None])
        / (np.linalg.norm(field * mask[None, None]) + 1e-8)
        * 100
    )

    vmax = max(float(np.max(_wake_intensity(field[t], inflow))) for t in range(nt)) or 1.0
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("white")
    wake_cmap = _blue_wake_cmap()
    for ax in axes:
        ax.set_facecolor("white")
        ax.axis("off")

    im_kwargs = dict(cmap=wake_cmap, vmin=0.0, vmax=vmax,
                     origin="lower", interpolation="bilinear", animated=True)
    ims_list = [
        axes[0].imshow(_wake_image(field[0], mask, inflow), **im_kwargs),
        axes[1].imshow(_wake_image(recon_gt[0], mask, inflow), **im_kwargs),
        axes[2].imshow(_wake_image(recon_model[0], mask, inflow), **im_kwargs),
    ]
    axes[0].set_title("True wake", color="black", pad=4)
    axes[1].set_title(f"GT recon ({len(mode_idx)} modes)", color="black", pad=4)
    axes[2].set_title(f"Model recon (err={rel_err:.1f}%)", color="black", pad=4)

    sup = f"U={p['velocity']:.2f}  nu={p['viscosity']:.3f}  R={p['radius']:.1f}  {_re_title(p)}"
    if title_extra:
        sup = f"{title_extra} | {sup}"
    time_title = fig.suptitle(f"{sup} | t = 0", color="black", fontsize=12)

    def update(t):
        ims_list[0].set_data(_wake_image(field[t], mask, inflow))
        ims_list[1].set_data(_wake_image(recon_gt[t], mask, inflow))
        ims_list[2].set_data(_wake_image(recon_model[t], mask, inflow))
        time_title.set_text(f"{sup} | t = {t:>4}")
        return ims_list

    ani = animation.FuncAnimation(fig, update, frames=nt, interval=1000 // fps, blit=True)
    if save_path:
        ext = Path(save_path).suffix.lower()
        if ext == ".gif":
            ani.save(save_path, writer="pillow", fps=fps)
        else:
            ani.save(save_path, fps=fps)
        print(f"Animation saved -> {save_path} (reconstruction err = {rel_err:.1f}%)")
    plt.close(fig)
    return ani, rel_err


def plot_loss_breakdown(history_path: str, save_path: str = None):
    with open(history_path, "rb") as f:
        h = pickle.load(f)

    comps = h.get("loss_components", [])
    if not comps:
        print("No loss_components in history; skipping loss breakdown plot.")
        return

    epochs = range(1, len(comps) + 1)
    keys = [k for k in comps[0].keys() if any(c.get(k, 0.0) != 0.0 for c in comps)]
    if not keys:
        print("All loss components are zero; skipping loss breakdown plot.")
        return

    colors = {
        "spec": "#1f77b4",
        "amp": "#ff7f0e",
        "recon": "#2ca02c",
        "vort": "#9467bd",
        "mask": "#9467bd",
        "zero": "#8c564b",
        "div": "#d62728",
        "bc": "#17becf",
        "pde": "#e377c2",
        "pde_div": "#bcbd22",
        "pde_vort": "#8c564b",
        "eq": "#7f7f7f",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor("white")
    for ax in axes:
        _style_light_axes(ax)

    for key in keys:
        vals = [c.get(key, 0.0) for c in comps]
        axes[0].semilogy(epochs, vals, label=key, color=colors.get(key), linewidth=1.5, alpha=0.85)
    axes[0].set_title("Loss Components (log scale)", color="black")
    axes[0].set_xlabel("Epoch", color="black")
    axes[0].set_ylabel("Loss value", color="black")
    axes[0].legend(facecolor="white", edgecolor="#b0b0b0", fontsize=8, ncol=2, loc="upper right")
    axes[0].grid(True, color="#d0d0d0", linewidth=0.5)

    totals = np.array([sum(c.get(k, 0.0) for k in keys) for c in comps])
    totals = np.maximum(totals, 1e-12)
    bottom = np.zeros(len(comps))
    for key in keys:
        vals = np.array([c.get(key, 0.0) for c in comps]) / totals * 100.0
        axes[1].fill_between(epochs, bottom, bottom + vals, label=key, color=colors.get(key), alpha=0.7)
        bottom += vals
    axes[1].set_title("Loss Component Share (%)", color="black")
    axes[1].set_xlabel("Epoch", color="black")
    axes[1].set_ylabel("Percentage", color="black")
    axes[1].set_ylim(0, 100)
    axes[1].legend(facecolor="white", edgecolor="#b0b0b0", fontsize=8, ncol=2, loc="center right")
    axes[1].grid(True, color="#d0d0d0", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved -> {save_path}")
    plt.close(fig)


def plot_gradient_flow(history_path: str, save_path: str = None):
    with open(history_path, "rb") as f:
        h = pickle.load(f)

    grad_stats = h.get("grad_stats", [])
    if not grad_stats:
        print("No grad_stats in history; skipping gradient flow plot.")
        return

    epochs = range(1, len(grad_stats) + 1)
    total_norms = [g["total_norm"] for g in grad_stats]

    all_layers = set()
    for g in grad_stats:
        all_layers.update(g.get("per_layer", {}).keys())
    layer_max = {layer: max(g.get("per_layer", {}).get(layer, 0.0) for g in grad_stats) for layer in all_layers}
    top_layers = sorted(all_layers, key=lambda name: layer_max[name], reverse=True)[:10]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor("white")
    for ax in axes:
        _style_light_axes(ax)

    axes[0].semilogy(epochs, total_norms, color="#1f77b4", linewidth=1.5)
    axes[0].set_title("Total Gradient Norm", color="black")
    axes[0].set_xlabel("Epoch", color="black")
    axes[0].set_ylabel("L2 norm", color="black")
    axes[0].grid(True, color="#d0d0d0", linewidth=0.5)

    cmap = plt.cm.tab10
    for i, layer in enumerate(top_layers):
        vals = [g.get("per_layer", {}).get(layer, 0.0) for g in grad_stats]
        short_name = layer.split(".")[-1] if "." in layer else layer
        axes[1].semilogy(epochs, vals, label=short_name, color=cmap(i % 10), linewidth=1.2, alpha=0.8)
    axes[1].set_title("Per-Layer Max Gradient Norm (top 10)", color="black")
    axes[1].set_xlabel("Epoch", color="black")
    axes[1].set_ylabel("Max |grad|", color="black")
    axes[1].legend(facecolor="white", edgecolor="#b0b0b0", fontsize=7, ncol=2, loc="upper right")
    axes[1].grid(True, color="#d0d0d0", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved -> {save_path}")
    plt.close(fig)


def plot_weight_distributions(history_path: str, save_path: str = None):
    with open(history_path, "rb") as f:
        h = pickle.load(f)

    weight_stats = h.get("weight_stats", [])
    if not weight_stats:
        print("No weight_stats in history; skipping weight distribution plot.")
        return

    epochs = range(1, len(weight_stats) + 1)
    all_layers = set()
    for ws in weight_stats:
        all_layers.update(ws.keys())
    layers = sorted(all_layers)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("white")
    for ax in axes:
        _style_light_axes(ax)

    cmap = plt.cm.tab20
    titles = ["Weight Mean", "Weight Std", "Weight |Max|"]
    keys = ["mean", "std", "absmax"]

    for panel, (stat_key, title) in enumerate(zip(keys, titles)):
        ax = axes[panel]
        for i, layer in enumerate(layers):
            vals = [ws.get(layer, {}).get(stat_key, 0.0) for ws in weight_stats]
            short_name = layer.split(".")[-1] if "." in layer else layer
            if panel == 2:
                ax.semilogy(epochs, vals, label=short_name, color=cmap(i % 20), linewidth=1.2, alpha=0.8)
            else:
                ax.plot(epochs, vals, label=short_name, color=cmap(i % 20), linewidth=1.2, alpha=0.8)
        ax.set_title(title, color="black")
        ax.set_xlabel("Epoch", color="black")
        ax.grid(True, color="#d0d0d0", linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(layers)),
               facecolor="white", edgecolor="#b0b0b0", fontsize=7,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved -> {save_path}")
    plt.close(fig)
