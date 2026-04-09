"""
2D Lattice Boltzmann generator for flow past a cylinder.

The saved dataset now targets phase-aligned velocity-space Fourier modes plus
reference velocity frames for query-conditioned operator learning.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from lbm import (
    initialize_populations,
    lbm_multistep_numpy,
    max_supported_physical_velocity,
    populations_to_velocity_numpy,
)
from spectral import reconstruct_from_modes_numpy


SIM_FORMAT_VERSION = 11
N_LBM_CHANNELS = 9


def _strouhal(Re):
    """Roshko-style Strouhal approximation for a circular cylinder."""
    if Re < 47.0:
        return 0.05 * max(Re / 47.0, 0.0)
    return 0.198 * (1.0 - 19.7 / max(Re, 1e-8))


def _record_schedule(radius, u_lbm, nt, st):
    """
    Choose warmup and frame stride in lattice steps.

    We record every ``frame_stride`` solver steps so the saved sequence spans
    about one shedding period while keeping generation reasonably fast.
    """
    shed_period = (2.0 * radius) / (max(st, 1e-3) * max(u_lbm, 1e-4))
    warmup_steps = max(200, int(1.5 * shed_period))
    frame_stride = max(1, int(round(shed_period / max(nt, 1))))
    return warmup_steps, frame_stride, shed_period


def _effective_reynolds(radius, lbm_u_in, lbm_nu):
    """Effective Reynolds number of the actually simulated lattice flow."""
    return lbm_u_in * (2.0 * radius) / max(lbm_nu, 1e-8)


def _regime_mismatch(radius: float, velocity: float, viscosity: float, lbm_params: dict) -> dict:
    """Summarize how well the requested physical regime matches the lattice regime."""
    re_target = velocity * (2.0 * radius) / max(viscosity, 1e-8)
    u_lbm = float(lbm_params.get("u_lbm", lbm_params.get("lbm_u_in")))
    nu_lbm = float(lbm_params.get("nu_lbm", lbm_params.get("lbm_nu")))
    re_effective = _effective_reynolds(radius, u_lbm, nu_lbm)
    rel_re_error = abs(re_effective - re_target) / max(abs(re_target), 1e-8)
    vmax_supported = max_supported_physical_velocity(viscosity, tau_min=0.51, u_lbm=u_lbm)
    return dict(
        Re_target=float(re_target),
        Re_effective=float(re_effective),
        rel_re_error=float(rel_re_error),
        velocity_max_supported=float(vmax_supported),
        tau_clipped=bool(lbm_params.get("tau_clipped", lbm_params.get("lbm_tau_clipped"))),
    )


def _simulate_populations(nx, ny, radius, velocity, viscosity, nt,
                          wake_strength=1.0, noise=0.0, seed=None):
    """Run the D2Q9 solver and return recorded populations plus metadata."""
    f, mask, sdf, obstacle, lbm_params, center = initialize_populations(
        nx=nx,
        ny=ny,
        radius=radius,
        velocity=velocity,
        viscosity=viscosity,
        wake_strength=wake_strength,
        seed=seed,
    )

    re_target = velocity * (2.0 * radius) / max(viscosity, 1e-8)
    re_effective = _effective_reynolds(radius, lbm_params["u_lbm"], lbm_params["nu_lbm"])
    st_effective = _strouhal(re_effective)
    warmup_steps, frame_stride, shed_period = _record_schedule(
        radius=radius,
        u_lbm=lbm_params["u_lbm"],
        nt=nt,
        st=st_effective,
    )
    total_steps = warmup_steps + nt * frame_stride

    snapshots = np.zeros((nt, N_LBM_CHANNELS, nx, ny), dtype=np.float32)
    snap_id = 0

    for step in range(total_steps):
        f = lbm_multistep_numpy(
            f,
            obstacle=obstacle,
            tau=lbm_params["tau"],
            u_in=lbm_params["u_lbm"],
            n_steps=1,
        )
        if step >= warmup_steps and (step - warmup_steps) % frame_stride == 0 and snap_id < nt:
            snapshots[snap_id] = f
            snap_id += 1

    if not np.all(np.isfinite(snapshots)):
        import warnings
        from lbm import equilibrium_numpy

        warnings.warn(
            f"LBM blew up (NaN/Inf in populations). "
            f"tau={lbm_params['tau']:.4f}, Re_target={re_target:.0f}, "
            f"Re_effective={re_effective:.0f}. Replacing with equilibrium fallback.",
            RuntimeWarning,
        )
        rho0 = np.ones((nx, ny), dtype=np.float32)
        ux0 = np.full((nx, ny), lbm_params["u_lbm"], dtype=np.float32) * mask
        uy0 = np.zeros((nx, ny), dtype=np.float32)
        f_eq = equilibrium_numpy(rho0, ux0, uy0).astype(np.float32)
        snapshots[:] = f_eq[None]

    if noise > 0.0:
        noise_field = noise * np.random.default_rng(seed).standard_normal(snapshots.shape).astype(np.float32)
        snapshots = snapshots + noise_field * mask[None, None]

    meta = dict(
        lbm_tau=float(lbm_params["tau"]),
        lbm_tau_target=float(lbm_params["tau_target"]),
        lbm_tau_clipped=bool(lbm_params["tau_clipped"]),
        lbm_u_in=float(lbm_params["u_lbm"]),
        lbm_nu=float(lbm_params["nu_lbm"]),
        lbm_frame_stride=int(frame_stride),
        lbm_warmup_steps=int(warmup_steps),
        lbm_shed_period=float(shed_period),
        cylinder_center=(float(center[0]), float(center[1])),
    )
    return snapshots.astype(np.float32), mask.astype(np.float32), sdf.astype(np.float32), meta


def _phase_alignment_shift(target_field: np.ndarray, center_x: float, center_y: float,
                           radius: float) -> tuple[int, tuple[int, int]]:
    """
    Pick a deterministic temporal roll so all samples share a common phase.

    We align by the peak positive cross-stream perturbation at a probe just
    behind the cylinder. This removes arbitrary phase offsets introduced by the
    random initialization while keeping the actual dynamics intact.
    """
    _, _, nx, ny = target_field.shape
    probe_x = int(np.clip(round(center_x + 2.0 * radius), 0, nx - 1))
    probe_y = int(np.clip(round(center_y), 0, ny - 1))
    y0 = max(0, probe_y - 1)
    y1 = min(ny, probe_y + 2)
    signal = target_field[:, 1, probe_x, y0:y1].mean(axis=-1)
    if not np.any(np.isfinite(signal)) or np.max(np.abs(signal)) < 1e-8:
        return 0, (probe_x, probe_y)
    return -int(np.argmax(signal)), (probe_x, probe_y)


def _fixed_mode_indices(nt: int, n_modes: int) -> np.ndarray:
    """Use the same positive harmonics for every sample."""
    max_mode = max(1, nt // 2 - 1)
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1")
    if n_modes > max_mode:
        raise ValueError(f"Requested n_modes={n_modes}, but nt={nt} supports at most {max_mode} positive modes.")
    return np.arange(1, n_modes + 1, dtype=np.int32)


def _select_reference_times(nt: int, n_ref_frames: int = 2) -> np.ndarray:
    """Choose a small set of reference phases from the aligned cycle."""
    if n_ref_frames < 1:
        raise ValueError("n_ref_frames must be >= 1")
    if n_ref_frames == 1:
        return np.array([0], dtype=np.int32)
    if n_ref_frames == 2:
        return np.array([0, nt // 4], dtype=np.int32)
    phases = np.linspace(0.0, 0.75, n_ref_frames)
    return np.round(phases * (nt - 1)).astype(np.int32)


def compute_fft(field):
    """Temporal FFT along axis 0, normalized by nt."""
    fft_field = np.fft.fft(field, axis=0) / field.shape[0]
    freqs = np.fft.fftfreq(field.shape[0]) * field.shape[0]
    return fft_field.astype(np.complex64), freqs.astype(np.int32)


def get_dominant_modes(fft_field, freqs, n_modes=5, power_threshold=1e-4):
    """Pick the most energetic non-negative temporal frequencies."""
    reduce_axes = tuple(range(1, fft_field.ndim))
    power = np.mean(np.abs(fft_field) ** 2, axis=reduce_axes)
    pos_power = np.where(freqs >= 0, power, 0.0)

    peak = np.nanmax(pos_power)
    if not np.isfinite(peak) or peak < 1e-30:
        return np.array([0], dtype=np.int32), fft_field[[0]].astype(np.complex64)

    significant = pos_power > power_threshold * peak
    ranked = np.argsort(pos_power)[::-1]
    top_k = ranked[significant[ranked]][:n_modes]
    top_k = np.sort(top_k)

    if len(top_k) == 0:
        return np.array([0], dtype=np.int32), fft_field[[0]].astype(np.complex64)

    return top_k.astype(np.int32), fft_field[top_k].astype(np.complex64)


def prepare_velocity_mode_targets(target_field: np.ndarray,
                                  mask: np.ndarray,
                                  nt: int,
                                  n_modes: int,
                                  n_ref_frames: int = 2) -> dict:
    """Prepare fixed-harmonic velocity targets plus reference frames."""
    fft_field, freqs = compute_fft(target_field)
    mode_idx = _fixed_mode_indices(nt, n_modes)
    ref_times = _select_reference_times(nt, n_ref_frames)
    wake_energy_map = np.sqrt(np.mean(np.sum(target_field ** 2, axis=1), axis=0)).astype(np.float32)
    wake_energy_map *= mask.astype(np.float32)
    fluid = mask > 0.5
    if np.any(fluid):
        ref_scale = float(np.sqrt(np.mean(target_field[:, :, fluid] ** 2)))
    else:
        ref_scale = 1.0
    ref_scale = max(ref_scale, 1e-6)
    return dict(
        fft_field=fft_field,
        freqs=freqs,
        mode_idx=mode_idx,
        mode_coeffs=fft_field[mode_idx].astype(np.complex64),
        ref_times=ref_times.astype(np.int32),
        ref_field=target_field[ref_times].astype(np.float32),
        wake_energy_map=wake_energy_map.astype(np.float32),
        ref_scale=np.float32(ref_scale),
    )


def generate_tide_field(nx=64, ny=64, nt=256,
                        radius=12.0,
                        velocity=2.0,
                        viscosity=0.1,
                        wake_strength=1.0,
                        noise=0.0,
                        seed=None):
    """
    Generate one aligned LBM simulation.

    Returns
    -------
    field        : float32 (nt, 2, nx, ny)   aligned full velocity [u, v]
    target_field : float32 (nt, 2, nx, ny)   aligned perturbation velocity
    base_field   : float32 (2, nx, ny)       mean velocity field
    mask         : float32 (nx, ny)          1=fluid, 0=solid
    sdf          : float32 (nx, ny)          signed distance (+ outside, - inside)
    params       : dict                      physical and LBM parameters
    """
    pop_field, mask, sdf, lbm_meta = _simulate_populations(
        nx=nx,
        ny=ny,
        radius=radius,
        velocity=velocity,
        viscosity=viscosity,
        nt=nt,
        wake_strength=wake_strength,
        noise=noise,
        seed=seed,
    )

    velocity_scale = float(velocity) / max(float(lbm_meta["lbm_u_in"]), 1e-8)
    field = (populations_to_velocity_numpy(pop_field, mask=mask) * velocity_scale).astype(np.float32)
    base_pop_field = pop_field.mean(axis=0).astype(np.float32)
    base_field = (populations_to_velocity_numpy(base_pop_field, mask=mask) * velocity_scale).astype(np.float32)
    target_field = (field - base_field[None]).astype(np.float32)
    target_pop_field = (pop_field - base_pop_field[None]).astype(np.float32)

    center_x, center_y = lbm_meta["cylinder_center"]
    phase_roll, phase_probe = _phase_alignment_shift(target_field, center_x, center_y, radius)
    if phase_roll != 0:
        field = np.roll(field, shift=phase_roll, axis=0)
        target_field = np.roll(target_field, shift=phase_roll, axis=0)
        pop_field = np.roll(pop_field, shift=phase_roll, axis=0)
        target_pop_field = np.roll(target_pop_field, shift=phase_roll, axis=0)

    re_target = velocity * (2.0 * radius) / max(viscosity, 1e-8)
    re_effective = _effective_reynolds(radius, lbm_meta["lbm_u_in"], lbm_meta["lbm_nu"])
    st_target = _strouhal(re_target)
    st_effective = _strouhal(re_effective)
    params = dict(
        nx=nx,
        ny=ny,
        nt=nt,
        radius=float(radius),
        velocity=float(velocity),
        viscosity=float(viscosity),
        wake_strength=float(wake_strength),
        Re=float(re_effective),
        St=float(st_effective),
        Re_target=float(re_target),
        St_target=float(st_target),
        noise=float(noise),
        phase_roll=int(phase_roll),
        phase_probe=(int(phase_probe[0]), int(phase_probe[1])),
        **lbm_meta,
    )
    return (
        field.astype(np.float32),
        target_field.astype(np.float32),
        base_field.astype(np.float32),
        mask.astype(np.float32),
        sdf.astype(np.float32),
        params,
        pop_field.astype(np.float32),
        target_pop_field.astype(np.float32),
        base_pop_field.astype(np.float32),
    )


def save_simulation(path, field, target_field, base_field, mask, sdf, params,
                    fft_field, freqs, mode_idx, mode_coeffs,
                    ref_field, ref_times, wake_energy_map, ref_scale,
                    pop_field=None, target_pop_field=None, base_pop_field=None):
    data = dict(
        schema_version=SIM_FORMAT_VERSION,
        field=field.astype(np.float32),
        target_field=target_field.astype(np.float32),
        base_field=base_field.astype(np.float32),
        field_components=("u", "v"),
        pop_field=None if pop_field is None else pop_field.astype(np.float32),
        target_pop_field=None if target_pop_field is None else target_pop_field.astype(np.float32),
        base_pop_field=None if base_pop_field is None else base_pop_field.astype(np.float32),
        population_components=tuple(f"f{i}" for i in range(N_LBM_CHANNELS)),
        mask=mask.astype(np.float32),
        sdf=sdf.astype(np.float32),
        params=params,
        fft_field=fft_field.astype(np.complex64),
        freqs=freqs.astype(np.int32),
        mode_idx=mode_idx.astype(np.int32),
        mode_coeffs=mode_coeffs.astype(np.complex64),
        mode_space="velocity",
        ref_field=ref_field.astype(np.float32),
        ref_times=ref_times.astype(np.int32),
        wake_energy_map=wake_energy_map.astype(np.float32),
        ref_scale=np.float32(ref_scale),
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_simulation(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def generate_dataset(output_dir="data/",
                     n_simulations=1,
                     n_modes=5,
                     n_ref_frames=2,
                     nx=64, ny=64, nt=256,
                     radius=12.0,
                     velocity=2.0,
                     viscosity=0.1,
                     wake_strength=1.0,
                     noise=0.0,
                     vel_range=(1.0, 3.5),
                     visc_range=(0.05, 0.25),
                     rad_range=(8.0, 16.0),
                     seed=0,
                     strict_regime=True,
                     max_sampling_attempts=2000,
                     max_re_mismatch=0.10):
    """Generate simulation files and return their paths."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    attempt = 0
    while len(paths) < n_simulations:
        i = len(paths)
        attempt += 1
        if attempt > max_sampling_attempts:
            raise RuntimeError(
                "Could not generate enough unclipped simulations within the requested parameter ranges. "
                "The requested (velocity, viscosity) space is outside the representable LBM regime on this grid. "
                "Reduce velocity, increase viscosity, or relax strict_regime."
            )

        if n_simulations == 1:
            vel_i = velocity
            visc_i = viscosity
            rad_i = radius
        else:
            vel_i = rng.uniform(*vel_range)
            visc_i = rng.uniform(*visc_range)
            rad_i = rng.uniform(*rad_range)

        result = generate_tide_field(
            nx=nx,
            ny=ny,
            nt=nt,
            radius=rad_i,
            velocity=vel_i,
            viscosity=visc_i,
            wake_strength=wake_strength,
            noise=noise,
            seed=seed + i,
        )
        field, target_field, base_field, mask, sdf, params, pop_field, target_pop_field, base_pop_field = result
        regime = _regime_mismatch(rad_i, vel_i, visc_i, params)

        if strict_regime and (regime["tau_clipped"] or regime["rel_re_error"] > max_re_mismatch):
            if n_simulations == 1:
                raise ValueError(
                    f"Requested regime is not representable on the current lattice setup: "
                    f"U={vel_i:.3f}, nu={visc_i:.3f}, R={rad_i:.2f}, "
                    f"Re_target={regime['Re_target']:.1f}, Re_effective={regime['Re_effective']:.1f}, "
                    f"tau clipped at {params['lbm_tau']:.4f}. "
                    f"For nu={visc_i:.3f}, the no-clipping velocity upper bound is about "
                    f"{regime['velocity_max_supported']:.3f}."
                )
            continue

        target_info = prepare_velocity_mode_targets(
            target_field=target_field,
            mask=mask,
            nt=nt,
            n_modes=n_modes,
            n_ref_frames=n_ref_frames,
        )

        path = output_dir / f"sim_{i:04d}.pkl"
        save_simulation(
            path,
            field,
            target_field,
            base_field,
            mask,
            sdf,
            params,
            target_info["fft_field"],
            target_info["freqs"],
            target_info["mode_idx"],
            target_info["mode_coeffs"],
            target_info["ref_field"],
            target_info["ref_times"],
            target_info["wake_energy_map"],
            target_info["ref_scale"],
            pop_field=pop_field,
            target_pop_field=target_pop_field,
            base_pop_field=base_pop_field,
        )
        paths.append(str(path))

        print(
            f"  [{i + 1:>{len(str(n_simulations))}}/{n_simulations}]  "
            f"U={vel_i:.2f}  nu={visc_i:.3f}  R={rad_i:.1f}  "
            f"Re={params['Re']:.0f} (target {params['Re_target']:.0f})  "
            f"St={params['St']:.3f}  tau={params['lbm_tau']:.4f} "
            f"-> fixed modes at k={target_info['mode_idx'].tolist()}"
        )

    return paths


def main():
    print("Generating 1 simulation (self-test)...")
    paths = generate_dataset(
        n_simulations=1,
        n_modes=5,
        n_ref_frames=2,
        output_dir="data/",
        velocity=2.0,
        viscosity=0.1,
    )
    sample = load_simulation(paths[0])

    target_field = sample["target_field"]
    recon = reconstruct_from_modes_numpy(sample["mode_coeffs"], sample["mode_idx"], target_field.shape[0])
    rel_err = np.linalg.norm(recon - target_field) / (np.linalg.norm(target_field) + 1e-8)

    print(
        f"  Re = {sample['params']['Re']:.0f} "
        f"(target {sample['params']['Re_target']:.0f}),  "
        f"St = {sample['params']['St']:.3f}, tau = {sample['params']['lbm_tau']:.4f}, "
        f"phase_roll = {sample['params']['phase_roll']}"
    )
    print(f"  Velocity reconstruction relative error ({len(sample['mode_idx'])} modes): {rel_err * 100:.4f}%")
    print("  [PASS]" if rel_err < 0.60 else "  [WARN] error > 60%")


if __name__ == "__main__":
    main()
