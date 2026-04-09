"""
PyTorch dataset for the query-conditioned Tide operator.

Each sample exposes:
  geom            : (4 + 2*n_ref, nx, ny)    [SDF_norm, mask, X, Y, ref frames...]
  params          : (3,)                     [velocity, viscosity, radius], z-normalized
  params_raw      : (3,)                     raw physical values
  query_xy        : (Q, 2)                   normalized query coordinates in [-1, 1]
  query_ij        : (Q, 2)                   integer grid indices [ix, iy]
  query_mask      : (Q,)                     1=fluid, 0=solid at the query center
  query_wake      : (Q,)                     temporal RMS wake intensity, normalized by sample mean
        query_div_mask  : (Q,)                     1 if a central divergence stencil is valid
        patch_mask      : (Q, P)                   fluid mask over the local patch
        patch_inlet     : (Q, P)                   1 where the patch lies on the inlet boundary
        target          : (Q, P, n_modes, 2, 2)    patch velocity coefficients [u/v, real/imag], normalized
        mode_idx        : (n_modes,)               shared temporal frequency bins
        base_field      : (2, nx, ny)              mean velocity field
        frame_dt        : scalar                   physical time between stored frames
        mask            : (nx, ny)                 full fluid mask
        nt              : scalar                   number of time steps
"""

from __future__ import annotations

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from generate import SIM_FORMAT_VERSION


def make_coordinate_channels(nx: int, ny: int):
    """Return normalized coordinate channels and flattened query coordinates."""
    x_coords = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    y_coords = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
    coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)
    ii, jj = np.meshgrid(np.arange(nx, dtype=np.int64), np.arange(ny, dtype=np.int64), indexing="ij")
    query_ij = np.stack([ii.reshape(-1), jj.reshape(-1)], axis=-1)
    return X, Y, coords.astype(np.float32), query_ij.astype(np.int64)


def make_patch_offsets(patch_size: int):
    """Return flattened patch offsets and key neighbor indices for a square stencil."""
    patch_size = int(patch_size)
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer.")
    radius = patch_size // 2
    dx, dy = np.meshgrid(
        np.arange(-radius, radius + 1, dtype=np.int64),
        np.arange(-radius, radius + 1, dtype=np.int64),
        indexing="ij",
    )
    offsets = np.stack([dx.reshape(-1), dy.reshape(-1)], axis=-1)
    lookup = {tuple(int(v) for v in off): idx for idx, off in enumerate(offsets)}
    return offsets, dict(
        center=lookup[(0, 0)],
        left=lookup[(-1, 0)],
        right=lookup[(1, 0)],
        down=lookup[(0, -1)],
        up=lookup[(0, 1)],
    )


def build_geom_channels(sample: dict, use_sdf: bool = True) -> np.ndarray:
    """Build the geometry/reference tensor used by the query-conditioned model."""
    p = sample["params"]
    nx = int(p["nx"])
    ny = int(p["ny"])
    X, Y, _, _ = make_coordinate_channels(nx, ny)

    if use_sdf:
        geom_ch0 = sample["sdf"] / (p["radius"] + 1e-8)
    else:
        geom_ch0 = sample["mask"]

    ref_field = sample.get("ref_field")
    if not isinstance(ref_field, np.ndarray) or ref_field.ndim != 4 or ref_field.shape[1] != 2:
        raise ValueError("Simulation is missing ref_field with shape (n_ref, 2, nx, ny).")
    ref_scale = float(sample.get("ref_scale", 1.0))
    ref_scale = max(ref_scale, 1e-6)

    channels = [
        geom_ch0.astype(np.float32),
        sample["mask"].astype(np.float32),
        X.astype(np.float32),
        Y.astype(np.float32),
    ]
    for ref_idx in range(ref_field.shape[0]):
        channels.append((ref_field[ref_idx, 0] / ref_scale).astype(np.float32))
        channels.append((ref_field[ref_idx, 1] / ref_scale).astype(np.float32))
    return np.stack(channels, axis=0).astype(np.float32)


def make_inference_inputs(sim_data: dict,
                          param_mean: np.ndarray,
                          param_std: np.ndarray,
                          use_sdf: bool = True):
    """Prepare full-grid query inputs for evaluation or visualization."""
    geom = build_geom_channels(sim_data, use_sdf=use_sdf)
    nx = int(sim_data["params"]["nx"])
    ny = int(sim_data["params"]["ny"])
    _, _, coords, query_ij = make_coordinate_channels(nx, ny)
    params_raw = np.array(
        [
            sim_data["params"]["velocity"],
            sim_data["params"]["viscosity"],
            sim_data["params"]["radius"],
        ],
        dtype=np.float32,
    )
    params = (params_raw - param_mean) / param_std
    query_mask = sim_data["mask"].reshape(-1).astype(np.float32)
    return dict(
        geom=torch.tensor(geom, dtype=torch.float32).unsqueeze(0),
        params=torch.tensor(params, dtype=torch.float32).unsqueeze(0),
        query_xy=torch.tensor(coords, dtype=torch.float32).unsqueeze(0),
        query_ij=torch.tensor(query_ij, dtype=torch.long),
        query_mask=torch.tensor(query_mask, dtype=torch.float32).unsqueeze(0),
        mode_idx=torch.tensor(sim_data["mode_idx"], dtype=torch.long),
        nt=torch.tensor(sim_data["params"]["nt"], dtype=torch.long),
        mask=torch.tensor(sim_data["mask"], dtype=torch.float32),
    )


class TideQueryDataset(Dataset):
    def __init__(self,
                 sim_paths,
                 n_modes=5,
                 queries_per_sample=256,
                 patch_size=3,
                 use_sdf=True,
                 full_grid=False,
                 wake_query_frac=0.5,
                 stats=None,
                 seed=0):
        self.paths = list(sim_paths)
        self.n_modes = int(n_modes)
        self.queries_per_sample = queries_per_sample
        self.patch_size = int(patch_size)
        self.use_sdf = use_sdf
        self.full_grid = full_grid or queries_per_sample is None
        self.wake_query_frac = float(np.clip(wake_query_frac, 0.0, 1.0))
        self.base_seed = int(seed)
        self.rng = np.random.default_rng(seed)

        self.samples = [self._load(p) for p in self.paths]
        self.n_components = 2
        self.n_ref_frames = int(self.samples[0]["ref_field"].shape[0])
        self.geom_channels = 4 + 2 * self.n_ref_frames
        self.mode_idx = self.samples[0]["mode_idx"][:self.n_modes].astype(np.int64)
        self.patch_offsets, self.patch_index = make_patch_offsets(self.patch_size)
        self.n_patch = int(self.patch_offsets.shape[0])

        for sample in self.samples:
            mode_idx = sample["mode_idx"][:self.n_modes].astype(np.int64)
            if not np.array_equal(mode_idx, self.mode_idx):
                raise ValueError(
                    "All simulations must share the same fixed mode indices for query training. "
                    f"Expected {self.mode_idx.tolist()}, got {mode_idx.tolist()}."
                )

        if stats is None:
            pv = np.array(
                [[s["params"]["velocity"], s["params"]["viscosity"], s["params"]["radius"]]
                 for s in self.samples],
                dtype=np.float32,
            )
            self.param_mean = pv.mean(0)
            self.param_std = pv.std(0) + 1e-8
            self.coeff_scale = self._compute_coeff_scale()
        else:
            self.param_mean = np.asarray(stats["param_mean"], dtype=np.float32)
            self.param_std = np.asarray(stats["param_std"], dtype=np.float32)
            self.coeff_scale = np.asarray(stats["coeff_scale"], dtype=np.float32)

        self._coord_cache = {}

    @staticmethod
    def _load(path):
        with open(path, "rb") as f:
            sample = pickle.load(f)

        field = sample.get("field")
        target_field = sample.get("target_field")
        coeffs = sample.get("mode_coeffs")
        ref_field = sample.get("ref_field")
        wake_energy_map = sample.get("wake_energy_map")
        ref_scale = sample.get("ref_scale", None)
        version = sample.get("schema_version", 0)

        if version != SIM_FORMAT_VERSION:
            raise ValueError(
                f"{path} has schema_version={version}; expected {SIM_FORMAT_VERSION}. "
                "Regenerate the dataset with the updated generator."
            )
        if not isinstance(field, np.ndarray) or field.ndim != 4 or field.shape[1] != 2:
            raise ValueError(f"{path} does not contain a velocity field of shape (nt, 2, nx, ny).")
        if not isinstance(target_field, np.ndarray) or target_field.ndim != 4 or target_field.shape[1] != 2:
            raise ValueError(f"{path} does not contain a perturbation field of shape (nt, 2, nx, ny).")
        if not isinstance(coeffs, np.ndarray) or coeffs.ndim != 4 or coeffs.shape[1] != 2:
            raise ValueError(f"{path} does not contain valid velocity Fourier coefficients.")
        if sample.get("mode_space") != "velocity":
            raise ValueError(f"{path} stores mode_space={sample.get('mode_space')}; expected 'velocity'.")
        if not isinstance(ref_field, np.ndarray) or ref_field.ndim != 4 or ref_field.shape[1] != 2:
            raise ValueError(f"{path} does not contain reference velocity frames.")
        if not isinstance(wake_energy_map, np.ndarray) or wake_energy_map.ndim != 2:
            raise ValueError(f"{path} does not contain wake_energy_map with shape (nx, ny).")
        if ref_scale is not None and not np.isfinite(ref_scale):
            raise ValueError(f"{path} does not contain a finite ref_scale.")
        return sample

    def _compute_coeff_scale(self) -> np.ndarray:
        sum_sq = np.zeros((self.n_modes, self.n_components), dtype=np.float64)
        total_count = 0
        for sample in self.samples:
            coeffs = sample["mode_coeffs"][:self.n_modes]
            sum_sq += (np.abs(coeffs).astype(np.float64) ** 2).sum(axis=(2, 3))
            total_count += coeffs.shape[2] * coeffs.shape[3]
        scale = np.sqrt(sum_sq / max(total_count, 1)).astype(np.float32)
        return np.maximum(scale, 1e-6)

    def normalization_state(self):
        return dict(
            param_mean=self.param_mean.copy(),
            param_std=self.param_std.copy(),
            coeff_scale=self.coeff_scale.copy(),
        )

    def denorm_params(self, params_norm):
        pn = params_norm.detach().cpu().numpy() if isinstance(params_norm, torch.Tensor) else np.asarray(params_norm)
        return pn * self.param_std + self.param_mean

    def denorm_coeffs(self, coeff_norm):
        scale = self.coeff_scale
        if isinstance(coeff_norm, torch.Tensor):
            out = coeff_norm
            scale_t = torch.as_tensor(scale, device=out.device, dtype=out.dtype)
            while scale_t.ndim < out.ndim - 1:
                scale_t = scale_t.unsqueeze(0)
            return out * scale_t.unsqueeze(-1)
        out = np.asarray(coeff_norm)
        scale_np = scale
        while scale_np.ndim < out.ndim - 1:
            scale_np = scale_np[None]
        return out * scale_np[..., None]

    def __len__(self):
        return len(self.samples)

    def _cached_coords(self, nx: int, ny: int):
        key = (nx, ny)
        if key not in self._coord_cache:
            self._coord_cache[key] = make_coordinate_channels(nx, ny)
        return self._coord_cache[key]

    def _sample_flat_indices(self, sample: dict) -> np.ndarray:
        nx = int(sample["params"]["nx"])
        ny = int(sample["params"]["ny"])
        total = nx * ny
        if self.full_grid:
            return np.arange(total, dtype=np.int64)

        mask_flat = sample["mask"].reshape(-1) > 0.5
        fluid_idx = np.flatnonzero(mask_flat)
        if len(fluid_idx) == 0:
            return np.arange(min(total, int(self.queries_per_sample)), dtype=np.int64)

        q = int(self.queries_per_sample)
        n_wake = int(round(q * self.wake_query_frac))
        n_uniform = q - n_wake

        uniform_idx = self.rng.choice(fluid_idx, size=n_uniform, replace=len(fluid_idx) < n_uniform)
        wake_values = sample["wake_energy_map"].reshape(-1)[fluid_idx].astype(np.float64)
        if wake_values.sum() < 1e-12:
            wake_idx = self.rng.choice(fluid_idx, size=n_wake, replace=len(fluid_idx) < n_wake)
        else:
            wake_probs = wake_values / wake_values.sum()
            wake_idx = self.rng.choice(fluid_idx, size=n_wake, replace=len(fluid_idx) < n_wake, p=wake_probs)

        flat_idx = np.concatenate([uniform_idx, wake_idx], axis=0).astype(np.int64)
        self.rng.shuffle(flat_idx)
        return flat_idx

    def __getitem__(self, idx):
        sample = self.samples[idx]
        params = sample["params"]
        nx = int(params["nx"])
        ny = int(params["ny"])
        _, _, coords, query_ij = self._cached_coords(nx, ny)

        geom = torch.tensor(build_geom_channels(sample, use_sdf=self.use_sdf), dtype=torch.float32)

        pv_raw = np.array([params["velocity"], params["viscosity"], params["radius"]], dtype=np.float32)
        pv_norm = (pv_raw - self.param_mean) / self.param_std

        flat_idx = self._sample_flat_indices(sample)
        centers_ij = query_ij[flat_idx]
        patch_ij = centers_ij[:, None, :] + self.patch_offsets[None, :, :]
        patch_ij[..., 0] = np.clip(patch_ij[..., 0], 0, nx - 1)
        patch_ij[..., 1] = np.clip(patch_ij[..., 1], 0, ny - 1)
        patch_flat = patch_ij[..., 0] * ny + patch_ij[..., 1]

        coeffs_flat = sample["mode_coeffs"][:self.n_modes].reshape(self.n_modes, self.n_components, -1)
        coeffs = coeffs_flat[:, :, patch_flat.reshape(-1)]
        coeffs = coeffs.reshape(self.n_modes, self.n_components, len(flat_idx), self.n_patch)
        target = np.stack([coeffs.real, coeffs.imag], axis=-1)
        target = np.transpose(target, (2, 3, 0, 1, 4)).astype(np.float32)
        target = target / self.coeff_scale[None, None, :, :, None]

        mask_flat = sample["mask"].reshape(-1).astype(np.float32)
        wake_flat = sample["wake_energy_map"].reshape(-1).astype(np.float32)
        fluid_mean = np.maximum(wake_flat[mask_flat > 0.5].mean(), 1e-6)
        query_mask = mask_flat[flat_idx]
        query_wake = (wake_flat[flat_idx] / fluid_mean).astype(np.float32)
        patch_mask = mask_flat[patch_flat]
        patch_inlet = (patch_ij[..., 0] == 0).astype(np.float32)

        left_idx = self.patch_index["left"]
        right_idx = self.patch_index["right"]
        down_idx = self.patch_index["down"]
        up_idx = self.patch_index["up"]
        center_i = centers_ij[:, 0]
        center_j = centers_ij[:, 1]
        query_div_mask = (
            (center_i > 0) & (center_i < nx - 1) &
            (center_j > 0) & (center_j < ny - 1) &
            (patch_mask[:, self.patch_index["center"]] > 0.5) &
            (patch_mask[:, left_idx] > 0.5) &
            (patch_mask[:, right_idx] > 0.5) &
            (patch_mask[:, down_idx] > 0.5) &
            (patch_mask[:, up_idx] > 0.5)
        ).astype(np.float32)

        return dict(
            geom=geom,
            params=torch.tensor(pv_norm, dtype=torch.float32),
            params_raw=torch.tensor(pv_raw, dtype=torch.float32),
            query_xy=torch.tensor(coords[flat_idx], dtype=torch.float32),
            query_ij=torch.tensor(centers_ij, dtype=torch.long),
            query_mask=torch.tensor(query_mask, dtype=torch.float32),
            query_wake=torch.tensor(query_wake, dtype=torch.float32),
            query_div_mask=torch.tensor(query_div_mask, dtype=torch.float32),
            patch_mask=torch.tensor(patch_mask, dtype=torch.float32),
            patch_inlet=torch.tensor(patch_inlet, dtype=torch.float32),
            target=torch.tensor(target, dtype=torch.float32),
            mode_idx=torch.tensor(self.mode_idx, dtype=torch.long),
            base_field=torch.tensor(sample["base_field"], dtype=torch.float32),
            frame_dt=torch.tensor(
                float(sample["params"]["lbm_frame_stride"]) * float(sample["params"]["lbm_u_in"]) /
                max(float(sample["params"]["velocity"]), 1e-8),
                dtype=torch.float32,
            ),
            mask=torch.tensor(sample["mask"], dtype=torch.float32),
            nt=torch.tensor(params["nt"], dtype=torch.long),
        )


TideDataset = TideQueryDataset


if __name__ == "__main__":
    from generate import generate_dataset

    paths = generate_dataset(n_simulations=1, n_modes=5, n_ref_frames=2, output_dir="data/")
    ds = TideQueryDataset(paths, n_modes=5, queries_per_sample=32, patch_size=3)

    sample = ds[0]
    print("geom       :", sample["geom"].shape, sample["geom"].dtype)
    print("params     :", sample["params"].shape, sample["params"].numpy().round(3))
    print("query_xy   :", sample["query_xy"].shape)
    print("target     :", sample["target"].shape)
    print("mode_idx   :", sample["mode_idx"])
    print("query_mask :", sample["query_mask"].shape)
    print("patch_mask :", sample["patch_mask"].shape)
    print("[OK]")
