"""
Query-conditioned Tide Fourier operator.

This model learns the map
    (geometry, reference velocity frames, parameters, query coordinate)
        -> local velocity Fourier coefficients

Querying all grid points recovers a dense coefficient map.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    return min(max(1, channels // 8), channels)


class ResidualBlock(nn.Module):
    """Small residual block with optional downsampling."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(_group_count(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_ch), out_ch)
        if in_ch != out_ch or downsample:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.shortcut(x)
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(h + skip)


class UpBlock(nn.Module):
    """Upsample, fuse a skip connection, then apply a residual block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch + skip_ch, out_ch, 1)
        self.block = ResidualBlock(out_ch, out_ch, downsample=False)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        return self.block(x)


class FourierCoordinateEmbedding(nn.Module):
    """Sinusoidal embedding of normalized query coordinates.

    When ``learnable=True``, the frequency bands become trainable parameters
    (initialised with even spacing) with an optional L1 penalty that activates
    after a warm-start period — following the data-space preconditioning
    strategy from the SML course material.
    """

    def __init__(self, n_frequencies: int = 6, learnable: bool = False,
                 l1_strength: float = 1e-5, l1_start_epoch: int = 200):
        super().__init__()
        self.n_frequencies = int(n_frequencies)
        self.learnable = learnable
        self.l1_strength = l1_strength
        self.l1_start_epoch = l1_start_epoch

        if learnable:
            # Evenly spaced init covering the useful spatial frequency range
            # for a 64x64 grid with coords in [-1, 1] (max ~32 oscillations).
            init_bands = torch.linspace(1.0, 32.0, self.n_frequencies) * torch.pi
            self.bands = nn.Parameter(init_bands)
        else:
            bands = (2.0 ** torch.arange(self.n_frequencies, dtype=torch.float32)) * torch.pi
            self.register_buffer("bands", bands, persistent=False)

    @property
    def out_dim(self) -> int:
        return 2 + 4 * self.n_frequencies

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        parts = [coords]
        for i in range(self.n_frequencies):
            band = self.bands[i]
            parts.append(torch.sin(band * coords))
            parts.append(torch.cos(band * coords))
        return torch.cat(parts, dim=-1)

    def l1_penalty(self, epoch: int) -> torch.Tensor:
        """L1 penalty on band magnitudes, zero before warm-start epoch."""
        if not self.learnable or epoch < self.l1_start_epoch:
            return self.bands.new_zeros(())
        return self.l1_strength * self.bands.abs().sum()


def apply_hard_constraints(pred: torch.Tensor,
                           patch_mask: torch.Tensor | None = None,
                           patch_inlet: torch.Tensor | None = None) -> torch.Tensor:
    """Apply exact zero constraints on solid cells and inlet perturbations."""
    constrained = pred
    if patch_mask is not None:
        constrained = constrained * patch_mask[..., None, None, None]
    if patch_inlet is not None:
        constrained = constrained * (1.0 - patch_inlet[..., None, None, None])
    return constrained


def extract_center_patch(pred: torch.Tensor, center_patch_index: int) -> torch.Tensor:
    """Return the central pixel from a patch-query prediction tensor."""
    return pred[:, :, int(center_patch_index)]


class TideQueryOperator(nn.Module):
    """Query-conditioned operator for local velocity Fourier coefficients."""

    def __init__(self,
                 n_modes: int = 8,
                 n_params: int = 3,
                 base_ch: int = 48,
                 in_channels: int = 8,
                 n_components: int = 2,
                 coord_frequencies: int = 6,
                 patch_size: int = 3,
                 learnable_coord_embed: bool = False,
                 n_mode_groups: int = 1):
        super().__init__()
        if n_components != 2:
            raise ValueError("The query operator predicts velocity coefficients, so n_components must be 2.")
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer.")

        C = int(base_ch)
        self.n_modes = int(n_modes)
        self.n_components = int(n_components)
        self.patch_size = int(patch_size)
        self.n_patch = self.patch_size * self.patch_size
        self.center_patch_index = self.n_patch // 2
        self.n_mode_groups = max(1, int(n_mode_groups))

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(C), C),
            nn.GELU(),
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(C), C),
            nn.GELU(),
        )
        self.enc2 = ResidualBlock(C, 2 * C, downsample=True)
        self.enc3 = ResidualBlock(2 * C, 4 * C, downsample=True)
        self.bottleneck = nn.Sequential(
            ResidualBlock(4 * C, 4 * C),
            ResidualBlock(4 * C, 4 * C),
        )
        self.dec2 = UpBlock(4 * C, 2 * C, 2 * C)
        self.dec1 = UpBlock(2 * C, C, C)
        self.feature_head = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(C), C),
            nn.GELU(),
        )
        self.sharp_head = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(C), C),
            nn.GELU(),
        )

        self.param_mlp = nn.Sequential(
            nn.Linear(n_params, C),
            nn.GELU(),
            nn.Linear(C, C),
            nn.GELU(),
        )
        self.coord_embed = FourierCoordinateEmbedding(coord_frequencies,
                                                      learnable=learnable_coord_embed)

        head_in = C + C + C + C + self.coord_embed.out_dim

        if self.n_mode_groups <= 1:
            # Single head predicting all modes (original behavior).
            total_out = self.n_patch * self.n_modes * self.n_components * 2
            head_hidden = max(4 * C, total_out // 2)
            self.head = nn.Sequential(
                nn.Linear(head_in, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, total_out),
            )
            nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-4)
            nn.init.zeros_(self.head[-1].bias)
            self.heads = None
            self._modes_per_group = [self.n_modes]
        else:
            # Separate heads per mode group — independent NTK per group.
            base_size, remainder = divmod(self.n_modes, self.n_mode_groups)
            self._modes_per_group = [base_size + (1 if i < remainder else 0)
                                     for i in range(self.n_mode_groups)]
            self.head = None
            self.heads = nn.ModuleList()
            for n_modes_g in self._modes_per_group:
                group_out = self.n_patch * n_modes_g * self.n_components * 2
                group_hidden = max(4 * C, group_out // 2)
                h = nn.Sequential(
                    nn.Linear(head_in, group_hidden),
                    nn.GELU(),
                    nn.Linear(group_hidden, group_hidden),
                    nn.GELU(),
                    nn.Linear(group_hidden, group_out),
                )
                nn.init.normal_(h[-1].weight, mean=0.0, std=1e-4)
                nn.init.zeros_(h[-1].bias)
                self.heads.append(h)

    def encode(self, geom: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.stem(geom)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d2 = self.dec2(b, e2)
        d1 = self.dec1(d2, e1)
        return self.feature_head(d1), self.sharp_head(e1)

    def encode_context(self, geom: torch.Tensor, params: torch.Tensor):
        feature_map, sharp_map = self.encode(geom)
        global_feat = F.adaptive_avg_pool2d(feature_map, output_size=1).flatten(1)
        param_feat = self.param_mlp(params)
        return feature_map, sharp_map, global_feat, param_feat

    @staticmethod
    def _grid_sample_features(feature_map: torch.Tensor, query_xy: torch.Tensor) -> torch.Tensor:
        """
        Bilinearly sample local features at query points.

        ``query_xy`` uses the natural [x, y] order on the dataset grid, while
        ``grid_sample`` expects [width, height], so we swap the axes here.
        """
        grid = torch.stack([query_xy[..., 1], query_xy[..., 0]], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(-1).transpose(1, 2)

    def decode_queries(self,
                       feature_map: torch.Tensor,
                       sharp_map: torch.Tensor,
                       global_feat: torch.Tensor,
                       param_feat: torch.Tensor,
                       query_xy: torch.Tensor) -> torch.Tensor:
        B = feature_map.shape[0]
        Q = query_xy.shape[1]
        local_feat = self._grid_sample_features(feature_map, query_xy)
        sharp_feat = self._grid_sample_features(sharp_map, query_xy)
        global_feat = global_feat[:, None, :].expand(-1, Q, -1)
        param_feat = param_feat[:, None, :].expand(-1, Q, -1)
        coord_feat = self.coord_embed(query_xy)

        head_in = torch.cat([local_feat, sharp_feat, global_feat, param_feat, coord_feat], dim=-1)

        if self.heads is not None:
            # Separate mode-group heads — concatenate along mode dimension.
            parts = []
            for h, n_modes_g in zip(self.heads, self._modes_per_group):
                raw = h(head_in)
                parts.append(raw.view(B, Q, self.n_patch, n_modes_g, self.n_components, 2))
            out = torch.cat(parts, dim=3)
        else:
            out = self.head(head_in)
            out = out.view(B, Q, self.n_patch, self.n_modes, self.n_components, 2)
        return out

    def forward(self,
                geom: torch.Tensor,
                params: torch.Tensor,
                query_xy: torch.Tensor,
                query_mask: torch.Tensor | None = None) -> torch.Tensor:
        feature_map, sharp_map, global_feat, param_feat = self.encode_context(geom, params)
        return self.decode_queries(feature_map, sharp_map, global_feat, param_feat, query_xy)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


TideFourierOperator = TideQueryOperator


if __name__ == "__main__":
    model = TideQueryOperator(n_modes=8, n_params=3, base_ch=32, in_channels=8, n_components=2, patch_size=3)
    print(f"Parameters: {model.count_params():,}")

    batch = 2
    nx = ny = 64
    q = 128
    geom = torch.randn(batch, 8, nx, ny)
    params = torch.randn(batch, 3)
    query_xy = torch.rand(batch, q, 2) * 2.0 - 1.0
    query_mask = (torch.rand(batch, q) > 0.1).float()
    out = model(geom, params, query_xy, query_mask=query_mask)

    print("Output shape:", out.shape)
    assert out.shape == (batch, q, 9, 8, 2, 2)
    print("[OK]")
