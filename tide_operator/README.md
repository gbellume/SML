# Tide Fourier Operator

**Query-conditioned, physics-informed neural operator for temporal Fourier coefficients of 2D incompressible flow past a cylinder.**

MSc Scientific Machine Learning (SML), TU Delft.

---

## 1. Problem Statement

### 1.1 Governing equations

Two-dimensional viscous incompressible flow satisfies the Navier-Stokes system. In primitive form:

```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u        (momentum)
∇·u = 0                                (continuity)
```

Taking the curl eliminates pressure and yields the vorticity-transport equation for ω = ∂v/∂x - ∂u/∂y:

```
∂ω/∂t + u·∂ω/∂x + v·∂ω/∂y = ν(∂²ω/∂x² + ∂²ω/∂y²)
```

The cylinder imposes a no-slip boundary on the obstacle surface Γ_c, uniform inflow U on the left edge, free-slip on the top/bottom walls, and a zero-gradient outflow on the right. The non-dimensional number controlling the regime is the Reynolds number:

```
Re = U · D / ν        with D = 2R (cylinder diameter)
```

Our parameter sweep (U ∈ [4, 8], ν ∈ [0.05, 0.25], R ∈ [6, 11] in lattice units) spans Re ≈ 200–500, firmly inside the periodic von Kármán vortex-shedding regime (Strouhal number St ≈ 0.18–0.19).

### 1.2 Temporal Fourier decomposition

Because vortex shedding is periodic, the velocity field separates cleanly into a time-mean plus a band-limited oscillatory perturbation:

```
u(x, y, t) = ū(x, y) + u'(x, y, t)        with  u'(x, y, t) = Σ_k â_k(x, y) e^(2πikt/T)
```

We retain only the K = 12 lowest positive harmonics; negative-frequency bins are handled by the conjugate-symmetry constraint of a real-valued signal. The discrete Fourier coefficient at mode k is a 2D complex field:

```
â_k(x, y) = (1/T) Σ_{t=0}^{T-1} u'(x, y, t) · e^(-2πikt/T)     ∈ ℂ^2
```

Here T = 256 stored frames (≈ one shedding period, see §2.2 for how the frame stride is chosen). Each coefficient is stored as a `(Re, Im)` pair per velocity component, giving 4 real numbers per `(x, y, k)` triple and 48 numbers per `(x, y)` location for K = 12.

### 1.3 Learning problem

We want the operator:

```
G_θ : (geometry, reference velocity, (U, ν, R), x_q) ↦ { â_k(x_q) }_{k=1}^{K}
```

**Inputs** at a single evaluation:
- `sdf` ∈ ℝ^(64×64): signed distance function of the cylinder
- `mask` ∈ {0, 1}^(64×64): fluid indicator
- `ref_frames` ∈ ℝ^(n_ref×2×64×64): 12 reference velocity snapshots (see §2.3)
- `(U, ν, R)` ∈ ℝ^3: physical parameters
- `x_q` ∈ [−1, 1]^2: continuous query coordinate

**Output** at x_q: `(â_1, â_2, …, â_K)` as 12 × 2 × 2 = 48 real numbers (12 modes × 2 velocity components × 2 real/imag parts).

Evaluating G_θ at every pixel of the 64×64 grid reconstructs the full spectral-spatial coefficient tensor, from which time-domain velocities are obtained by inverse DFT.

---

## 2. Data Generation

### 2.1 D2Q9 Lattice Boltzmann Solver

The ground-truth simulator uses the D2Q9 (2D, 9-velocity) lattice Boltzmann method with BGK collision. The population densities `f_i(x, t)` for i = 0…8 evolve as:

```
f_i(x + c_i·Δt, t + Δt) = f_i(x, t) − (1/τ)[f_i(x, t) − f_i^eq(ρ, u)]    (collision + streaming)
```

with equilibrium distribution:

```
f_i^eq = w_i·ρ·[1 + (c_i·u)/c_s² + (c_i·u)²/(2c_s⁴) − |u|²/(2c_s²)]
```

The D2Q9 lattice velocities and weights are:

```
c_0 = (0,0),     w_0 = 4/9
c_{1..4} = axial (±1, 0), (0, ±1),     w = 1/9
c_{5..8} = diagonal (±1, ±1),           w = 1/36
c_s² = 1/3  (lattice speed of sound squared)
```

Kinematic viscosity is tied to the relaxation time via:
```
ν = c_s² · (τ − 1/2) = (1/3)(τ − 1/2)
```

Macroscopic quantities are recovered by moments:
```
ρ = Σ_i f_i,       ρu = Σ_i c_i · f_i
```

### 2.2 Boundary conditions and stability

- **Inlet (x = 0)**: Zou-He velocity BC fixing `u = (U_lbm, 0)`.
- **Outlet (x = nx−1)**: zero-gradient via copy from the interior.
- **Top/bottom walls**: free-slip (specular reflection).
- **Cylinder surface**: half-way bounce-back (second-order accurate no-slip).

`generate_lbm.py` automatically rescales the physical input `(U, ν)` to lattice units subject to `τ > 0.51` (hard stability floor) and `U_lbm < U_max ≈ 0.3` (compressibility floor). Requests outside the representable regime are rejected (`strict_regime=True`).

### 2.3 Time sampling and phase alignment

For each sim, a target Strouhal number `St(Re) = 0.198(1 − 19.7/Re)` determines the shedding period `T_shed = 2R / (U · St)` in lattice steps. The warmup is `1.5 · T_shed` (minimum 200 steps) and the recording stride is chosen so that the 256 saved frames span exactly one shedding period:

```
frame_stride = round(T_shed / 256)
```

This phase alignment is critical: it makes the temporal spectrum of u' concentrate on integer multiples of the shedding frequency, so the 12 retained modes capture > 99% of the perturbation energy on average.

A separate set of 12 **reference frames** is extracted at evenly spaced times over the recording window. They are normalized by `ref_scale = max(|u'|)` so the encoder sees phase-aligned, amplitude-normalized velocity snapshots as input channels.

### 2.4 Stored artifacts per simulation

```
sim_NNNN.pkl:
    field            (256, 2, 64, 64)   full velocity u(x, t)
    target_field     (256, 2, 64, 64)   perturbation u' = u − ū
    base_field       (2, 64, 64)        time-mean ū
    mask, sdf        (64, 64)           geometry
    params           dict               (velocity, viscosity, radius, nt, dt, ...)
    fft_field        (K, 2, 64, 64)     complex FFT coefficients (kept modes)
    mode_idx         (K,)               frequency bin indices = [1..12]
    mode_coeffs      (K, 2, 64, 64)     alias of fft_field (complex)
    ref_field        (12, 2, 64, 64)    phase-aligned reference snapshots
    ref_scale        scalar             normalization for ref_field
    wake_energy_map  (64, 64)           time-RMS of |u'|, used for query sampling
```

---

## 3. Dataset and Query Sampling

### 3.1 Geometry channels (model input)

For each sim, `build_geom_channels` produces a `(4 + 2·n_ref, 64, 64)` input tensor. With n_ref = 12 this gives **28 channels**:

| Channel | Content | Normalization |
|---|---|---|
| 0 | `sdf` | divided by cylinder radius (scale-invariant) |
| 1 | `mask` | 0/1 |
| 2 | `X` coordinate | linspace `[−1, 1]` |
| 3 | `Y` coordinate | linspace `[−1, 1]` |
| 4..27 | 12 ref frames × (u, v) | divided by `ref_scale` |

The scalar parameters `(U, ν, R)` are z-normalized by `(param_mean, param_std)` computed from the training set, then passed to the `param_mlp` head separately.

### 3.2 Coefficient normalization

The complex target coefficients are divided **per-mode per-component** by:

```
coeff_scale[k, c] = sqrt( E_{sims, x, y} [ |â_k^{c}|² ] )      shape (K, 2)
```

This is the RMS magnitude of coefficient k of component c over the training set. It brings all modes to order unity and is the only statistic the model needs to denormalize predictions at inference time. Stored in `normalization_state` alongside `param_mean/std`.

### 3.3 Query sampling and wake bias

Training uses 256 query points per simulation per iteration. The queries are drawn from two pools:

```
n_wake    = round(256 · wake_query_frac)        # default frac = 0.5 → 128
n_uniform = 256 − n_wake                        # → 128

uniform queries: drawn uniformly from fluid cells
wake queries:    drawn from fluid cells with probability ∝ wake_energy_map
```

`wake_energy_map[x, y] = (1/T) Σ_t |u'(x, y, t)|²` concentrates most of its mass in the shedding wake behind the cylinder, so biased sampling forces the model to see the informative region at every step.

Validation uses `full_grid=True` (all 4096 pixels), so val error is computed over the entire fluid domain with a single forward pass per sim.

### 3.4 Patch extraction

Each query center spawns a 5×5 patch of neighbors (25 samples):

```
patch_offsets = [(i, j) for i in {−2, −1, 0, 1, 2} for j in {−2, −1, 0, 1, 2}]
patch_index   = { center: 12, left: 11, right: 13, down: 10, up: 14, ... }
```

Patch targets are fetched from the sim's `mode_coeffs` at the patch pixel indices. This gives the model 25 coefficient predictions per query — enough to apply central finite differences for divergence and boundary losses directly in coefficient space (§4.4).

Patch masks:
- `patch_mask` = fluid indicator at each patch pixel; zeros the coefficients inside the obstacle.
- `patch_inlet` = 1 at `i = 0` (the inlet column); enforces `u' = 0` at the fixed-inlet BC.

These are applied as **exact hard constraints** on both pred and target (`apply_hard_constraints`), so the loss never penalizes boundary leakage.

---

## 4. Model Architecture

### 4.1 Overview

```
geom  (28, 64, 64)                    params (3,)
   │                                       │
   │  U-Net encoder                        │  param_mlp
   │  ────────────                         │  (Linear 3→64, GELU, Linear 64→64, GELU)
   │                                       │
   ├──▶ feature_map (64, 64, 64)           │
   └──▶ sharp_map   (64, 64, 64)           │
               │                           │
               │  avg_pool                 │
               └──▶ global_feat (64,) ─────┤
                                           │
   query_xy (Q, 2) ──▶ coord_embed ────▶ coord_feat (Q, 2+4·6 = 26)
          │
          ├──▶ grid_sample(feature_map) ──▶ local_feat  (Q, 64)
          └──▶ grid_sample(sharp_map)   ──▶ sharp_feat  (Q, 64)

   concat(local_feat, sharp_feat, global_feat, param_feat, coord_feat)
          │   64 + 64 + 64 + 64 + 26 = 282
          │
          ▼
   3 MLP heads (one per mode group)   ─▶   output (B, Q, 25, 12, 2, 2)
```

Total parameters: **4.80 M** at base_ch C = 64.

### 4.2 U-Net encoder

```
stem         Conv3x3(28→C) → GN → GELU → Conv3x3(C→C) → GN → GELU          [64×64, C]
enc2         ResBlock(C → 2C, stride 2)                                      [32×32, 2C]
enc3         ResBlock(2C → 4C, stride 2)                                     [16×16, 4C]
bottleneck   ResBlock(4C → 4C) × 2                                           [16×16, 4C]
dec2         bilinear ↑ → concat(enc2) → Conv1x1(6C→2C) → ResBlock(2C→2C)   [32×32, 2C]
dec1         bilinear ↑ → concat(stem) → Conv1x1(3C→C)  → ResBlock(C→C)     [64×64,  C]
feature_head Conv3x3(C→C) → GN → GELU                                        [64×64,  C]
sharp_head   Conv3x3(C→C) → GN → GELU      ← applied to stem output         [64×64,  C]
```

`ResBlock` = `Conv3x3 → GN → GELU → Conv3x3 → GN → (+skip) → GELU`. `GroupNorm` groups = `min(max(1, out_ch/8), out_ch)`.

The two output heads serve different purposes:
- `feature_map` comes out of the full U-Net and has the largest receptive field (≈ global), carrying context about the wake and domain.
- `sharp_map` branches off the stem (receptive field ≈ 5×5 pixels), preserving fine geometric detail near the cylinder that the deep path would blur.

At query time both are **bilinearly sampled** at `x_q` and concatenated — this is the only way spatial convolutional information reaches the per-query MLP.

### 4.3 Continuous coordinate embedding

```python
FourierCoordinateEmbedding(n_frequencies = 6, learnable = True)
```

Given `x_q ∈ [−1, 1]^2`, the embedding is:

```
φ(x_q) = [x_q, sin(b_1·x_q), cos(b_1·x_q), …, sin(b_6·x_q), cos(b_6·x_q)]     (26-dim)
```

with `b_i ∈ ℝ` learnable. Init: `b_i = linspace(1, 32)·π` (covers spatial frequencies from 0.5 to 16 cycles across the domain, matching the 64-pixel grid). Differential LR = 0.5× the base LR prevents early collapse — the `SML/models.py` course experiments found 0.1× too conservative.

An L1 penalty `l1_strength · mean(|b|)` is activated after epoch 200 to prune unused frequencies (data-space preconditioning, §5 of `SLIDE_ANALYSIS.md`).

### 4.4 Mode-group heads (NTK decomposition)

A single MLP head predicting all 12 modes exhibits spectral bias: the NTK's eigenvalue spectrum is wide, so low-frequency modes converge orders of magnitude faster than high-frequency ones. The structural fix is to split the prediction into **independent heads per mode group**, each developing its own NTK:

```
head_A: modes 1–4,   in_dim = 282 → hidden = max(4C, out/2) → out = 25·4·2·2 = 400
head_B: modes 5–8,   same hidden sizing, out = 400
head_C: modes 9–12,  same, out = 400

hidden_dim = max(256, 200) = 256   at C = 64
```

Each head: `Linear(282 → 256) → GELU → Linear(256 → 256) → GELU → Linear(256 → out)`, final layer initialized `N(0, 1e−4)` with zero bias so the network starts near the zero-perturbation solution.

Outputs are concatenated along the mode dim, giving the final shape `(B, Q, 25, 12, 2, 2)`.

**Why this works**: Each head sees gradients only from its own 4 modes. The NTK spectrum of each sub-head is narrower, so high-frequency modes in group C are no longer drowned out by the mode-1 gradients in group A. See `SLIDE_ANALYSIS.md` §2 for the NTK theory and Runs 3/5 for the +1.3pp gain.

### 4.5 Hard constraints

Applied after the head output (non-learned):

```python
pred = pred * patch_mask      # zero inside obstacle
pred = pred * (1 − patch_inlet) # zero at inlet column
```

These are enforced on targets too, so the loss sees exactly the space of physically admissible coefficients. Following Lagaris et al., hard constraints outperform soft penalties by 1–2 orders of magnitude (`physics-informed_neural_networks.pdf` §3).

---

## 5. Loss function

The total loss (§`run_epoch`, lines 530–631 of `query_train.py`):

```
L = λ_spec · L_spec  +  λ_amp · L_amp  +  λ_recon · L_recon
  + λ_div  · L_div   +  λ_bc  · L_bc   +  λ_vort  · L_vort
  + w(epoch) · λ_pde · L_pde
```

Active weights:
```
λ_spec = 1.0    λ_amp = 0.25   λ_recon = 0.1
λ_div  = 0.05   λ_bc  = 0.01   λ_vort  = 0.01
λ_pde  = 0.05   (with warmup w(epoch) = min(1, epoch/300))
```

All query-level losses share the **wake weight**:
```
weight(q) = query_mask(q) · (1 + wake_focus · max(0, query_wake(q)))
            with wake_focus = 4.0
```

and reduce via a weighted mean `(Σ weight · value) / (Σ weight)`.

### 5.1 Coefficient loss (primary)

Let `p, t ∈ ℝ^(B×Q×K×C×2)` be predicted and target coefficients after hard constraints and center-patch extraction. Then:

```
L_spec = E_q [ mean over (k, c, r) of  smoothL1(p − t) ]
```

Smooth L1 (Huber, default β = 1) behaves like ½ x² for |x| < 1 and like |x| − ½ for |x| > 1. It is less sensitive than MSE to early-training outliers but keeps a proportional gradient once predictions are close.

Real and imaginary parts enter the loss **independently** (they occupy a flattened axis inside the `mean`). This is equivalent to the complex L2 norm up to a factor and matches the natural regression geometry of a 2D vector in ℂ.

### 5.2 Amplitude loss

Uses the complex magnitude per `(k, c)`:

```
|p|_{k,c} = sqrt(p_Re² + p_Im²)       (torch.linalg.vector_norm on last axis)
L_amp     = E_q [ mean over (k, c) of  smoothL1(|p| − |t|) ]
```

This couples real and imaginary components of each coefficient, providing gradient signal about energy even when phase is wrong.

### 5.3 Reconstruction loss

Inverse DFT via `reconstruct_query_modes_torch`:

```
u'(x_q, t)_c = Σ_k w_k [ Re(ĉ_k^c)·cos(2π k t/T) − Im(ĉ_k^c)·sin(2π k t/T) ]
```

with `w_k = 2` for non-self-conjugate modes (accounts for the hidden negative-frequency bins). Then:

```
L_recon = E_q [ (1/(T·C)) · ||u'_pred(x_q, ·) − u'_target(x_q, ·)||² ]
```

Because the time-domain signal is naturally dominated by low modes (high energy), MSE in time implicitly gives those modes more gradient than uniform L1 on coefficients would. This is a data-space regularizer that rewards *physical* reconstruction quality rather than per-mode accuracy.

### 5.4 Divergence loss (patch-local FD)

Operates directly on the patch-neighbor coefficients (in unnormalized coefficient space, using `coeff_scale`):

```
(∂û_k/∂x)(x_q) ≈ (û_k(right) − û_k(left)) / (2 Δx)       [patch indices]
(∂v̂_k/∂y)(x_q) ≈ (v̂_k(up)    − v̂_k(down))  / (2 Δy)
div_k = (∂û_k/∂x) + (∂v̂_k/∂y)
L_div = E_q [ mean_k (div_k)² ]      (only where query_div_mask = 1)
```

`Δx = Δy = 2/(nx − 1)`. `query_div_mask` excludes queries whose 4 neighbors leave the fluid or the domain.

**Why per-mode divergence?** `∇·u' = ∇·(Σ_k â_k e^(iωt)) = Σ_k (∇·â_k) e^(iωt)`. Linear independence of the Fourier basis means `∇·u' = 0` ⟺ `∇·â_k = 0` for every k separately, so penalizing per-mode divergence is equivalent to enforcing incompressibility in physical space (without the cost of time-domain reconstruction).

### 5.5 Boundary condition loss

Enforces zero-gradient (Neumann) on the outlet and top/bottom walls by requiring the center pixel to equal its interior neighbor:

```
outlet (i = nx−1):     smoothL1(center − left)
top    (j = ny−1):     smoothL1(center − down)
bottom (j = 0):        smoothL1(center − up)
L_bc = mean over active edges
```

### 5.6 Vorticity-patch loss

Like divergence but on the reconstructed vorticity field inside each patch. Uses the central FD stencil on the 25-pixel patch to compute `ω = ∂v/∂x − ∂u/∂y` from the reconstructed time signal at the patch neighbors, then MSE against the target vorticity. Small weight (0.01) — mostly a regularizer that couples u and v.

### 5.7 Global PDE residual loss

This is the expensive one. The full pipeline:

```
1. Predict coefficients at ALL 4096 grid points:
      coeffs_norm[b, q, K, 2, 2] = G_θ(geom, params, x_full)
      coeffs_raw = coeffs_norm · coeff_scale

2. Inverse DFT to time-samples at a subset of times:
      u'(x, y, t_j)  for  j ∈ {0, …, M−1},  M = pde_time_samples = 8
      ∂_t u'(x, y, t_j) analytically from (−Re·sin, −Im·cos) · (2πk/T)

3. Add the mean flow:
      u(x, y, t_j) = ū(x, y) + u'(x, y, t_j)

4. Spatial derivatives with 4th-order central FD:
      ∂_x f ≈ (−f[i+2] + 8f[i+1] − 8f[i−1] + f[i−2]) / (12·Δx)
      ∂²_x f ≈ (−f[i+2] + 16f[i+1] − 30f[i] + 16f[i−1] − f[i−2]) / (12·Δx²)

   Falls back to 2nd-order `(f[i+1] − f[i−1])/(2Δx)` at the cells adjacent to the boundary.

5. Vorticity transport residual:
      ω = ∂_x v − ∂_y u
      R_ω = ∂_t ω + u·∂_x ω + v·∂_y ω − ν·(∂²_x ω + ∂²_y ω)

6. Incompressibility residual:
      R_div = ∂_x u + ∂_y v

7. Loss:
      L_pde = w_vort · ⟨R_ω², M_int⟩ + w_div · ⟨R_div², M_int⟩
      M_int = fluid cells with all 8 FD neighbors inside fluid (2-cell border)
```

The PDE loss is expensive because step 1 runs the decoder on 4096 query points per physics batch, step 2 reshapes to a space-time tensor, and steps 4–5 run multiple Laplacians. The `pde_time_samples = 8` and `pde_query_chunk_size = 1024` parameters keep memory bounded.

**Warmup is essential**: for `epoch < 300` the weight is linearly ramped `w(epoch) = epoch/300`. Enabling L_pde at full strength from the start kills training — the model has not yet learned meaningful spatial patterns, so it minimizes the PDE residual by predicting zero everywhere.

### 5.8 Relative L2 error (eval metric)

During validation (and for per-mode diagnostics) we track:

```
ε(%) = 100 · sqrt( Σ_{q fluid} ||p_raw − t_raw||²  /  Σ_{q fluid} ||t_raw||² )
```

where the norms are taken over the 48-dim `(K, C, real/imag)` coefficient vector and `p_raw, t_raw` are un-normalized (multiplied back by `coeff_scale`). This is what the "val error %" column in the training log reports and is the single number by which runs are compared.

---

## 6. Training Configuration

### 6.1 Optimizer and schedule

```python
optim.Adam([
    { params: non-embed, lr: 1e-3 },
    { params: coord_embed.bands, lr: 0.5 · 1e-3 },   # differential LR
], weight_decay = 1e-5)

sched = CosineAnnealingLR(T_max = epochs, eta_min = 1e-6)
```

Gradient clipping by global L2 norm with `max_norm = 5.0`.

**Why Cosine and not Plateau?** With `val_every = 20` and `patience = 30`, the plateau scheduler's rescale trigger `patience · val_every = 600` exceeds typical run lengths, so the LR never reduces. Cosine's monotonic decay guarantees the late-training refinement phase actually happens.

### 6.2 Data split and batching

- 96 simulations total → 80/20 deterministic train/val split (seed 42 → 76 train, 20 val).
- `batch_size = 8` training, `val_batch_size = 1` (val uses full-grid 4096 queries).
- Single-GPU (CUDA). No distributed or mixed-precision — the bottleneck is the PDE loss, not matmul throughput.

### 6.3 Physics-loss schedule

```
epoch 1..200   :  λ_pde = 0                    (pure data-driven)
epoch 1..200   :  L1 on coord_embed = 0        (frequencies free to drift)
epoch 200..    :  L1 on coord_embed = 1e-4·⟨|b|⟩  (prune unused bands)
epoch 1..299   :  λ_pde = 0.05 · epoch/300     (linear warmup)
epoch 300..    :  λ_pde = 0.05                 (full strength)
```

### 6.4 Hyperparameters (Run 7, current best config)

| Category | Param | Value |
|---|---|---|
| Data | `n_sims` | 96 |
| Data | `n_modes` / `n_ref_frames` / `patch_size` | 12 / 12 / 5 |
| Data | `queries` / `wake_query_frac` / `wake_focus` | 256 / 0.5 / 4.0 |
| Model | `base_ch` / `n_mode_groups` | 64 / 3 |
| Model | `learnable_embed` / `embed_lr_scale` | True / 0.5 |
| Optim | `lr` / `grad_clip_norm` / `weight_decay` | 1e-3 / 5.0 / 1e-5 |
| Loss  | `lam_amp` / `lam_recon` / `lam_div` | 0.25 / 0.1 / 0.05 |
| Loss  | `lam_bc` / `lam_vort` / `lam_pde` | 0.01 / 0.01 / 0.05 |
| Loss  | `pde_warmup_epoch` / `pde_time_samples` | 300 / 8 |
| Sched | `epochs` / `val_every` / scheduler | 1000 / 20 / cosine |

---

## 7. Results

| Run | Config delta | Best val | Gap (pp) | Notes |
|---|---|---|---|---|
| Baseline | head_hidden = 96, 2.5M params | 41.16% | ~15 | — |
| Run 1 | `head_hidden = 600` (3.5M) | 30.36% | ~14 | Head capacity (DeepONet trunk) |
| Run 2 | + learnable embed, mode weights | 29.29% | ~14 | Data-space preconditioning |
| Run 3 | + `n_mode_groups = 3` | 27.95% | ~14 | NTK decomposition per group |
| Run 4 | + `base_ch = 64` (4.8M), 600 ep | 28.15% | ~15 | Encoder capacity |
| Run 5 | + PDE loss (warmup 200), 700 ep | **26.75%** | 14.6 | Physics regularization |
| Run 6 | + 96 sims, 4th-order FD, mode-norm loss | 24.40% | 14.5 | Broken double-normalization bug |
| Run 7 | 96 sims, 4th-order FD, no mode-norm | **≤23.1%** | ~13 | Clean data + stencil improvements |

**Run 6 vs Run 7 finding**: The `normalize_mode_loss` flag introduced in Run 6 divided the per-mode coefficient loss by `coeff_scale`. But the dataset **already** normalizes targets by the same `coeff_scale` — so the effective weight was `1/coeff_scale²`, giving mode 12 (scale ~0.037) about 400× more gradient than mode 1 (scale ~0.73). The model overfitted noise on high modes (group C went from 72% → 80% avg error) while the overall L2 metric still improved slightly thanks to the 3× data. Run 7 removes the double normalization and uses the cleaner loss.

---

## 8. File Map

```
run_multi.py        entry point — CLI args, data gen hook, train(), plots, GIFs
query_model.py      TideQueryOperator (U-Net, coord embed, mode-group heads)
query_train.py      training loop, all losses, global_physics_loss, 4th-order FD
query_dataset.py    TideQueryDataset — wake-biased sampling, patch extraction, normalization
query_visualize.py  diagnostic plots, animations, loss breakdowns
generate_lbm.py     D2Q9 simulator + Fourier extraction + regime checks
generate.py         compatibility wrapper (re-exports from generate_lbm)
lbm.py              D2Q9 collision/streaming, Zou-He inlet, bounce-back, stability helpers
spectral.py         inverse-DFT and analytic time-derivative on sparse modes
equations.py        pure-arithmetic PDE residual functions (numpy + torch)
_test_lbm.py        LBM sanity check (run to verify solver is stable)

SLIDE_ANALYSIS.md   course-concept mapping + per-run quantitative analysis
README.md           this file

data_stage_next/    96 LBM simulations (sim_NNNN.pkl)
results_run{5,6,7}/ model_best.pt, model_final.pt, history.pkl, plots, GIFs
```

---

## 9. Quickstart

Best-known training command (Run 7 config):

```bash
python run_multi.py \
  --n_sims 96 --epochs 1000 --val_every 20 \
  --base_ch 64 --lr 1e-3 --queries 256 \
  --n_modes 12 --n_ref_frames 12 --patch_size 5 \
  --wake_strength 1.5 --lam_amp 0.25 --lam_recon 0.1 \
  --lam_vort 0.01 --lam_div 0.05 --lam_bc 0.01 \
  --lam_pde 0.05 --pde_warmup_epoch 300 \
  --wake_focus 4.0 --wake_query_frac 0.5 --grad_clip_norm 5.0 \
  --n_mode_groups 3 --learnable_embed --embed_lr_scale 0.5 \
  --data_dir data_stage_next --out_dir results_run7
```

### Reproducing the experiments

**No data, results, or model checkpoints ship with this repository.** Everything is regenerated from source by the command above.

The pipeline runs in two phases:

1. **LBM data generation** (CPU-bound, ~25 min for 96 sims). The first time `run_multi.py` is invoked with a non-existent `--data_dir`, it calls `generate_dataset()` from `generate_lbm.py` and writes 96 simulation pickles to `data_stage_next/sim_NNNN.pkl`. Subsequent runs see the cached files and skip directly to training. The simulator is deterministic (seeded), so re-running on a different machine yields the same dataset.

2. **GPU training** (~75 min on an RTX 4070, ~1000 epochs over 76 train sims + 20 val sims). Outputs are written to `--out_dir`: `model_best.pt`, `model_final.pt`, `history.pkl`, `metadata.pkl`, and a set of diagnostic plots and animated GIFs.

**End-to-end runtime on a fresh clone is ~100 min on an RTX 4070-class GPU.** Roughly 25 min CPU + 75 min GPU. The disk footprint after a full run is ~10 GB for `data_stage_next/` and ~250 MB for `results_run7/`.

Requirements:
- Python 3.10 or newer
- PyTorch 2.0+ with CUDA (the LBM generator runs on CPU; only training uses the GPU)
- NumPy, Matplotlib

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy matplotlib
```

If you only want a sanity check rather than the full run, the LBM solver alone can be tested in a few seconds with `python _test_lbm.py`.
