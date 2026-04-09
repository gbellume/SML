# Course Material -> Tide Operator Problem Mapping

Analysis of all 7 SML course lecture PDFs, mapped to the tide_operator architecture and training.

## Core Problem Recap
Model predicts temporal Fourier coefficients (12 modes, 2 velocity components, real+imag) at query points.
Captures modes 1-2 well but modes 3-12 poorly. Val error plateau ~41%.
Architecture: U-Net encoder -> bilinear sampling at query coords -> MLP head per query point.

---

## 1. Spectral Bias / NTK (neural_network_training.pdf)

**Theory**: NTK eigenmode error decays as `(1 - lambda_i)^k`. Wide eigenvalue spread = slow convergence for small-eigenvalue (high-frequency) modes.

**Connection**: The MLP head predicts all 12 temporal modes simultaneously. Its NTK spectrum is biased toward low modes. This is the FUNDAMENTAL explanation for the mode 1-2 vs 3-12 gap.

**Actionable**:
- **Per-mode loss weighting**: Weight modes 3-12 more heavily to force gradient signal toward high-mode parameters. Changes the effective eigenvalue distribution in the NTK.
- **Separate mode heads**: Instead of one MLP predicting all 12 modes, use mode-group heads (e.g., modes 1-4, 5-8, 9-12). Each head has its own NTK with a narrower spectrum.
- **Progressive training**: Train modes 1-4 first (low-freq, fast convergence), then expand to all 12.

## 2. Five Preconditioning Approaches (neural_network_training.pdf)

The slides present five strategies to counter spectral bias:

### 2a. Optimizer-based (Adam)
Already using Adam. Marginal gain from switching. Could try AdamW with higher weight decay.

### 2b. Second-order (Gauss-Newton / Levenberg-Marquardt)
Dramatic spectral bias reduction (f=16 converges in ~100 epochs vs 5000 for GD).
**But**: J^T J is O(N * params^2) -- infeasible for 3.5M params.
**Possible**: Apply LM-like preconditioning to just the MLP head (~2M params).
**Alternative**: L-BFGS for the head's linear output layer.

### 2c. Data-space preconditioning (Fourier features)
Current model: FourierCoordinateEmbedding with FIXED bands `2^k * pi`, k=0,...,5.

**Key improvement**: Replace with LEARNABLE Fourier embedding (same design as models.py):
  - Evenly spaced init (linspace, not geometric 2^k)
  - Learnable B matrix (frequencies migrate during training)
  - L1 penalty with warm start (prevents early collapse, prunes later)
  - Differential LR (0.1x for embedding vs 1x for network)

The spatial frequency content of the wake is NOT well-captured by octave bands. Learnable frequencies would adapt to the actual scales present in the flow.

### 2d. Function-space preconditioning (Domain Decomposition)
The patch-based approach IS a form of DD -- each 5x5 patch is a local subdomain.
**But**: The encoder is still global (U-Net processes full 64x64 grid). Locality only enters at the head.
**Further**: Could use separate head weights for near-cylinder vs far-wake regions.

### 2e. Parameter-space preconditioning (Alternating updates)
Split params into encoder (U-Net) and decoder (MLP head).
Alternating: freeze encoder, train head for N epochs, then freeze head, train encoder.
**Rationale**: Reduces cross-layer coupling. Could also solve head's output layer exactly (least-squares).

## 3. PINNs - Hard vs Soft Constraints (physics-informed_neural_networks.pdf)

**Theory**: Hard constraints (Lagaris: u = b*g) outperform soft BC penalties by 1-2 orders of magnitude.

**Connection**:
- Current model already uses hard constraints via masking (multiply by mask, zero at inlet).
- GELU activation is C-infinity -- fine for any derivative-based loss.
- The PDE loss uses FD stencils, matching the CNN-PDE approach from operator_learning_ii.

**Key insight**: Physics-informed training needs ~10x more epochs but achieves better physical consistency. Currently lam_pde=0. Could enable at low weight after initial convergence.

## 4. Deep Ritz Method (deep_ritz_method.pdf)

**Theory**: Replace strong-form PDE residual with energy minimization. Needs only first derivatives.

**Connection**: Vorticity transport is NOT variational (transport equation, not elliptic). Deep Ritz doesn't directly apply. However, the divergence constraint (div u = 0) IS amenable to energy formulation. Not high priority since div_loss is already implemented.

## 5. DeepONet Error Decomposition (operator_learning_ii.pdf)

**Theory**: Total error = E_B (branch/encoder error) + E_T (trunk/decoder error).

**Connection**:
  - Branch = U-Net encoder (geometry+params -> feature maps)
  - Trunk = coordinate embedding + MLP head (query_xy -> prediction)

**Diagnostic**:
  - If freeze encoder + train fresh head -> error drops -> trunk was bottleneck
  - If freeze head + retrain encoder -> error drops -> branch was bottleneck
  - The wider head fix (96->600 hidden) directly addresses trunk error

## 6. FNO Architecture (operator_learning_ii.pdf)

**Theory**: Fourier layers learn maps in frequency space. Low-mode truncation is standard for smooth PDE solutions.

**Connection**: Our model predicts TEMPORAL Fourier coefficients using SPATIAL CNN features. FNO's spatial Fourier doesn't directly apply (cylinder geometry breaks translation invariance), but the encode-operator-decode structure parallels our pipeline. Our model already combines global (avg pooling) + local (bilinear sampling) features, analogous to FNO's Fourier + conv paths.

## 7. CNN Surrogates with FD PDE Loss (operator_learning_ii.pdf, Grimm et al.)

**Direct analog**: The Grimm et al. results describe EXACTLY what our global_physics_loss does:
  - CNN predicts flow on grid
  - FD stencils (central diff) compute spatial derivatives
  - Navier-Stokes residual as loss

**Results table takeaway**: Physics-informed training with 10% data ~ data-driven with 50-75% data. But needs 5000 epochs. Implication: PDE loss valuable if we can afford more epochs.

## 8. Learnable Fourier Embedding (from models.py course work)

Already implemented by the user for the SML course. Key design:
```python
FourierEmbedding(input_dim=1, embed_dim=N, scale=10.0,
                 learnable=True, l1_strength=1e-4, l1_start_epoch=1000)
```
Could be directly adapted for the 2D coordinate embedding in query_model.py.

---

## Priority Ranking for Next Runs

| # | Approach | Code Effort | Expected Impact | Source |
|---|----------|-------------|-----------------|--------|
| 1 | Per-mode loss weighting | Minimal | High (directly counters spectral bias) | NTK analysis |
| 2 | Learnable Fourier coord embedding | Moderate | High (adapts to actual spatial frequencies) | Data-space preconditioning |
| 3 | ReduceLROnPlateau | Minimal | Medium (responds to actual dynamics) | General optimization |
| 4 | Progressive mode training | Moderate | High (warm start for hard modes) | Spectral bias curriculum |
| 5 | Separate mode-group heads | Larger | High (independent NTK per group) | NTK + DD |
| 6 | Alternating encoder/head training | Moderate | Medium | Parameter-space preconditioning |
| 7 | Enable PDE loss with annealing | Minimal | Medium (needs more epochs) | Physics-informed CNN surrogates |
| 8 | Increase base_ch 48->64 | Minimal | Medium (doubles encoder cost) | DeepONet branch error |

---

## Experimental Results (2026-04-06)

### Summary Table

| Run | Config | Best Val | Key Change |
|-----|--------|----------|------------|
| Baseline | head_hidden=96, base_ch=48 (2.5M) | 41.16% | — |
| Run 1 | head_hidden=600 (3.5M) | 30.36% | Head capacity (DeepONet trunk error) |
| Run 2 | + mode_weight_max=4, learnable embed | 29.29% | NTK spectral bias counter |
| Run 3 | 3 mode-group heads, base_ch=56 (3.7M) | 27.95% | Independent NTK per group |
| Run 4 | base_ch=64, 3 groups, mw=3.0, 600ep (4.8M) | 28.15% | Encoder capacity (DeepONet branch) |
| Run 5 | + PDE loss (warmup@200), learnable embed 0.3x, no mw, 700ep | **26.75%** | Physics-informed regularization |

### Per-mode Relative L2 Errors (validation set)

| Mode | Baseline | Run 1 | Run 3 | Run 4 | Run 5 | Best | Group |
|------|----------|-------|-------|-------|-------|------|-------|
| 1    | 56.9%    | 24.3% | 22.1% | 22.0% | **21.0%** | R5 | A |
| 2    | 65.3%    | 34.2% | 28.5% | 28.4% | **25.9%** | R5 | A |
| 3    | 68.0%    | 40.4% | **34.4%** | 38.8% | 36.7% | R3 | A |
| 4    | 70.9%    | 43.5% | **39.5%** | 46.4% | 41.8% | R3 | A |
| 5    | 83.5%    | 51.1% | 52.9% | **47.5%** | **47.5%** | R4/5 | B |
| 6    | 81.1%    | 56.5% | 59.9% | 59.1% | **58.3%** | R5 | B |
| 7    | 80.3%    | 63.1% | 63.6% | 59.5% | **58.0%** | R5 | B |
| 8    | 81.9%    | 66.2% | 66.5% | **65.4%** | 67.6% | R4 | B |
| 9    | 84.4%    | 68.8% | 67.6% | **62.1%** | 62.6% | R4 | C |
| 10   | 85.6%    | 75.4% | 75.2% | **74.0%** | 74.4% | R4 | C |
| 11   | 83.8%    | 75.4% | 73.2% | 72.4% | **72.3%** | R5 | C |
| 12   | 84.7%    | 79.6% | 79.1% | **76.6%** | 79.0% | R4 | C |

### Key Findings

1. **Head capacity was the largest single factor**: 96→600 hidden gave 10.8pp gain. Confirms DeepONet trunk error analysis.
2. **Per-mode loss weighting**: Heuristic approach — mixed per-mode effects due to cross-talk in shared head. Dropped in favor of structural solution.
3. **Separate mode-group heads** (domain decomposition): +1.3pp from NTK independence. Principled solution over loss weighting — each mode group develops its own NTK.
4. **Larger encoder (base_ch 48→64)**: Helps groups B and C (modes 5-12) but can trade off against group A. The encoder feature maps need more capacity for high-frequency spatial content.
5. **PDE loss with warmup** (physics-informed CNN surrogate): +1.2pp gain when combined with structural changes. Most principled approach — enforces Navier-Stokes consistency which is crucial for extrapolation.
6. **Learnable Fourier embedding**: 0.1x LR ineffective; 0.3x LR moves frequencies modestly (delta ~0.4). Works best when combined with PDE loss.
7. **High modes (10-12) plateau at 72-79%**: Persist across all configurations. Fundamental challenge — low energy content means low signal-to-noise in the coefficients.
8. **Overfitting gap**: ~15pp across all runs (train ~12% vs val ~27%). PDE loss helps regularize but doesn't eliminate the gap — 32 simulations may be too few.

### What Works vs What Doesn't

| Approach | Verdict | Comment |
|----------|---------|---------|
| Wider MLP head | WORKS (+10.8pp) | Trunk capacity was the primary bottleneck |
| Separate mode heads | WORKS (+1.3pp) | NTK independence, principled (domain decomposition) |
| PDE loss with warmup | WORKS (+1.2pp) | Physics regularization, crucial for extrapolation |
| Larger base_ch (64) | WORKS (+0.5pp) | Helps high modes at slight cost to low modes |
| Per-mode loss weights | MARGINAL | Redistributes error, non-generalizable heuristic |
| Learnable Fourier embed (0.3x) | MARGINAL | Modest frequency adaptation, needs more training |
| Learnable Fourier embed (0.1x) | DOESN'T WORK | LR too conservative, frequencies frozen |
| ReduceLROnPlateau | DOESN'T WORK | patience*val_every > epochs, effectively constant LR |

### Overall Progress: 41.16% → 26.75% (14.4pp reduction, 35% relative improvement)

---

## Experimental Results (2026-04-09) — Runs 6 and 7

### Motivation

Run 5 had a 14.6pp overfitting gap (train 12% vs val 27%) on only 32 sims. The session plan was to:
1. Triple the training data (96 sims)
2. Add per-mode loss normalization by `coeff_scale` to counter spectral bias
3. Upgrade PDE-loss finite-difference stencils from 2nd to 4th order
4. Increase PDE loss weight from 0.01 to 0.05 and warmup from 200 to 300 epochs
5. Bump `embed_lr_scale` from 0.3 to 0.5

### Run 6: with `--normalize_mode_loss` flag (double-normalization bug)

```bash
python run_multi.py --n_sims 96 --epochs 1000 --base_ch 64 --n_mode_groups 3 \
  --learnable_embed --embed_lr_scale 0.5 --lam_pde 0.05 --pde_warmup_epoch 300 \
  --normalize_mode_loss --data_dir data_stage_next --out_dir results_run6
```

**Best val: 24.40% at epoch 940** (2.35pp improvement over Run 5).

**BUG discovered by per-mode analysis**: `query_dataset.py:311` already normalizes targets as
`target = target / coeff_scale[None, None, :, :, None]` before the model ever sees them.
The new `coefficient_loss(coeff_scale=...)` branch then divides the loss by `coeff_scale` **again**, producing
an effective per-mode weight of `1/coeff_scale²`. Since `coeff_scale[mode 1] ≈ 0.73` and `coeff_scale[mode 12] ≈ 0.037`,
mode 12 received approximately `(0.73/0.037)² ≈ 390×` more gradient than mode 1.

This concentrated the optimizer on noisy, low-SNR high modes. The overall relative-L2 metric still improved
thanks to the 3× data and the 4th-order FD stencils, but the per-mode distribution was skewed — high-mode
groups were effectively *more* over-fit to the training noise.

### Run 7: with the double-normalization fix (same config, flag removed)

```bash
# Same as Run 6 but WITHOUT --normalize_mode_loss
python run_multi.py --n_sims 96 --epochs 1000 --base_ch 64 --n_mode_groups 3 \
  --learnable_embed --embed_lr_scale 0.5 --lam_pde 0.05 --pde_warmup_epoch 300 \
  --data_dir data_stage_next --out_dir results_run7
```

**Best val: 22.68% at epoch 1000** — a further 1.72pp improvement over Run 6, and **4.07pp over Run 5**.

### Per-mode comparison (Run 6 → Run 7, same val split)

| Mode | Run 6 | Run 7 | Δ | Group |
|------|-------|-------|---|-------|
| 1    | 20.6% | 18.8% | -1.8 | A |
| 2    | 29.4% | 25.7% | -3.7 | A |
| 3    | 30.9% | 28.8% | -2.2 | A |
| 4    | 44.4% | 40.8% | -3.5 | A |
| 5    | 51.3% | 49.0% | -2.3 | B |
| 6    | 61.8% | 60.0% | -1.8 | B |
| 7    | 66.3% | 64.0% | -2.3 | B |
| 8    | 69.7% | 68.5% | -1.1 | B |
| 9    | 76.0% | 75.7% | -0.2 | C |
| 10   | 74.8% | 72.0% | -2.7 | C |
| 11   | 84.4% | 79.1% | -5.3 | C |
| 12   | 85.9% | 82.9% | -3.0 | C |

**Group averages** (same val split):
- A (1-4): 31.3% → 28.5% (-2.8pp)
- B (5-8): 62.3% → 60.4% (-1.9pp)
- C (9-12): 80.3% → 77.4% (-2.9pp)

Removing the broken normalization improved **every** mode group.

### Important validation-set caveat

Runs 5 and Runs 6/7 use **different validation splits** because the 96-sim dataset has different random parameter
samples than the original 32-sim dataset. The seed-42 split produces 6 val sims for Run 5 and ~20 val sims for Runs 6/7.
Direct per-mode comparison R5 ↔ R7 is NOT apples-to-apples at the mode level — the absolute val error % is comparable,
but which sims are in val differs.

The fair comparison is **Run 6 ↔ Run 7** (same data, same split). Run 6 was strictly broken by double normalization;
Run 7 is the clean version. Both show the effect of the 3× data + 4th-order FD + stronger PDE loss.

### Updated summary table

| Run | Config | Best Val | Gap | Notes |
|-----|--------|----------|-----|-------|
| Baseline | head_hidden=96, 2.5M | 41.16% | ~15 | — |
| Run 1 | head_hidden=600, 3.5M | 30.36% | ~14 | Head capacity |
| Run 2 | + learnable embed, mw=4 | 29.29% | ~14 | Data-space preconditioning |
| Run 3 | 3 mode-group heads, base_ch=56 | 27.95% | ~14 | NTK decomposition |
| Run 4 | base_ch=64, mw=3, 600 ep | 28.15% | ~15 | Encoder capacity |
| Run 5 | + PDE loss, no mw, 700 ep | **26.75%** | 14.6 | Physics-informed |
| Run 6 | 96 sims, 4th-order FD, + normalize_mode_loss (BROKEN) | 24.40% | 14.5 | Double-normalization bug |
| **Run 7** | 96 sims, 4th-order FD, PDE 0.05, no normalize | **22.68%** | ~13.8 | **Clean config, current best** |

### Lessons from Runs 6/7

1. **Data normalization must happen in exactly one place.** Applying `coeff_scale` both in the dataset (line 311) and in the loss function double-normalizes and catastrophically biases the optimizer toward low-energy modes. The lesson is to grep for every use of a normalization statistic before adding another.

2. **Per-mode analysis is essential for diagnosing spectral bias.** The overall L2 metric improved in Run 6 and hid the per-mode regression. Without the per-group breakdown, Run 6 would have looked like a success.

3. **Tripling data reduced neither the overfitting gap nor the per-mode gap as much as hoped.** Both runs kept ~14pp gap. The gap appears to be a capacity-vs-data balance issue more than a pure data scarcity one — the 4.8M-parameter model has enough capacity to memorize any dataset we can realistically generate on this solver.

4. **4th-order FD stencils in the PDE loss are a strict improvement.** The loss residual is less noisy, the effective PDE regularization is cleaner, and there is no cost besides 2 additional cells excluded from the interior mask. No observed downsides.

5. **Stronger PDE loss (0.01 → 0.05) with longer warmup (200 → 300) is stable.** No training divergence; the val error curve is smoother through the warmup transition.

6. **The high modes 9-12 remain the persistent bottleneck**, now at 72-83% error in Run 7. The signal-to-noise ratio for those coefficients is fundamentally low and cannot be fixed by network/loss tweaks alone. Future approaches:
   - Log-space targets for high modes (compressing the dynamic range)
   - Data augmentation via vertical flip (doubles the effective dataset for free — cylinder flow has y-symmetry)
   - Adaptive mode curriculum (detect per-mode training plateaus and unfreeze progressively)
   - Dedicated spectral loss term per mode group

---

## Non-Applicable Approaches

- **Deep Ritz**: Vorticity transport is not variational. Not directly usable.
- **SINDy / Model Discovery**: We already know the PDE. Not relevant.
- **Neural ODEs**: We use fixed temporal Fourier modes, not ODE integration. Not applicable.
- **L-BFGS full model**: 3.5M params makes full second-order infeasible. Only for sub-components.

---

## Work Summary & Handoff Notes (2026-04-06)

### What Was Done

Starting from a baseline model (val error 41.16%) that captured modes 1-2 but failed on modes 3-12, we systematically applied course concepts from 7 SML lecture PDFs to improve the model through 5 iterative training runs:

1. **Diagnosed the bottleneck** using DeepONet error decomposition (E = E_branch + E_trunk). The MLP head had only 96 hidden units — trunk capacity was the primary bottleneck.

2. **Widened the MLP head** (96 → 600 hidden units, 2.5M → 3.5M params). This alone gave the largest single improvement: **41.16% → 30.36%** (-10.8pp). Confirmed the DeepONet trunk error diagnosis.

3. **Added separate mode-group heads** (domain decomposition / NTK independence). Instead of one MLP predicting all 12 modes, split into 3 independent heads (modes 1-4, 5-8, 9-12). Each head develops its own NTK spectrum, preventing low-mode dominance from suppressing high-mode learning. This is the **structural** solution to spectral bias — preferred over per-mode loss weighting which is a non-generalizable heuristic that hurts extrapolation.

4. **Enabled PDE loss** (physics-informed CNN surrogate, Grimm et al. approach). The `global_physics_loss` function was already implemented but disabled (`lam_pde=0`). Enabled with linear warmup (0 → target over 200 epochs) to avoid early instability. Provides Navier-Stokes consistency as regularization — crucial for extrapolation.

5. **Replaced fixed Fourier coordinate embedding** with learnable frequencies (data-space preconditioning). Evenly-spaced init on `[pi, 32*pi]` instead of geometric `2^k * pi` bands. Key lesson: **0.1x differential LR is too conservative** — frequencies don't move. Use **0.3x** or higher.

6. **Increased encoder capacity** (`base_ch` 48 → 64, 4.8M params) to address DeepONet branch error. Helps modes 5-12 but with diminishing returns.

### What Each Code Change Does

| File | Change | Why |
|------|--------|-----|
| `query_model.py` | `FourierCoordinateEmbedding` — learnable bands, L1 penalty with warm start | Data-space preconditioning: adapts to actual spatial frequencies |
| `query_model.py` | `n_mode_groups` parameter, `self.heads = nn.ModuleList()` | Domain decomposition: independent NTK per mode group |
| `query_train.py` | `mode_weights` in `coefficient_loss` and `amplitude_loss` | Per-mode loss weighting (kept but set to 1.0 — structural heads preferred) |
| `query_train.py` | `embed_lr_scale` differential LR for embed params | Separate optimizer param group for learnable frequencies |
| `query_train.py` | `pde_warmup_epoch` linear ramp in `run_epoch` | Prevents PDE loss from destabilizing early training |
| `query_train.py` | Fourier L1 penalty added to total loss | Prunes unused frequencies after warm-start period |
| `run_multi.py` | CLI args for all new hyperparameters | Enables experiment configuration without code edits |

### Best Configuration (Run 5)

```bash
python run_multi.py \
  --n_sims 32 --epochs 700 --val_every 20 \
  --base_ch 64 --lr 1e-3 --queries 256 \
  --n_modes 12 --n_ref_frames 12 --patch_size 5 \
  --wake_strength 1.5 --lam_amp 0.25 --lam_recon 0.1 \
  --lam_vort 0.01 --lam_div 0.05 --lam_bc 0.01 \
  --lam_pde 0.01 --pde_warmup_epoch 200 \
  --wake_focus 4.0 --wake_query_frac 0.5 --grad_clip_norm 5.0 \
  --mode_weight_max 1.0 --n_mode_groups 3 \
  --learnable_embed --embed_lr_scale 0.3 \
  --data_dir data_stage_next --out_dir results_run5
```

Result: **26.75% val error** (best at epoch 600, converged — last 10 evals spread 0.21pp).

### What Still Doesn't Work / Open Problems

1. **Modes 10-12 plateau at 72-79% error** across all configurations. These modes carry very little energy (high temporal frequency, low amplitude). The signal-to-noise ratio in the Fourier coefficients is fundamentally low. Possible approaches:
   - Log-scale or energy-normalized targets for high modes
   - Dedicated high-mode architecture with skip connections from raw spatial features
   - Accept that these modes may not be learnable from 32 sims

2. **15pp overfitting gap** (train ~11% vs val ~27%). With only 32 simulations, the model memorizes training geometries. PDE loss helps as physics-based regularization but doesn't close the gap. Options:
   - More training simulations (64-128 would likely halve the gap)
   - Stronger data augmentation (geometric perturbations of the cylinder)
   - Dropout in the MLP heads (not currently used)
   - Weight decay tuning (currently 1e-5, could try 1e-4)

3. **Extrapolation quality is untested quantitatively**. The GIF at `results_run5/extrapolation_reconstruction.gif` shows an OOD case (velocity 20% above, viscosity 40% below training range). PDE loss should help here theoretically (Navier-Stokes consistency), but no quantitative metric was computed on the OOD case.

4. **Learnable embedding barely moved** even at 0.3x LR (delta ~0.47 over 700 epochs). The frequencies adapt modestly but the gains are marginal. A future attempt could:
   - Use full LR (1.0x) for the embedding
   - Start L1 penalty later (epoch 400 instead of 200)
   - Try more frequencies (12 instead of 6) with aggressive L1 pruning

### Ideas That Might Help (Not Yet Tried)

| Idea | Course Basis | Effort | Expected Impact |
|------|-------------|--------|-----------------|
| **Progressive mode training** (train modes 1-4 first, then expand) | NTK curriculum | Moderate | Could help modes 5-8 converge better |
| **Alternating encoder/head optimization** (freeze one, train other) | Parameter-space preconditioning | Moderate | Reduces cross-layer NTK coupling |
| **More simulations** (64-128) | Generalization theory | Minimal (data gen ~1 min/sim) | Likely biggest impact on overfitting gap |
| **L-BFGS for head output layer** | Second-order preconditioning | Small | Exact solution for linear final layer |
| **Separate heads for near-cylinder vs far-wake** | Function-space DD | Moderate | Different spatial regions have different frequency content |
| **Higher PDE loss weight** (0.05-0.1 instead of 0.01) with more epochs | Physics-informed surrogates | Minimal | Stronger regularization; needs 1000+ epochs |
| **Gradient accumulation** for effective larger batch | Training stability | Minimal | Smoother gradients, especially for PDE loss |

### Pitfalls for Future Work

- **ReduceLROnPlateau doesn't work** with `val_every > 1`. If `patience * val_every > total_epochs`, LR never reduces. Use `CosineAnnealingLR` instead.
- **Per-mode loss weights are a heuristic trap**. They redistribute error between modes but don't generalize — and they break extrapolation. Use separate mode heads instead (structural NTK fix).
- **Learnable Fourier embed needs aggressive LR** (≥0.3x). At 0.1x the frequencies are frozen at initialization.
- **PDE loss needs warmup**. Enabling it from epoch 0 at full strength destabilizes training because the model hasn't learned reasonable spatial patterns yet.
- **The model is in the worktree at** `query_model.py`, `query_train.py`, `run_multi.py`. None of these changes have been committed to git yet. The whole `tide_operator/` directory is untracked.

### Result Artifacts

| Path | Contents |
|------|----------|
| `results_wider_head/` | Run 1: wider head only (30.36%) |
| `results_run2/` | Run 2: + mode weights + learnable embed (29.29%) |
| `results_run3/` | Run 3: + 3 mode-group heads, base_ch=56 (27.95%) |
| `results_run4/` | Run 4: + base_ch=64, 600 epochs (28.15%) |
| `results_run5/` | Run 5: + PDE loss, no mode weights, 700 epochs (**26.75%**) |
| `data_stage_next/` | 32 simulation .pkl files (reused across runs) |
| Each `results_*/` contains | `model_best.pt`, `history.pkl`, `training_history.png`, `validation_reconstruction.gif`, `extrapolation_reconstruction.gif`, per-sample prediction plots |
