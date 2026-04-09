"""
run_multi.py - One-click multi-simulation training + evaluation.

Usage:
    python run_multi.py
    python run_multi.py --n_sims 10 --epochs 300
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")

import torch

from generate import (
    generate_tide_field,
    prepare_velocity_mode_targets,
    save_simulation,
)
from query_model import TideFourierOperator
from query_train import DEFAULT_CONFIG, train
from query_visualize import (
    animate_reconstruction,
    animate_reconstruction_from_data,
    plot_gradient_flow,
    plot_loss_breakdown,
    plot_prediction_vs_truth,
    plot_simulation_overview,
    plot_training_history,
    plot_weight_distributions,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_sims", type=int, default=30)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--val_every", type=int, default=10)
    p.add_argument("--base_ch", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--queries", type=int, default=256)
    p.add_argument("--val_batch", type=int, default=1)
    p.add_argument("--n_modes", type=int, default=12)
    p.add_argument("--n_ref_frames", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=5)
    p.add_argument("--wake_strength", type=float, default=1.5)
    p.add_argument("--lam_amp", type=float, default=0.25)
    p.add_argument("--lam_recon", type=float, default=0.25)
    p.add_argument("--lam_vort", type=float, default=0.0)
    p.add_argument("--lam_div", type=float, default=0.05)
    p.add_argument("--lam_bc", type=float, default=0.0)
    p.add_argument("--lam_pde", type=float, default=0.0)
    p.add_argument("--lam_eq", type=float, default=0.0)
    p.add_argument("--wake_focus", type=float, default=4.0)
    p.add_argument("--wake_query_frac", type=float, default=0.5)
    p.add_argument("--recon_query_chunk_size", type=int, default=128)
    p.add_argument("--grad_clip_norm", type=float, default=15.0)
    p.add_argument("--pde_batch_size", type=int, default=1)
    p.add_argument("--pde_time_samples", type=int, default=8)
    p.add_argument("--pde_query_chunk_size", type=int, default=1024)
    p.add_argument("--pde_div_weight", type=float, default=1.0)
    p.add_argument("--pde_vort_weight", type=float, default=1.0)
    p.add_argument("--pde_warmup_epoch", type=int, default=0,
                   help="Linear PDE loss warmup from 0 to lam_pde over this many epochs")
    p.add_argument("--embed_lr_scale", type=float, default=0.3,
                   help="Differential LR scale for learnable Fourier embed")
    p.add_argument("--mode_weight_max", type=float, default=1.0,
                   help="Max per-mode loss weight (1.0=uniform, >1 upweights high modes)")
    p.add_argument("--learnable_embed", action="store_true",
                   help="Use learnable Fourier coordinate embedding")
    p.add_argument("--n_mode_groups", type=int, default=1,
                   help="Separate MLP heads per mode group (1=single, 3=three groups)")
    p.add_argument("--normalize_mode_loss", action="store_true",
                   help="Normalize per-mode loss by coeff_scale (relative error per mode)")
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau"])
    p.add_argument("--plateau_patience", type=int, default=30)
    p.add_argument("--force_regen", action="store_true")
    p.add_argument("--data_dir", type=str, default="data_multi/")
    p.add_argument("--out_dir", type=str, default="results_multi/")
    p.add_argument("--ood_velocity", type=float, default=None)
    p.add_argument("--ood_viscosity", type=float, default=None)
    p.add_argument("--ood_radius", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(dict(
        n_sims=args.n_sims,
        epochs=args.epochs,
        val_every=args.val_every,
        n_modes=args.n_modes,
        n_ref_frames=args.n_ref_frames,
        patch_size=args.patch_size,
        base_ch=args.base_ch,
        lr=args.lr,
        queries_per_sample=args.queries,
        val_batch_size=args.val_batch,
        wake_strength=args.wake_strength,
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
        pde_warmup_epoch=args.pde_warmup_epoch,
        embed_lr_scale=args.embed_lr_scale,
        data_dir=args.data_dir,
        output_dir=args.out_dir,
        force_regen=args.force_regen,
        log_every=args.val_every,
        batch_size=8,
        vel_range=(4.0, 8.0),
        visc_range=(0.05, 0.25),
        rad_range=(6.0, 11.0),
        mode_weight_max=args.mode_weight_max,
        learnable_coord_embed=args.learnable_embed,
        n_mode_groups=args.n_mode_groups,
        scheduler=args.scheduler,
        plateau_patience=args.plateau_patience,
        normalize_mode_loss=args.normalize_mode_loss,
    ))

    model, history, bundle = train(cfg)
    train_dataset = bundle["train_dataset"]
    val_dataset = bundle["val_dataset"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_best = TideFourierOperator(
        n_modes=cfg["n_modes"],
        n_params=3,
        base_ch=cfg["base_ch"],
        in_channels=train_dataset.geom_channels,
        n_components=2,
        patch_size=cfg["patch_size"],
        learnable_coord_embed=cfg.get("learnable_coord_embed", False),
        n_mode_groups=cfg.get("n_mode_groups", 1),
    ).to(device)
    model_best.load_state_dict(torch.load(out / "model_best.pt", map_location=device, weights_only=True))
    model_best.eval()

    print("\nGenerating diagnostic plots...")

    history_pkl = str(out / "history.pkl")
    plot_training_history(history_pkl, save_path=str(out / "training_history.png"))
    plot_loss_breakdown(history_pkl, save_path=str(out / "loss_breakdown.png"))
    plot_gradient_flow(history_pkl, save_path=str(out / "gradient_flow.png"))
    plot_weight_distributions(history_pkl, save_path=str(out / "weight_distributions.png"))

    plot_simulation_overview(bundle["train_paths"][0], save_path=str(out / "sim_overview.png"))

    val_indices = list(range(min(3, len(val_dataset))))
    for i in val_indices:
        plot_prediction_vs_truth(
            model_best,
            val_dataset,
            sample_idx=i,
            device=device,
            save_path=str(out / f"pred_vs_truth_sample{i}.png"),
        )

    print("Rendering validation and extrapolation videos...")

    val_sim_path = str(pathlib.Path(bundle["val_paths"][0]))

    animate_reconstruction(
        val_sim_path,
        model=model_best,
        dataset=val_dataset,
        sample_idx=0,
        device=device,
        save_path=str(out / "validation_reconstruction.gif"),
        fps=15,
        title_extra="Validation sample",
    )

    ood_velocity = args.ood_velocity if args.ood_velocity is not None else cfg["vel_range"][1] * 1.20
    ood_viscosity = args.ood_viscosity if args.ood_viscosity is not None else max(1e-4, cfg["visc_range"][0] * 0.60)
    ood_radius = args.ood_radius if args.ood_radius is not None else 0.5 * (cfg["rad_range"][0] + cfg["rad_range"][1])

    field, target_field, base_field, mask, sdf, params, pop_field, target_pop_field, base_pop_field = generate_tide_field(
        nx=cfg["nx"],
        ny=cfg["ny"],
        nt=cfg["nt"],
        radius=ood_radius,
        velocity=ood_velocity,
        viscosity=ood_viscosity,
        wake_strength=cfg["wake_strength"],
        noise=cfg["noise"],
        seed=1234,
    )
    target_info = prepare_velocity_mode_targets(
        target_field=target_field,
        mask=mask,
        nt=cfg["nt"],
        n_modes=cfg["n_modes"],
        n_ref_frames=train_dataset.n_ref_frames,
    )
    ood_data = dict(
        field=field,
        target_field=target_field,
        base_field=base_field,
        field_components=("u", "v"),
        pop_field=pop_field,
        target_pop_field=target_pop_field,
        base_pop_field=base_pop_field,
        population_components=tuple(f"f{i}" for i in range(pop_field.shape[1])),
        mask=mask,
        sdf=sdf,
        params=params,
        fft_field=target_info["fft_field"].astype("complex64"),
        freqs=target_info["freqs"],
        mode_idx=target_info["mode_idx"],
        mode_coeffs=target_info["mode_coeffs"],
        mode_space="velocity",
        ref_field=target_info["ref_field"],
        ref_times=target_info["ref_times"],
        wake_energy_map=target_info["wake_energy_map"],
        ref_scale=target_info["ref_scale"],
    )
    save_simulation(
        out / "extrapolation_case.pkl",
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

    animate_reconstruction_from_data(
        ood_data,
        model_best,
        train_dataset,
        device=device,
        save_path=str(out / "extrapolation_reconstruction.gif"),
        fps=15,
        title_extra="Extrapolation sample",
    )

    best_val = min(history["val_err"])
    final_val = history["val_err"][-1]

    print(f"""
==========================================
Multi-sim run complete
------------------------------------------
Simulations  : {args.n_sims}
Epochs       : {args.epochs}
Best val err : {best_val:.2f}%
Final val err: {final_val:.2f}%
Outputs in   : {str(out)}
Validation video    : {str(out / 'validation_reconstruction.gif')}
Extrapolation video : {str(out / 'extrapolation_reconstruction.gif')}
==========================================
""")


if __name__ == "__main__":
    main()
