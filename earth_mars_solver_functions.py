"""
earth_mars_transfer_solver.py
=============================

Lean interface for solving Earth-Mars transfers and generating
training datasets.

    solve_transfer()      — single solve, returns numpy arrays
    generate_dataset()    — batch solve with pre-allocated storage

Requires: earth_mars_transfer_helpers.py, SPICE kernels loaded.
"""

from datetime import datetime
import time as time_module
import matplotlib.pyplot as plt
import numpy as np
from tudatpy import constants
from tudatpy.dynamics import propagation_setup

from tudatpy.interface import spice

from earth_mars_transfer_helpers import (
    create_simulation_bodies,
    get_lambert_problem_result,
    get_lambert_arc_history,
    find_propagation_time_soi,
    propagate_trajectory,
    propagate_variational_equations,
)

# Load SPICE kernels once when this module is imported.
spice.load_standard_kernels()


###########################################################################
# UTILITIES
###########################################################################

def calendar_to_epoch(year, month, day, hour=0, minute=0, second=0):
    """Calendar date → seconds since J2000."""
    j2000 = datetime(2000, 1, 1, 12, 0, 0)
    dt = datetime(year, month, day, hour, minute, second)
    return (dt - j2000).total_seconds()

def plot_transfers(lamb, soi, sample_indices=None, title="Lambert transfers (XY plane)"):
    """
    lamb: (M, N+1, 6)
    soi:  (M, 2)
    sample_indices: which samples to plot, default all
    """
    AU = 1.496e11
    if sample_indices is None:
        sample_indices = range(lamb.shape[0])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(0, 0, s=100, color="gold", zorder=10, label="Sun")
    
    for i in sample_indices:
        x = lamb[i, :, 0] / AU
        y = lamb[i, :, 1] / AU
        ax.plot(x, y, linewidth=0.5, alpha=0.6)
        ax.scatter(x[0], y[0], s=15, color="blue", zorder=5)   # departure
        ax.scatter(x[-1], y[-1], s=15, color="red", zorder=5)   # arrival
    
    ax.set_xlabel("X [AU]")
    ax.set_ylabel("Y [AU]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend()
    plt.show()
###########################################################################
# SINGLE SOLVE
###########################################################################

def solve_transfer(
    departure_epoch,
    tof_days,
    number_of_arcs,
    target_body="Mars",
    max_deviation=1.0,
    max_iterations=120,
    verbose=False,
):
    """
    Solve for impulsive Dv corrections on a perturbed Lambert transfer.

    Parameters
    ----------
    departure_epoch : float       seconds since J2000
    tof_days        : float       time of flight [days]
    number_of_arcs  : int         sub-arcs between SOI exit and entry
    target_body     : str         default "Mars"
    max_deviation   : float       Gauss-Newton tolerance on ||dx_6D||
    max_iterations  : int         max GN iterations per arc
    verbose         : bool

    Returns
    -------
    soi_epochs      : (2,)        [soi_dep_epoch, soi_arr_epoch]
    lambert_states  : (N+1, 6)    Lambert state at each arc boundary
    perturbed_states: (N+1, 6)    actual propagated state at each boundary
    corrections     : (N, 3)      Dv [m/s] at each arc start
    arc_length      : float       duration of each arc [s]
    """
    arrival_epoch = departure_epoch + tof_days * constants.JULIAN_DAY

    bodies = create_simulation_bodies()
    lambert_eph = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )

    soi_dep, _, soi_arr, _, soi_tof = find_propagation_time_soi(
        lambert_eph, bodies,
        departure_epoch, "Earth", target_body, arrival_epoch,
    )

    arc_length = soi_tof / number_of_arcs
    arc_epochs = np.linspace(soi_dep, soi_arr, number_of_arcs + 1)

    # Lambert states at boundaries
    lambert_states = np.empty((number_of_arcs + 1, 6))
    for i, ep in enumerate(arc_epochs):
        lambert_states[i] = lambert_eph.cartesian_state(ep)

    # Allocate outputs
    perturbed_states = np.empty((number_of_arcs + 1, 6))
    perturbed_states[0] = lambert_states[0]
    corrections = np.empty((number_of_arcs, 3))

    current_state_correction = np.zeros(6)

    for arc_idx in range(number_of_arcs):

        t0 = arc_epochs[arc_idx]
        tf = arc_epochs[arc_idx + 1]
        term = propagation_setup.propagator.time_termination(tf)

        # Variational equations (no RSW) → STM
        var_solver = propagate_variational_equations(
            t0, term, bodies, lambert_eph,
            initial_state_correction=current_state_correction,
            use_rsw_acceleration=False,
        )
        stm_hist = var_solver.state_transition_matrix_history
        state_hist = var_solver.state_history
        lh = get_lambert_arc_history(lambert_eph, state_hist)
        final_epoch = list(stm_hist.keys())[-1]

        control_matrix = stm_hist[final_epoch][:, 3:6]
        delta_x = state_hist[final_epoch] - lh[final_epoch]

        # Least-squares initial guess
        correction, _, _, _ = np.linalg.lstsq(control_matrix, -delta_x, rcond=None)

        # Gauss-Newton
        it = 0
        while True:
            it += 1
            impulse = np.concatenate([np.zeros(3), correction])
            sim = propagate_trajectory(
                t0, term, bodies, lambert_eph,
                initial_state_correction=current_state_correction + impulse,
                use_rsw_acceleration=False,
            )
            sh = sim.propagation_results.state_history
            lh_sim = get_lambert_arc_history(lambert_eph, sh)
            sh_arr = np.vstack(list(sh.values()))
            lh_arr = np.vstack(list(lh_sim.values()))

            delta_x_new = sh_arr[-1] - lh_arr[-1]
            if np.linalg.norm(delta_x_new) <= max_deviation:
                break
            if it >= max_iterations:
                break
            dp, _, _, _ = np.linalg.lstsq(control_matrix, -delta_x_new, rcond=None)
            correction = correction + dp

        corrections[arc_idx] = correction
        perturbed_states[arc_idx + 1] = lh_arr[-1] + delta_x_new
        current_state_correction = delta_x_new.copy()

        if verbose and (arc_idx == 0 or (arc_idx + 1) % max(number_of_arcs // 10, 1) == 0):
            print(f"  [{arc_idx+1}/{number_of_arcs}]  it={it}  ||dx||={np.linalg.norm(delta_x_new):.2e}")

    soi_epochs = np.array([soi_dep, soi_arr])
    return soi_epochs, lambert_states, perturbed_states, corrections, arc_length


###########################################################################
# BATCH DATASET GENERATION
###########################################################################

def generate_dataset(
    departure_epochs,
    tof_days_array,
    number_of_arcs,
    target_body="Mars",
    max_deviation=1.0,
    max_iterations=120,
    verbose=True,
):
    """
    Generate a training dataset for a fixed arc count.

    Parameters
    ----------
    departure_epochs : (M,) array   departure epochs [s since J2000]
    tof_days_array   : (M,) array   time of flight per sample [days]
    number_of_arcs   : int          fixed for all samples
    target_body      : str
    max_deviation    : float
    max_iterations   : int
    verbose          : bool

    Returns
    -------
    soi_epochs       : (M, 2)            [soi_dep, soi_arr] per sample
    lambert_states   : (M, N+1, 6)       Lambert states at arc boundaries
    perturbed_states : (M, N+1, 6)       propagated states at arc boundaries
    corrections      : (M, N, 3)         Dv per arc
    arc_lengths      : (M,)              arc duration per sample [s]
    """
    M = len(departure_epochs)
    N = number_of_arcs

    soi_epochs_all = np.empty((M, 2))
    lambert_all = np.empty((M, N + 1, 6))
    perturbed_all = np.empty((M, N + 1, 6))
    corrections_all = np.empty((M, N, 3))
    arc_lengths_all = np.empty(M)

    t_start = time_module.time()

    for i in range(M):
        soi_ep, lamb, pert, corr, al = solve_transfer(
            departure_epochs[i],
            tof_days_array[i],
            number_of_arcs,
            target_body=target_body,
            max_deviation=max_deviation,
            max_iterations=max_iterations,
            verbose=False,
        )

        soi_epochs_all[i] = soi_ep
        lambert_all[i] = lamb
        perturbed_all[i] = pert
        corrections_all[i] = corr
        arc_lengths_all[i] = al

        if verbose:
            elapsed = time_module.time() - t_start
            rate = elapsed / (i + 1)
            remaining = rate * (M - i - 1)
            total_dv = np.sum(np.linalg.norm(corr, axis=1))
            print(
                f"  [{i+1}/{M}]  "
                f"tof={tof_days_array[i]:.0f}d  "
                f"Dv={total_dv:.1f} m/s  "
                f"~{remaining/60:.1f}m left"
            )

    return soi_epochs_all, lambert_all, perturbed_all, corrections_all, arc_lengths_all