"""
Earth-Mars Low-Thrust Transfer via Lambert Arc Subdivision
==========================================================

Approach
--------
1. Solve the Lambert problem for an Earth -> Mars transfer with a
   user-defined time of flight (default: 4 months).
2. Trim the resulting arc to the segment between first exit of
   Earth's SOI and first entry into Mars' SOI.
3. Divide this heliocentric segment into N sub-arcs (default: 1000).
4. For each sub-arc, starting from the *exact* Lambert state:
   - Propagate the variational equations (perturbed dynamics,
     RSW empirical acceleration enabled with magnitude [0,0,0])
     to obtain the 6x3 sensitivity matrix  S = dx/dp  at the
     arc's final epoch.
   - Compute the 6-D state deviation  dx = x_prop - x_lambert
     at the final epoch (no thrust applied yet).
   - Solve for the RSW thrust via least-squares (6 equations,
     3 unknowns):  p = argmin ||S p + dx||^2 .
   - Re-propagate with the computed thrust.  If ||dx_new|| > tol,
     apply a Gauss-Newton correction  dp = lstsq(S, -dx_new)
     and repeat until convergence.
5. Store and plot the piecewise-constant RSW thrust profile and
   the residual state deviations.

Files required
--------------
earth_mars_transfer_helpers.py  (same directory)

Usage
-----
    python earth_mars_low_thrust_transfer.py
"""

from datetime import datetime
import time as time_module

import numpy as np
import matplotlib.pyplot as plt
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy.dynamics import propagation_setup

from earth_mars_transfer_helpers import *
#     create_simulation_bodies,
#     get_lambert_problem_result,
#     get_lambert_arc_history,
#     find_propagation_time_soi,
#     propagate_trajectory,
#     propagate_variational_equations,
# )

# Load SPICE kernels (must be called before any ephemeris queries)
spice.load_standard_kernels()


###########################################################################
# USER-CONFIGURABLE PARAMETERS
###########################################################################

# ---- Departure date (calendar) ----
departure_year   = 2028
departure_month  = 11
departure_day    = 15
departure_hour   = 0
departure_minute = 0

# ---- Time of flight ----
time_of_flight_days = 120          

# ---- Transfer configuration ----
target_body = "Mars"
number_of_arcs = 500
max_state_deviation_tolerance = 1.0    
max_iterations_per_arc = 120

# ---- Thrust mode: "continuous" or "impulsive" ----
# continuous: piecewise-constant RSW empirical acceleration per arc
#             control matrix = sensitivity matrix S (6x3)
# impulsive:  delta-v applied at the start of each arc
#             control matrix = Phi[:, 3:6] from STM (6x3)
thrust_mode = "impulsive"


###########################################################################
# DERIVED EPOCHS  (seconds since J2000, TDB approximation)


_j2000_dt = datetime(2000, 1, 1, 12, 0, 0)          # J2000 reference
_dep_dt   = datetime(departure_year, departure_month, departure_day,
                     departure_hour, departure_minute, 0)

departure_epoch = (_dep_dt - _j2000_dt).total_seconds()
time_of_flight  = time_of_flight_days * constants.JULIAN_DAY   # JULIAN_DAY = 86400 s
arrival_epoch   = departure_epoch + time_of_flight

print(f"Departure epoch : {departure_epoch/constants.JULIAN_DAY:.4f} JD from J2000  "
      f"({_dep_dt.strftime('%Y-%m-%d %H:%M')})")
print(f"Arrival epoch   : {arrival_epoch/constants.JULIAN_DAY:.4f} JD from J2000")
print(f"Time of flight  : {time_of_flight_days} days")


if __name__ == "__main__":

    print("EARTH-MARS LOW-THRUST TRANSFER VIA LAMBERT ARC SUBDIVISION")
    print(f"Thrust mode: {thrust_mode}")
    print("-" * 70)

    # -----------------------------------------------------------------
    # 1. Create body system and solve Lambert problem
    # -----------------------------------------------------------------
    bodies = create_simulation_bodies()

    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )

    dep_state = lambert_arc_ephemeris.cartesian_state(departure_epoch)
    arr_state = lambert_arc_ephemeris.cartesian_state(arrival_epoch)
    print(f"\nLambert departure |r| : {np.linalg.norm(dep_state[0:3])/1e9:.4f} Gm")
    print(f"Lambert arrival   |r| : {np.linalg.norm(arr_state[0:3])/1e9:.4f} Gm")
    print(f"Lambert departure |v| : {np.linalg.norm(dep_state[3:6])/1e3:.4f} km/s")
    print(f"Lambert arrival   |v| : {np.linalg.norm(arr_state[3:6])/1e3:.4f} km/s")

    # -----------------------------------------------------------------
    # 2. Trim to SOI-exit -> SOI-entry
    # -----------------------------------------------------------------
    (soi_dep_epoch, soi_dep_state,
     soi_arr_epoch, soi_arr_state,
     soi_tof) = find_propagation_time_soi(
        lambert_arc_ephemeris, bodies,
        departure_epoch, "Earth", target_body, arrival_epoch,
    )

    print(f"\nSOI departure : {soi_dep_epoch/constants.JULIAN_DAY:.4f} JD from J2000")
    print(f"SOI arrival   : {soi_arr_epoch/constants.JULIAN_DAY:.4f} JD from J2000")
    print(f"SOI-to-SOI TOF: {soi_tof/constants.JULIAN_DAY:.4f} days "
          f"({soi_tof:.0f} s)")

    # -----------------------------------------------------------------
    # 3. Define sub-arc boundaries

    arc_length = soi_tof / number_of_arcs
    arc_epochs = np.linspace(soi_dep_epoch, soi_arr_epoch, number_of_arcs + 1)

    print(f"\nNumber of arcs : {number_of_arcs}")
    print(f"Arc length     : {arc_length:.2f} s  ({arc_length/3600:.2f} h)")

    # Quick sanity check against integration step size
    steps_per_arc = arc_length / fixed_step_size
    print(f"RK4 steps/arc  : ~{steps_per_arc:.1f}  (step size = {fixed_step_size} s)")
    if steps_per_arc < 5:
        print("  *** WARNING: fewer than 5 integration steps per arc. "
              "Consider reducing fixed_step_size in the helpers file. ***")

    # -----------------------------------------------------------------
    # 4. Allocate storage
    # -----------------------------------------------------------------
    # For continuous: stores RSW acceleration [m/s^2]
    # For impulsive:  stores delta-v [m/s]
    correction_history = np.zeros((number_of_arcs, 3))
    final_state_deviation_norm = np.zeros(number_of_arcs)
    final_pos_deviation_norm = np.zeros(number_of_arcs)
    final_vel_deviation_norm = np.zeros(number_of_arcs)
    iterations_per_arc = np.zeros(number_of_arcs, dtype=int)
    arc_midpoint_epochs = 0.5 * (arc_epochs[:-1] + arc_epochs[1:])
    current_state_correction = np.zeros(6)
    # -----------------------------------------------------------------
    # 5. Main loop over sub-arcs
    # -----------------------------------------------------------------
    print(f"\n{'-'*70}")
    print("Starting arc-by-arc thrust computation …")
    print(f"{'-'*70}\n")

    t_loop_start = time_module.time()

    for arc_idx in range(number_of_arcs):

        t0 = arc_epochs[arc_idx]
        tf = arc_epochs[arc_idx + 1]

        termination_settings = propagation_setup.propagator.time_termination(tf)

        # ---- 5a. Propagate variational equations ----
        #
        # continuous: with RSW enabled (p=0) → sensitivity matrix S (6x3)
        # impulsive:  without RSW          → STM Phi (6x6), use Phi[:,3:6] (6x3)
        #
        if thrust_mode == "continuous":
            variational_solver = propagate_variational_equations(
                t0,
                termination_settings,
                bodies,
                lambert_arc_ephemeris,
                initial_state_correction=current_state_correction,
                use_rsw_acceleration=True,
            )
            sensitivity_hist = variational_solver.sensitivity_matrix_history
            state_hist       = variational_solver.state_history
            lambert_hist     = get_lambert_arc_history(lambert_arc_ephemeris, state_hist)
            final_epoch      = list(sensitivity_hist.keys())[-1]

            # S_full : 6x3  (maps RSW acceleration -> 6-D state deviation)
            control_matrix = sensitivity_hist[final_epoch]

        else:  # impulsive
            variational_solver = propagate_variational_equations(
                t0,
                termination_settings,
                bodies,
                lambert_arc_ephemeris,
                initial_state_correction=current_state_correction,
                use_rsw_acceleration=False,
            )
            stm_hist    = variational_solver.state_transition_matrix_history
            state_hist  = variational_solver.state_history
            lambert_hist = get_lambert_arc_history(lambert_arc_ephemeris, state_hist)
            final_epoch = list(stm_hist.keys())[-1]

            # Phi_v : 6x3  (maps delta-v at t0 -> 6-D state deviation at tf)
            control_matrix = stm_hist[final_epoch][:, 3:6]

        # 6-D state deviation without any correction
        delta_x = state_hist[final_epoch] - lambert_hist[final_epoch]

        # ---- 5b. Least-squares initial guess ----
        #
        # continuous: min_p ||S p + dx||^2
        # impulsive:  min_dv ||Phi_v dv + dx||^2
        #
        correction, _, _, _ = np.linalg.lstsq(control_matrix, -delta_x, rcond=None)

        # ---- 5c. Iterative Gauss-Newton correction ----
        it = 0

        while True:
            it += 1

            if thrust_mode == "continuous":
                sim = propagate_trajectory(
                    t0,
                    termination_settings,
                    bodies,
                    lambert_arc_ephemeris,
                    initial_state_correction=current_state_correction,
                    use_rsw_acceleration=True,
                    rsw_acceleration_magnitude=correction,
                )
            else:  # impulsive
                impulse_correction = np.concatenate([np.zeros(3), correction])
                sim = propagate_trajectory(
                    t0,
                    termination_settings,
                    bodies,
                    lambert_arc_ephemeris,
                    initial_state_correction=current_state_correction + impulse_correction,
                    use_rsw_acceleration=False,
                )

            sh = sim.propagation_results.state_history
            lh = get_lambert_arc_history(lambert_arc_ephemeris, sh)
            sh_arr = np.vstack(list(sh.values()))
            lh_arr = np.vstack(list(lh.values()))

            delta_x_new = sh_arr[-1, :] - lh_arr[-1, :]
            state_dev = np.linalg.norm(delta_x_new)

            if state_dev <= max_state_deviation_tolerance:
                break
            if it >= max_iterations_per_arc:
                break

            # Gauss-Newton update
            dp, _, _, _ = np.linalg.lstsq(control_matrix, -delta_x_new, rcond=None)
            correction = correction + dp

        # ---- 5d. Store results ----
        correction_history[arc_idx, :] = correction
        final_state_deviation_norm[arc_idx] = state_dev
        final_pos_deviation_norm[arc_idx] = np.linalg.norm(delta_x_new[0:3])
        final_vel_deviation_norm[arc_idx] = np.linalg.norm(delta_x_new[3:6])
        iterations_per_arc[arc_idx] = it

        current_state_correction = delta_x_new.copy()
        # ---- Progress report ----
        if arc_idx == 0 or (arc_idx + 1) % 5 == 0:
            t_elapsed   = time_module.time() - t_loop_start
            t_per_arc   = t_elapsed / (arc_idx + 1)
            t_remaining = t_per_arc * (number_of_arcs - arc_idx - 1)
            print(
                f"  [{arc_idx+1:4d}/{number_of_arcs}]  "
                f"||dr||={np.linalg.norm(delta_x_new[0:3]):.4e} m  "
                f"||dv||={np.linalg.norm(delta_x_new[3:6]):.4e} m/s  "
                f"||dx||={state_dev:.4e}  "
                f"iters={it}  "
                f"~{t_remaining/60:.1f} min left"
            )

    total_time = time_module.time() - t_loop_start

    # -----------------------------------------------------------------
    # 6. Summary
    # -----------------------------------------------------------------
    n_converged = int(np.sum(final_state_deviation_norm <= max_state_deviation_tolerance))
    correction_norms = np.linalg.norm(correction_history, axis=1)

    corr_label = "RSW accel" if thrust_mode == "continuous" else "delta-v"
    corr_unit  = "m/s^2" if thrust_mode == "continuous" else "m/s"

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Thrust mode            : {thrust_mode}")
    print(f"Computation time       : {total_time:.1f} s  ({total_time/60:.1f} min)")
    print(f"Converged arcs         : {n_converged}/{number_of_arcs}")
    print(f"Max  ||dx||            : {np.max(final_state_deviation_norm):.4e}")
    print(f"Mean ||dx||            : {np.mean(final_state_deviation_norm):.4e}")
    print(f"Max  ||dr||            : {np.max(final_pos_deviation_norm):.4e} m")
    print(f"Max  ||dv_dev||        : {np.max(final_vel_deviation_norm):.4e} m/s")
    print(f"Mean iterations/arc    : {np.mean(iterations_per_arc):.2f}")
    print(f"Max  iterations/arc    : {np.max(iterations_per_arc)}")
    print(f"Max  |{corr_label}|       : {np.max(correction_norms):.4e} {corr_unit}")
    print(f"Mean |{corr_label}|       : {np.mean(correction_norms):.4e} {corr_unit}")
    if thrust_mode == "impulsive":
        print(f"Total delta-v          : {np.sum(correction_norms):.4e} m/s")

    # -----------------------------------------------------------------
    # 7. Plots
    # -----------------------------------------------------------------
    time_days = (arc_midpoint_epochs - soi_dep_epoch) / constants.JULIAN_DAY

    if thrust_mode == "continuous":
        comp_labels = ["R  (radial)", "S  (along-track)", "W  (cross-track)"]
        y_unit = "m/s$^2$"
        suptitle = f"Piecewise-constant RSW thrust profile  ({number_of_arcs} arcs)"
    else:
        comp_labels = ["$\\Delta v_x$", "$\\Delta v_y$", "$\\Delta v_z$"]
        y_unit = "m/s"
        suptitle = f"Impulsive $\\Delta v$ profile  ({number_of_arcs} arcs)"

    # ---- 7a. Correction components vs time ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for k, ax in enumerate(axes):
        ax.step(time_days, correction_history[:, k],
                where="mid", linewidth=0.6)
        ax.set_ylabel(f"{comp_labels[k]}  [{y_unit}]", fontsize=12)
        ax.set_title(f"{comp_labels[k]} component", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)

    axes[-1].set_xlabel("Time since SOI departure  [days]", fontsize=12)
    fig.suptitle(suptitle, fontsize=15, y=1.01)
    plt.tight_layout()
    plt.show()

    # ---- 7b. Correction magnitude vs time ----
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(time_days, correction_norms, where="mid",
            linewidth=0.6, color="tab:red")
    ax.set_xlabel("Time since SOI departure  [days]", fontsize=12)
    ax.set_ylabel(f"|correction|  [{y_unit}]", fontsize=12)
    ax.set_title(f"{'Thrust acceleration' if thrust_mode == 'continuous' else 'Delta-v'} magnitude",
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.show()

    # ---- 7c. State / position / velocity deviations at arc endpoints ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].semilogy(time_days, final_state_deviation_norm,
                     linewidth=0.5, color="tab:blue")
    axes[0].axhline(max_state_deviation_tolerance,
                    color="red", linestyle="--", label="Tolerance")
    axes[0].set_ylabel("$\\|\\delta\\mathbf{x}\\|$", fontsize=12)
    axes[0].set_title("6-D state deviation norm (mixed units)", fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(time_days, final_pos_deviation_norm,
                     linewidth=0.5, color="tab:green")
    axes[1].set_ylabel("$\\|\\delta\\mathbf{r}\\|$  [m]", fontsize=12)
    axes[1].set_title("Position deviation norm", fontsize=13)
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(time_days, final_vel_deviation_norm,
                     linewidth=0.5, color="tab:orange")
    axes[2].set_ylabel("$\\|\\delta\\mathbf{v}\\|$  [m/s]", fontsize=12)
    axes[2].set_title("Velocity deviation norm", fontsize=13)
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.tick_params(labelsize=11)
    axes[-1].set_xlabel("Time since SOI departure  [days]", fontsize=12)
    plt.tight_layout()
    plt.show()

    # ---- 7d. Iterations per arc ----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(number_of_arcs), iterations_per_arc,
           width=1.0, color="tab:purple", alpha=0.6)
    ax.set_xlabel("Arc index", fontsize=12)
    ax.set_ylabel("Iterations", fontsize=12)
    ax.set_title("Gauss-Newton iterations per arc", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.show()