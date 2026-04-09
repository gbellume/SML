""" 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

from platform import system
import csv


from interplanetary_transfer_helper_functions import *


# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./Assignment2/SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 3 #################################################
###########################################################################

if __name__ == "__main__":

    output_directory_file = "./Assignment2/SimulationOutput/CartesianResults_AE4868_2026_2_6550150.dat"

    # Create body objects
    bodies = create_simulation_bodies()

    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )



    departure_epoch, _, arrival_epoch, _, time_of_flight = find_propagation_time_soi(lambert_arc_ephemeris, bodies, departure_epoch, "Earth", target_body, arrival_epoch)
    ##############################################################
    # Compute number of arcs and arc length
    number_of_arcs = 10
    acceptable_delta = 1


    arc_length = (arrival_epoch - departure_epoch) / number_of_arcs
    current_arc_initial_epoch = departure_epoch- arc_length
    current_arc_final_epoch = departure_epoch
    # random colors for plotting
    colors = [plt.cm.tab20(i) for i in range(number_of_arcs)]
    ##############################################################
    fig_uncorrected, ax_uncorrected = plt.subplots(figsize=(10, 10))
    fig_corrected, ax_corrected = plt.subplots(figsize=(10, 10))
    ax_corrected.set_yscale("log")
    ax_corrected.set_title("Position deviation - with impulsive corrections", fontsize=20)
    ax_corrected.set_xlabel("Time [s]", fontsize=16)
    ax_corrected.tick_params(axis='both', labelsize=14)
    ax_corrected.set_ylabel("Position deviation [m]", fontsize=16)
    ax_corrected.grid(True)

    fig_corrected_single, axes_corrected_single = plt.subplots(5, 2, figsize=(6, 20))
    plt.subplots_adjust(hspace=0.5, wspace=0.3, bottom = 0.01)
    axes_corrected_single = axes_corrected_single.flatten()

    fig_converged_single, axes_converged_single = plt.subplots(5, 2, figsize=(6, 20))
    plt.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.01)
    axes_converged_single = axes_converged_single.flatten()


    fig_converged, ax_converged = plt.subplots(figsize=(10, 10))
    ax_converged.set_yscale("log")
    ax_converged.set_title("Position deviation - converged corrections", fontsize=20)
    ax_converged.set_xlabel("Time [s]", fontsize=16)
    ax_converged.set_ylabel("Position deviation [m]", fontsize=16)
    ax_converged.tick_params(axis='both', labelsize=14)
    ax_converged.grid(True)



    total_delta_v = 0
    # Compute relevant parameters (dynamics, state transition matrix, Delta V) for each arc
    dictionary_to_store_results = {}
    for arc_index in range(number_of_arcs):
        color = colors[arc_index]
        # Compute initial and final time for arc
        current_arc_initial_epoch += arc_length 
        current_arc_final_epoch += arc_length

        ###########################################################################
        # RUN CODE FOR QUESTION 3a ################################################
        ###########################################################################

        # Propagate dynamics on current arc (use propagate_trajectory function)
        termination_settings = propagation_setup.propagator.time_termination(current_arc_final_epoch)
        dynamics_simulator_no_corrects = propagate_trajectory(
            current_arc_initial_epoch,
            termination_settings,
            bodies,
            lambert_arc_ephemeris,
            use_perturbations=True,
            )

        sh_no_corrects = dynamics_simulator_no_corrects.propagation_results.state_history
        lambert_comparison_history_no_corrects = get_lambert_arc_history(lambert_arc_ephemeris, sh_no_corrects)
        lambert_comparison_history_array_no_corrects = np.vstack(list(lambert_comparison_history_no_corrects.values()))
        delta_pos_norm_no_corrects = np.linalg.norm(np.vstack(list(sh_no_corrects.values()))[:, 0:3] - lambert_comparison_history_array_no_corrects[:, 0:3], axis=1, keepdims = True)
        ax_uncorrected.plot(sh_no_corrects.keys(), delta_pos_norm_no_corrects, label=f"Arc {arc_index+1}", color=color)

        ###########################################################################
        # RUN CODE FOR QUESTION 3c/d/e ###########################################
        ###########################################################################
        # Note: for question 3e, part of the code below will be put into a loop
        # for the requested iterations

        # Solve for state transition matrix on current arc
        termination_settings = propagation_setup.propagator.time_termination(
            current_arc_final_epoch
        )
        termination_settings_list = [termination_settings, propagation_setup.propagator.time_termination(arrival_epoch, True)]
        termination_settings = propagation_setup.propagator.hybrid_termination(
            termination_settings_list, True
        )
        variational_equations_solver = propagate_variational_equations(
            current_arc_initial_epoch,
            termination_settings,
            bodies,
            lambert_arc_ephemeris,
        )
        state_transition_matrix_history = (
            variational_equations_solver.state_transition_matrix_history
        )
        state_history = variational_equations_solver.state_history
        lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

        # Get final state transition matrix (and its inverse)
        final_epoch = list(state_transition_matrix_history.keys())[-1]
        final_state_transition_matrix = state_transition_matrix_history[final_epoch]

        # Retrieve final state deviation
        final_state_deviation = (
            state_history[final_epoch] - lambert_history[final_epoch]
        )

        # I am only interested in the influence the velocity has, for the single 
        # impulsive thrusts at the arcs' intersections, so I only look at the position 
        # part of the final state deviation and only at the velocity contribution of the
        # system matrix. This is also because the initial positional deviation is null.
        stm_pos_vel = final_state_transition_matrix[0:3, 3:6]  #(3, 3)

        # We want final position deviation = 0, so either I put the negative of the spatial
        # deviation or the negative of the thrust that will be needed 
        rhs = -final_state_deviation[0:3]

        # Solving
        delta_v_correction = np.linalg.solve(stm_pos_vel, rhs)
        dvx, dvy, dvz = delta_v_correction

        # total_delta_v += delta_v_correction

        # # Compute required velocity change at beginning of arc to meet required final state
        # Build the state correction vector: zero position offset, velocity = delta-v
        initial_state_correction = np.array([0, 0, 0, dvx, dvy, dvz])
        cumulative_correction = np.array([0, 0, 0, dvx, dvy, dvz])

        # Propagate with the impulsive correction applied at the arc's start
        termination_settings = propagation_setup.propagator.time_termination(
            current_arc_final_epoch
        )
        termination_settings_list = [termination_settings, propagation_setup.propagator.time_termination(arrival_epoch, True)]
        termination_settings = propagation_setup.propagator.hybrid_termination(
            termination_settings_list, True
        )
        dynamics_simulator = propagate_trajectory(
            current_arc_initial_epoch,
            termination_settings,
            bodies,
            lambert_arc_ephemeris,
            use_perturbations=True,
            initial_state_correction=initial_state_correction,
        )
        sh_new_last = dynamics_simulator.propagation_results.state_history
        lambert_hist = get_lambert_arc_history(lambert_arc_ephemeris, sh_new_last)
        lambert_array = np.vstack(list(lambert_hist.values()))
        delta_pos_corrected = np.linalg.norm(
            np.vstack(list(sh_new_last.values()))[:, 0:3] - lambert_array[:, 0:3],
            axis=1, keepdims=True
        )
        ax_corrected.plot(sh_new_last.keys(), delta_pos_corrected, label=f"Arc {arc_index+1}", color=color)
        ax_corr = axes_corrected_single[arc_index]
        ax_corr.plot(sh_new_last.keys(), delta_pos_corrected.ravel(), color=color, label=f"Arc {arc_index+1}")
        ax_corr.set_yscale("log")

        # only major grid lines
        ax_corr.grid(True, which="major")
        ax_corr.minorticks_off()

        # readable ticks
        ax_corr.tick_params(axis="both", labelsize=11, pad=3)

        # no individual titles or axis labels
        ax_corr.legend(loc="lower center", fontsize=11, frameon=True)
        # print(f"Arc {arc_index+1}: Delta V correction applied in norm: {np.linalg.norm(delta_v_correction)} m/s \n Final deviation: {sh_new_last[final_epoch] - lambert_hist[final_epoch]}")
        dictionary_to_store_results[arc_index] = {}
        dictionary_to_store_results[arc_index][0] = {
            "delta_v_correction": delta_v_correction,
            "final_deviation_before": np.linalg.norm(final_state_deviation[0:3]),
            "final_deviation_after": float(delta_pos_corrected[-1, 0]),
        }
        iter = 0
        final_state_deviation = np.vstack(list(sh_new_last.values()))[-1, 0:3] - lambert_array[-1, 0:3]
        while dictionary_to_store_results[arc_index][iter]["final_deviation_after"] > acceptable_delta:
            iter += 1

            rhs = -final_state_deviation
            delta_v_correction = np.linalg.solve(stm_pos_vel, rhs)
            dvx, dvy, dvz = delta_v_correction

            # total_delta_v += delta_v_correction

            # # Compute required velocity change at beginning of arc to meet required final state
            # Build the state correction vector: zero position offset, velocity = delta-v
            cumulative_correction += np.array([0, 0, 0, dvx, dvy, dvz])

            # Propagate with the impulsive correction applied at the arc's start
            dynamics_simulator = propagate_trajectory(
                current_arc_initial_epoch,
                termination_settings,
                bodies,
                lambert_arc_ephemeris,
                use_perturbations=True,
                initial_state_correction=cumulative_correction,
            )
            sh_new_last = dynamics_simulator.propagation_results.state_history
            lambert_hist = get_lambert_arc_history(lambert_arc_ephemeris, sh_new_last)
            lambert_array = np.vstack(list(lambert_hist.values()))
            delta_pos_corrected = np.linalg.norm(
                np.vstack(list(sh_new_last.values()))[:, 0:3] - lambert_array[:, 0:3],
                axis=1, keepdims=True
            )
            # ax_corrected.plot(sh_new_last.keys(), delta_pos_corrected, label=f"Arc {arc_index+1}", color=color)
            # print(f"Arc {arc_index+1}: Delta V correction applied in norm: {np.linalg.norm(delta_v_correction)} m/s \n Final deviation: {sh_new_last[final_epoch] - lambert_hist[final_epoch]}")
            dictionary_to_store_results[arc_index][iter] = {
                "delta_v_correction": delta_v_correction,
                "final_deviation_before": np.linalg.norm(final_state_deviation[0:3]),
                "final_deviation_after": float(delta_pos_corrected[-1, 0]),
            }
            final_state_deviation = (sh_new_last[final_epoch] - lambert_hist[final_epoch])[0:3]

        ax_converged.plot(sh_new_last.keys(), delta_pos_corrected, label=f"Arc {arc_index+1}", color=color)
        total_delta_v += np.linalg.norm(cumulative_correction[3:6])
        if arc_index == 0 or arc_index == 4:
            sh_no_corrects_array = np.vstack(list(sh_no_corrects.values()))
            time_keys = list(sh_no_corrects.keys())
            with open("./Assignment2/SimulationOutput/CartesianResults_AE4868_2026_2_6550150.dat", "a") as f:
                f.write(f"{time_keys[0]} ")
                for elem in sh_no_corrects_array[0]:
                    f.write(f"{elem} ")
                f.write("\n")
                f.write(f"{time_keys[-1]} ")
                for elem in sh_no_corrects_array[-1]:
                    f.write(f"{elem} ")
                f.write("\n")
        ax_conv = axes_converged_single[arc_index]
        ax_conv.plot(sh_new_last.keys(), delta_pos_corrected.ravel(), color=color, label=f"Arc {arc_index+1}")
        ax_conv.set_yscale("log")
        ax_conv.grid(True, which="major")
        ax_conv.minorticks_off()
        ax_conv.tick_params(axis="both", labelsize=11, pad=3)
        ax_conv.legend(loc="lower center", fontsize=11, frameon=True)






    ax_uncorrected.set_title("Position deviation - no corrections", fontsize=20)
    ax_uncorrected.set_xlabel("Time [s]", fontsize=16)
    ax_uncorrected.set_yscale("log")
    ax_uncorrected.tick_params(axis='both', labelsize=14)
    ax_uncorrected.set_ylabel("Position deviation [m]", fontsize=16)
    ax_uncorrected.legend()
    ax_uncorrected.grid(True)
    plt.show()

    ax_corrected.set_title("Position deviation - with impulsive corrections", fontsize=20)
    ax_corrected.set_xlabel("Time [s]", fontsize=16)
    ax_corrected.set_ylabel("Position deviation [m]", fontsize=16)
    ax_corrected.legend()
    ax_corrected.grid(True)
    plt.show()

    ax_converged.legend()
    plt.show()

    fig_converged_single.suptitle("Position deviation - converged corrections", fontsize=18, y=0.98)
    fig_converged_single.supxlabel("Time [s]", fontsize=14, y=0.04)
    fig_converged_single.supylabel("Position deviation [m]", fontsize=14, x=0.04)
    fig_converged_single.subplots_adjust(
        left=0.10, right=0.98, top=0.99, bottom=0.02, hspace=0.10, wspace=0.10
    )
    plt.show()

    fig_corrected_single.suptitle("Position deviation - with impulsive corrections", fontsize=18, y=0.98)
    fig_corrected_single.supxlabel("Time [s]", fontsize=14, y=0.04)
    fig_corrected_single.supylabel("Position deviation [m]", fontsize=14, x=0.04)

    fig_corrected_single.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.99,
        bottom=0.02,
        hspace=0.10,
        wspace=0.10
    )

    plt.show()
    plt.show()

    print(f"Total Delta V required for correction: {np.linalg.norm(total_delta_v)} m/s")



    with open(output_directory + "correction_results_updated_correct_stepsize.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arc_index", "iteration", "delta_vx", "delta_vy", "delta_vz", "delta_v_norm", "final_deviation_before", "final_deviation_after"])
        
        for arc_index, iterations in dictionary_to_store_results.items():
            for iter_index, data in iterations.items():
                dv = data["delta_v_correction"]
                writer.writerow([
                    arc_index + 1,
                    iter_index,
                    dv[0], dv[1], dv[2],
                    np.linalg.norm(dv),
                    data["final_deviation_before"],
                    data["final_deviation_after"],
                ])
                

        

    # The identity term (1/r³) represents how acceleration spreads uniformly in all directions
    # The outer product term (3rrᵀ/r⁵) subtracts extra sensitivity along the radial direction, gravity gets weaker faster when you move toward/away from the source than when you move sideways
