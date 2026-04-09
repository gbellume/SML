""" 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

from interplanetary_transfer_helper_functions import *
import time

# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 4 #################################################
###########################################################################

if __name__ == "__main__":

    rsw_acceleration_magnitude = [0, 0, 0]

    # Create body objects
    bodies = create_simulation_bodies()

    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )

    ###########################################################################
    # RUN CODE FOR QUESTION 4b ################################################
    ###########################################################################

    # Set start and end times of full trajectory
    departure_epoch_with_buffer, _, arrival_epoch_with_buffer, _, _ = find_propagation_time_soi(
        lambert_arc_ephemeris, bodies, departure_epoch, "Earth", target_body, arrival_epoch
    )
    # Solve for state transition matrix on current arc
    termination_settings = propagation_setup.propagator.time_termination(
        arrival_epoch_with_buffer
    )
    termination_settings_list = [termination_settings, propagation_setup.propagator.time_termination(arrival_epoch, True)]
    termination_settings = propagation_setup.propagator.hybrid_termination(
        termination_settings_list, True
    )
    variational_equations_solver = propagate_variational_equations(
        departure_epoch_with_buffer,
        termination_settings,
        bodies,
        lambert_arc_ephemeris,
        use_rsw_acceleration = True)

    sensitivity_matrix_history = variational_equations_solver.sensitivity_matrix_history
    state_history = variational_equations_solver.state_history
    lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

    final_epoch = list(sensitivity_matrix_history.keys())[-1]

    S_final = sensitivity_matrix_history[final_epoch]          
    S_r = S_final[0:3, :]                                      

    # pos deviation from Lambert without thrust
    final_state_deviation = state_history[final_epoch] - lambert_history[final_epoch]
    delta_r_0 = -final_state_deviation[0:3]                     


    rsw_acceleration_magnitude = np.linalg.solve(S_r, delta_r_0)

    # Propagate dynamics with RSW acceleration. NOTE: use the rsw_acceleration_magnitude as
    # input to the propagate_trajectory function
    termination_settings = propagation_setup.propagator.time_termination(
        arrival_epoch_with_buffer
    )
    termination_settings_list = [termination_settings, propagation_setup.propagator.time_termination(arrival_epoch, True)]
    termination_settings = propagation_setup.propagator.hybrid_termination(
        termination_settings_list, True
    )
    dynamics_simulator = propagate_trajectory(
        departure_epoch_with_buffer,
        termination_settings,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=True,
        use_rsw_acceleration=True,
        rsw_acceleration_magnitude=rsw_acceleration_magnitude,
    )

    dep_var_vec_4b = np.vstack(list(dynamics_simulator.propagation_results.dependent_variable_history.values()))
    sh_thrust = dynamics_simulator.propagation_results.state_history
    sh_thrust_array = np.vstack(list(sh_thrust.values()))
    aero_acc_rsw = np.zeros((len(sh_thrust), 3))
    rsw_states = np.zeros((len(sh_thrust), 3, 3))
    for i in range(len(sh_thrust)):
        rsw_states[i] = astro.frame_conversion.inertial_to_rsw_rotation_matrix(sh_thrust_array[i, 0:6])
        aero_acc_rsw[i] = rsw_states[i] @ dep_var_vec_4b[i, 0:3]


    lambert_hist_thrust = get_lambert_arc_history(lambert_arc_ephemeris, sh_thrust)

    lambert_array = np.vstack(list(lambert_hist_thrust.values()))
    sh_thrust_array = np.vstack(list(sh_thrust.values()))
    sh_no_thrust_array = np.vstack(list(state_history.values()))

    delta_pos_thrust = np.linalg.norm(
        sh_thrust_array[:, 0:3] - lambert_array[:, 0:3], axis=1, keepdims=True
    )
    delta_pos_no_thrust = np.linalg.norm(
        sh_no_thrust_array[:, 0:3] - lambert_array[:, 0:3], axis=1, keepdims=True
    )

    print(f"Final position deviation without thrust: {float(delta_pos_no_thrust[-1, 0]):.4e} m")
    print(f"Final position deviation with RSW thrust: {float(delta_pos_thrust[-1, 0]):.4e} m")
    print(f"RSW acceleration magnitude [$m/s^2$]: {np.linalg.norm(rsw_acceleration_magnitude):.4e}")

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=15)
    ax.plot(state_history.keys(), delta_pos_no_thrust, label="No thrust")
    ax.plot(sh_thrust.keys(), delta_pos_thrust, label="With RSW correction")
    ax.set_title("Position deviation w.r.t. Lambert arc with low-thrust correction", fontsize=20)
    ax.set_xlabel("Time [s]", fontsize=16)
    ax.set_ylabel("Position deviation [m]", fontsize=16)
    ax.legend()
    plt.show()
    sh_thrust_keys = list(sh_thrust.keys())
    with open("./Assignment2/SimulationOutput/CartesianResults_AE4868_2026_2_6550150.dat", "a") as f:
        f.write(f"{sh_thrust_keys[-1]} ")
        for elem in sh_thrust_array[-1]:
            f.write(f"{elem} ")
        f.write("\n")

    ##################################################################################
    # part c and d
    ##################################################################################


    departure_epoch_with_buffer, _, arrival_epoch_with_buffer, _, _ = find_propagation_time_soi(
        lambert_arc_ephemeris, bodies, departure_epoch, "Earth", target_body, arrival_epoch
    )

    tmid = np.median(np.array(list(state_history.keys())))

    # Arc 1: I propagate from departure to tmid with the previously computed RSW acceleration
    p_arc1 = rsw_acceleration_magnitude
    termination_settings = propagation_setup.propagator.time_termination(
        tmid
    )
    termination_settings_list = [termination_settings, propagation_setup.propagator.time_termination(arrival_epoch, True)]
    termination_settings = propagation_setup.propagator.hybrid_termination(
        termination_settings_list, True
    )
    dynamics_simulator = propagate_trajectory(
        departure_epoch_with_buffer,
        termination_settings,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=True,
        use_rsw_acceleration=True,
        rsw_acceleration_magnitude=p_arc1,
    )

    sh_arc1 = dynamics_simulator.propagation_results.state_history
    lambert_hist_arc1 = get_lambert_arc_history(lambert_arc_ephemeris, sh_arc1)

    lambert_array_arc1 = np.vstack(list(lambert_hist_arc1.values()))
    sh_arc1_array = np.vstack(list(sh_arc1.values()))
    sh_no_thrust_array = np.vstack(list(state_history.values()))

    delta_x_tmid = (sh_arc1_array - lambert_array_arc1)[-1, :]

    # arc 2: I propagate from tmid to arrival epoch. I consider initial state correction at tmid,
    # and I use the newly computed RSW acceleration
    
    t_s_arc2 = propagation_setup.propagator.time_termination(
        arrival_epoch_with_buffer
    )
    t_s_list_arc2 = [t_s_arc2, propagation_setup.propagator.time_termination(arrival_epoch, True)]
    termination_settings = propagation_setup.propagator.hybrid_termination(
        t_s_list_arc2, True
    )
    variational_equations_solver = propagate_variational_equations(
        tmid,
        termination_settings,
        bodies,
        lambert_arc_ephemeris,
        use_rsw_acceleration = True
        )
    sh_arc2 = variational_equations_solver.state_history
    lambert_hist_arc2 = get_lambert_arc_history(lambert_arc_ephemeris, sh_arc2)

    lambert_array_arc2 = np.vstack(list(lambert_hist_arc2.values()))
    sh_arc2_array = np.vstack(list(sh_arc2.values()))


    stm_history_arc2         = variational_equations_solver.state_transition_matrix_history
    sensitivity_history_arc2 = variational_equations_solver.sensitivity_matrix_history

    final_epoch_arc2 = list(stm_history_arc2.keys())[-1]
    Phi2_final = stm_history_arc2[final_epoch_arc2]         # 6x6
    S2_final   = sensitivity_history_arc2[final_epoch_arc2] # 6x3

    Phi2_r = Phi2_final[0:3, :]  # 3x6
    S2_r   = S2_final[0:3, :]    # 3x3

    p_arc2 = np.linalg.solve(S2_r, -Phi2_r @ delta_x_tmid)

    termination_settings = propagation_setup.propagator.time_termination(
        arrival_epoch_with_buffer
    )
    termination_settings_list = [termination_settings, propagation_setup.propagator.time_termination(arrival_epoch, True)]
    termination_settings = propagation_setup.propagator.hybrid_termination(
        termination_settings_list, True
    )
    dynamics_simulator = propagate_trajectory(
        tmid,
        termination_settings,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=True,
        initial_state_correction=delta_x_tmid,
        use_rsw_acceleration=True,
        rsw_acceleration_magnitude=p_arc2,
    )

    sh_arc2 = dynamics_simulator.propagation_results.state_history
    lambert_hist_arc2 = get_lambert_arc_history(lambert_arc_ephemeris, sh_arc2)

    lambert_array_arc2 = np.vstack(list(lambert_hist_arc2.values()))
    sh_arc2_array = np.vstack(list(sh_arc2.values()))

    delta_x_tfinal = sh_arc2_array - lambert_array_arc2
    delta_r_final = delta_x_tfinal[-1, 0:3]

    p_arc_2, delta_x_tfinal, iter_arc_2, sh_arc2_array, sh_arc2 = iterative_correction_low_thrust(bodies, lambert_arc_ephemeris, S2_r, delta_x_tfinal, termination_settings,
    tmid, delta_x_tmid, p_arc2)


    print(f"RSW acceleration magnitude for arc 1: {np.linalg.norm(p_arc1):.4e} m/s^2")
    print(f"RSW acceleration magnitude for arc 2: {np.linalg.norm(p_arc2):.4e} m/s^2")
    print(f"Final position deviation with RSW correction on both arcs: {float(np.linalg.norm(delta_x_tfinal[-1, 0:3])):.4e} m")
    print(f"Number of iterations for arc 2: {iter_arc_2}")
    print(f"Difference in directional thrust between arc1 and arc2:\n")
    print(f"Along R: parc1x = {p_arc1[0]/p_arc2[0]}parc2r")
    print(f"Along S: parc1y = {p_arc1[1]/p_arc2[1]}parc2s")
    print(f"Along W: parc1z = {p_arc1[2]/p_arc2[2]}parc2w")

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', labelsize=15)
    plt.plot(sh_arc1.keys(), np.linalg.norm(sh_arc1_array[:, 0:3] - lambert_array_arc1[:, 0:3], axis=1), label="Arc 1")
    plt.plot(sh_arc2.keys(), np.linalg.norm(sh_arc2_array[:, 0:3] - lambert_array_arc2[:, 0:3], axis=1), label="Arc 2")
    plt.title("Position deviation w.r.t. Lambert arc with low-thrust correction on two arcs", fontsize=20)
    plt.xlabel("Time [s]", fontsize=16)
    plt.ylabel("Position deviation [m]", fontsize=16)

    plt.show()
    sh_thrust_keys = list(sh_thrust.keys())
    with open("./Assignment2/SimulationOutput/CartesianResults_AE4868_2026_2_6550150.dat", "a") as f:
        f.write(f"{sh_thrust_keys[-1]} ")
        for elem in sh_thrust_array[-1]:
            f.write(f"{elem} ")
        f.write("\n")

    ##################################################################################
    # MONTECARLO!!
    n_samples = 1000
    p_b = rsw_acceleration_magnitude
    std = 0.4*np.linalg.norm(p_b)

    rng = np.random.default_rng(seed=42)
    p1_samples = rng.normal(p_b, std, (n_samples, 3))
    p2_samples = np.zeros((n_samples, 3))
    dr_final = np.zeros((n_samples, 3))
    termination_settings_arc1_mc = propagation_setup.propagator.time_termination(tmid)
    termination_settings_list = [termination_settings_arc1_mc, propagation_setup.propagator.time_termination(arrival_epoch, True)]
    termination_settings_arc1_mc = propagation_setup.propagator.hybrid_termination(termination_settings_list, True)


    t_loop_start = time.time()
    for i, p1 in enumerate(p1_samples):
        t_start_iter = time.time()

        sim_arc1 = propagate_trajectory(
            departure_epoch_with_buffer, termination_settings_arc1_mc, bodies, lambert_arc_ephemeris,
            use_perturbations=True, use_rsw_acceleration=True, rsw_acceleration_magnitude=p1,
        )
        sh_a1         = sim_arc1.propagation_results.state_history
        lh_a1         = get_lambert_arc_history(lambert_arc_ephemeris, sh_a1)
        dx_tmid_i     = (np.vstack(list(sh_a1.values())) - np.vstack(list(lh_a1.values())))[-1, :]
        p2_temp= np.linalg.solve(S2_r, -Phi2_r @ dx_tmid_i)
        dynamics_simulator = propagate_trajectory(
            tmid,
            termination_settings,
            bodies,
            lambert_arc_ephemeris,
            use_perturbations=True,
            initial_state_correction=dx_tmid_i,
            use_rsw_acceleration=True,
            rsw_acceleration_magnitude=p2_temp,
        )

        sh_a2         = dynamics_simulator.propagation_results.state_history
        lh_a2         = get_lambert_arc_history(lambert_arc_ephemeris, sh_a2)
        delta_x_tfinal_i = np.vstack(list(sh_a2.values())) - np.vstack(list(lh_a2.values()))
        p2_samples[i] , dx_final_i , _ , sh_a2_array , sh_a2 = iterative_correction_low_thrust(bodies, lambert_arc_ephemeris, S2_r, delta_x_tfinal_i, termination_settings,
        tmid, dx_tmid_i, p2_temp,)
        dr_final[i, :] = dx_final_i[-1, 0:3]

        if i == 0:
            t_per_iter = time.time() - t_start_iter

        if (i + 1) % 10 == 0:
            t_per_iter  = (time.time() - t_loop_start) / (i + 1)
            t_remaining = t_per_iter * (n_samples - i - 1)
            print(f"  [{i+1}/{n_samples}]  ~{t_remaining/60:.1f} min remaining  ({t_per_iter:.2f} s/iter)")
            
    avg_thrust   = (np.linalg.norm(p1_samples, axis=1) + np.linalg.norm(p2_samples, axis=1)) / 2
    p1_deviation = np.linalg.norm(p1_samples - p_b, axis=1)
    idx_opt      = np.argmin(avg_thrust)
    p1_opt       = p1_samples[idx_opt]
    p2_opt       = p2_samples[idx_opt]


    ##################################################################################
    # d.i — average thrust vs deviation from baseline first-arc thrust
    ##################################################################################

    avg_thrust_c = 0.5 * (np.linalg.norm(p_arc1) + np.linalg.norm(p_arc_2))

    plt.figure(figsize=(10, 6))
    plt.scatter(p1_deviation, avg_thrust, s=20, alpha=0.7, label="Monte Carlo samples")
    plt.scatter(
        p1_deviation[idx_opt],
        avg_thrust[idx_opt],
        s=100,
        marker="*",
        label="Monte Carlo optimal",
        zorder=3,
    )
    plt.axhline(avg_thrust_c, linestyle="--", label="Part (c) baseline")

    plt.xlabel(r"$\|p_1 - p^{(b)}\|$ [m/s$^2$]")
    plt.ylabel(r"Average thrust $(\|p_1\|+\|p_2\|)/2$ [m/s$^2$]")
    plt.title("Monte Carlo trade-off between thrust level and first-arc deviation")
    plt.legend()
    plt.grid(True)
    plt.show()


    ##################################################################################
    # d.ii — RSW thrust components vs time (part c vs Monte Carlo optimal)
    ##################################################################################

    time_nodes = np.array([
        departure_epoch_with_buffer,
        tmid,
        arrival_epoch
    ])

    component_labels = ["R", "S", "W"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for k, ax in enumerate(axes):
        thrust_c = [p_arc1[k], p_arc1[k], p_arc1[k]]
        thrust_opt = [p1_opt[k], p2_opt[k], p2_opt[k]]

        ax.step(time_nodes, thrust_c, where="post", label="Part b")
        ax.step(time_nodes, thrust_opt, where="post", label="Monte Carlo optimal")

        ax.set_ylabel(f"{component_labels[k]} [m/s$^2$]")
        ax.set_title(f"{component_labels[k]}-component thrust")
        ax.grid(True)

    axes[-1].set_xlabel("Time [s]")
    axes[0].legend()
    plt.tight_layout()
    plt.show()


    sim_arc1 = propagate_trajectory(
            departure_epoch_with_buffer, termination_settings_arc1_mc, bodies, lambert_arc_ephemeris,
            use_perturbations=True, use_rsw_acceleration=True, rsw_acceleration_magnitude=p1_opt,
    )
    dep_var_vec_4d_arc1 = np.vstack(list(sim_arc1.propagation_results.dependent_variable_history.values()))
    sim_arc2 = propagate_trajectory(
            tmid, termination_settings, bodies, lambert_arc_ephemeris,
            use_perturbations=True, use_rsw_acceleration=True, rsw_acceleration_magnitude=p2_opt,
    )
    dep_var_vec_4d_arc2 = np.vstack(list(sim_arc2.propagation_results.dependent_variable_history.values()))
    dep_var_vec_4d = np.vstack((dep_var_vec_4d_arc1, dep_var_vec_4d_arc2))

    times_c = list(sh_thrust.keys())
    times_arc1 = list(sim_arc1.propagation_results.state_history.keys())
    times_arc2 = list(sim_arc2.propagation_results.state_history.keys())
    times_opt = times_arc1 + times_arc2

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for k, ax in enumerate(axes):
        ax.plot(times_opt, p_arc1[:, k], label="Part b")
        ax.plot(times_opt, dep_var_vec_4d[:, k], label="Monte Carlo optimal")

        ax.set_ylabel(f"{component_labels[k]} [m/s$^2$]")
        ax.set_title(f"{component_labels[k]}-component acceleration profile")
        ax.grid(True)

    axes[-1].set_xlabel("Time [s]")
    axes[0].legend()
    plt.tight_layout()
    plt.show()
