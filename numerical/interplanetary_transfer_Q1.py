""" 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

from turtle import position

from interplanetary_transfer_helper_functions import *
import matplotlib.pyplot as plt


# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./Assignment2/SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 1 #################################################
###########################################################################

if __name__ == "__main__":
    output_directory_file = "./Assignment2/SimulationOutput/CartesianResults_AE4868_2026_2_6550150.dat"
    # Create body objects
    bodies = create_simulation_bodies()

    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )

    # Create propagation settings and propagate dynamics
    termination_settings = propagation_setup.propagator.time_termination(arrival_epoch)
    dynamics_simulator = propagate_trajectory(
        departure_epoch,
        termination_settings,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=False,
    )

    # Write results to file
    write_propagation_results_to_file(
        dynamics_simulator, lambert_arc_ephemeris, "Q1", output_directory
    )

    propagation_results = dynamics_simulator.propagation_results
    dep_var_vec = np.vstack(list(propagation_results.dependent_variable_history.values()))
    pos_earth = dep_var_vec[:, 0:3]
    pos_venus = dep_var_vec[:, 3:6]
    # Extract state history from dynamics simulator
    state_history = dynamics_simulator.propagation_results.state_history
    state_history_array = np.vstack(list(state_history.values()))
    
    # Evaluate the Lambert arc model at each of the epochs in the state_history
    lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, state_history)
    lambert_history_array = np.vstack(list(lambert_history.values()))

    plot_orbits_3d(
        (state_history_array, "Spacecraft", "green")
    )

    diff_lambert = lambert_history_array[:, 0:3] - state_history_array[:, 0:3]
    # three subplots for x, y, z position errors
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("Lambert Arc vs Numerical Propagation - Position Error", fontsize=16)
    axs[0].plot(diff_lambert[:, 0], label="x error", color="blue", linewidth=0.5)
    axs[0].set_ylabel("x error [m]", fontsize=16)
    axs[0].tick_params(axis='both', labelsize=15)
    axs[0].grid(True)
    axs[0].legend()
    axs[1].plot(diff_lambert[:, 1], label="y error", color="green", linewidth=0.5)
    axs[1].set_ylabel("y error [m]", fontsize=16)
    axs[1].tick_params(axis='both', labelsize=15)
    axs[1].legend()
    axs[1].grid(True)
    axs[2].plot(diff_lambert[:, 2], label="z error", color="red", linewidth=0.5)
    axs[2].set_ylabel("z error [m]", fontsize=16)
    axs[2].set_xlabel("Time step", fontsize=16)
    axs[2].tick_params(axis='both', labelsize=15)
    axs[2].grid(True)
    axs[2].legend()
    plt.tight_layout()
    plt.show()


    # Hypothesis of the lambert arc: departure from centers of masss, unfeasible. 
    # no perturbations, so the spacecraft follows the lambert arc exactly.
    # Point mass gravity from the sun only,
    # Central body at the centre of the non-moving Sun
    # start position, end position and time of flight are defined
    # single portion of keplerian orbit in prograde direction

    # Of these hypothesis, non of the simplifications are different from the ones of the propagation.
    # The only hypothesis that screams I AM DIFFERENT is that the Lambert Arc considers the Sun in its initial position. 
    # At first glance this is no issue, but we know the whole of the Solar System is moving around the Solar System Barycenter
    # so if the numerical propagation considers each body which exerts a gravitational pull to be in the position given by it ephemeris
    # at each timestep, then the Sun will be moving in the numerical propagation, but not in the Lambert Arc.


    # plot the Sun ephemeris in the same time range as the state history, to check if the Sun is moving or not
    sun_start = spice.get_body_cartesian_position_at_epoch("Sun", "SSB", "ECLIPJ2000", "None", departure_epoch)
    sun_end   = spice.get_body_cartesian_position_at_epoch("Sun", "SSB", "ECLIPJ2000", "None", arrival_epoch)
    delta     = sun_end - sun_start  # metres
    print("Sun position at departure epoch: ", sun_start)
    print("Sun position at arrival epoch: ", sun_end)
    print("Sun position change during transfer: ", delta)

    # Sun position at departure epoch:  [-2.79092244e+07 -8.07597819e+08  7.03793942e+06]
    # Sun position at arrival epoch:  [ 1.85472909e+08 -7.25871932e+08  1.19044854e+06]
    # Sun position change during transfer:  [ 2.13382133e+08  8.17258866e+07 -5.84749088e+06]
    time_array = np.array(list(propagation_results.dependent_variable_history.keys()))
    sun_history = np.array([spice.get_body_cartesian_position_at_epoch("Sun", "SSB", "ECLIPJ2000", "None", time) for time in time_array])
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("Positional variation of Sun wrt Lambert Arc approximation", fontsize=16)
    axs[0].plot(sun_history[:, 0], label="x error")
    axs[0].set_ylabel("x error [m]", fontsize=14)
    axs[0].tick_params(axis='both', labelsize=13)
    axs[0].grid(True)
    axs[0].legend()
    axs[1].plot(sun_history[:, 1], label="y error")
    axs[1].set_ylabel("y error [m]", fontsize=14)
    axs[1].tick_params(axis='both', labelsize=13)
    axs[1].grid(True)
    axs[1].legend()
    axs[2].plot(sun_history[:, 2], label="z error")
    axs[2].set_ylabel("z error [m]", fontsize=14)
    axs[2].set_xlabel("Time step", fontsize=14)
    axs[2].tick_params(axis='both', labelsize=13)
    axs[2].grid(True)
    axs[2].legend()
    plt.tight_layout()
    plt.show()

    with open(output_directory_file, "w") as f:
        f.write(f"{departure_epoch} ")
        for elem in state_history_array[0]:
            f.write(f"{elem} ")
        f.write("\n")
        f.write(f"{arrival_epoch} ")
        for elem in state_history_array[-1]:
            f.write(f"{elem} ")
        f.write("\n")