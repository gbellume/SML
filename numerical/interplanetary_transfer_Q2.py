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
from tudatpy import astro
# Load spice kernels.
spice.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./Assignment2/SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 2 #################################################
###########################################################################

if __name__ == "__main__":
    output_directory_file = "./Assignment2/SimulationOutput/CartesianResults_AE4868_2026_2_6550150.dat"

    # Create body objects
    bodies = create_simulation_bodies()
    departure_body = "Earth"
    arrival_body = "Venus"
    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(
        bodies, target_body, departure_epoch, arrival_epoch
    )
    # Compute once before the loops:
    # APPROXIMATION, SINCE ANALYTICAL AND PROPAGATION ARE VERY VERY SIMILAR, I WILL USE PROPAGATION
    termination_settings = propagation_setup.propagator.time_termination(arrival_epoch)
    dynamics_simulator = propagate_trajectory(
        departure_epoch,
        termination_settings,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=False,
    )
    lambert_history = get_lambert_arc_history(lambert_arc_ephemeris, dynamics_simulator.propagation_results.state_history)
    lambert_states = np.vstack(list(lambert_history.values()))
    time_array = np.array(list(lambert_history.keys()))


    # approximation semimajoraxis \sim constant
    earth_state = spice.get_body_cartesian_state_at_epoch(departure_body, "Sun", "ECLIPJ2000", "None", 892.0386735 * constants.JULIAN_DAY)
    mu_sun = spice.get_body_gravitational_parameter("Sun")
    a_earth = astro.element_conversion.cartesian_to_keplerian(earth_state, mu_sun)[0]
    venus_state = spice.get_body_cartesian_state_at_epoch(arrival_body, "Sun", "ECLIPJ2000", "None", 892.0386735 * constants.JULIAN_DAY)
    a_venus = astro.element_conversion.cartesian_to_keplerian(venus_state, mu_sun)[0]

    """
    case_i: The initial and final propagation time equal to the initial and final times of the Lambert arc.
    case_ii: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t=1 hour.
    case_iii: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t such that we start/end on the sphere of influence
    case_iv: The initial and final propagation time shifted forward and backward in time, respectively, by ∆t=1 hour. The propagation is started from the middle point in time of the Lambert arc and propagated forward and backward in time.

    """
    # List cases to iterate over. STUDENT NOTE: feel free to modify if you see fit
    cases = ["case_i", "case_ii", "case_iii", "case_iv"]
    # cases = ["case_iii", "case_iv"]

    # Run propagation for each of cases i-iii
    for case in cases:
        departure_epoch = 340.6905189 * constants.JULIAN_DAY
        time_of_flight = 165.3067532 * constants.JULIAN_DAY
        arrival_epoch = departure_epoch + time_of_flight
        if case == "case_ii":
            deltat = 3600
            departure_epoch += deltat
            time_of_flight -= 2*deltat
            arrival_epoch = departure_epoch + time_of_flight
        elif case == "case_iii":
            bool = True
            index_check = 0
            while bool:
                # time instant = time instant at lambert_arc ephemeris index index_check
                time_instant = time_array[index_check]
                pos_lamb = np.linalg.norm(lambert_states[index_check, 0:3])
                pos_earth_dep = np.linalg.norm(spice.get_body_cartesian_position_at_epoch(departure_body, "Sun", "ECLIPJ2000", "None", time_instant)[0:3])
                r_soi_earth =  a_earth * (bodies.get(departure_body).gravitational_parameter / bodies.get("Sun").gravitational_parameter)**(2/5)
                if np.abs(pos_lamb-pos_earth_dep) > r_soi_earth:
                    bool = False
                    departure_epoch = time_instant
                    departure_state = lambert_states[index_check]
                index_check += 1
                                


            bool = True
            index_check = len(time_array) - 1
            while bool:
                time_instant = time_array[index_check]
                pos_lamb = np.linalg.norm(lambert_states[index_check, 0:3])
                pos_venus_arr = np.linalg.norm(spice.get_body_cartesian_position_at_epoch(arrival_body, "Sun", "ECLIPJ2000", "None", time_instant)[0:3])
                r_soi_venus =  a_venus * (bodies.get(arrival_body).gravitational_parameter / bodies.get("Sun").gravitational_parameter)**(2/5)
                if np.abs(pos_lamb-pos_venus_arr) > r_soi_venus:
                    bool = False
                    arrival_epoch = time_instant
                    arrival_state = lambert_states[index_check]
                    time_of_flight = arrival_epoch-departure_epoch
                index_check -= 1


############################################################################
# ****** ASK THE PROFES IF I NEED TO USE HYBRID TERMINATION IN CASE 3 EVEN IF I AM DOING IT WITH BOOL
# ****** SINCE I AM USING BOOL, IF IT NEVER ENTERS THE SOI THE LAST INSTANT OF THE TIME ARRAY WHICH COMES FROM THE INITIAL 
# ****** PROPAGATION WILL BE THE ONE USED, WHICH IS OUTSIDE THE SOI
############################################################################

        elif case == "case_iv":
            departure_epoch = 340.6905189 * constants.JULIAN_DAY + 3600
            time_of_flight = 165.3067532 * constants.JULIAN_DAY - 7200
            arrival_epoch = departure_epoch + time_of_flight    
            propag_start_iv = np.median(time_array)

            

        # Perform propagation
        termination_settings = propagation_setup.propagator.time_termination(arrival_epoch)

        if case == "case_iv":
            termination_settings_forward = propagation_setup.propagator.time_termination(arrival_epoch, True)
            termination_settings_backward = propagation_setup.propagator.time_termination(departure_epoch, True)
            termination_settings = propagation_setup.propagator.non_sequential_termination(
                forward_termination_settings=termination_settings_forward,
                backward_termination_settings=termination_settings_backward,
            )
            departure_epoch = propag_start_iv


        dynamics_simulator = propagate_trajectory(
        departure_epoch,
        termination_settings,
        bodies,
        lambert_arc_ephemeris,
        use_perturbations=True,
        )

        sh = dynamics_simulator.propagation_results.state_history

        # for the different masses
        if case == "case_iv":
            sh_reference = dict(sh)

        time_arr = np.array(list(sh.keys()))
        lambert_comparison_history = get_lambert_arc_history(lambert_arc_ephemeris, sh)

        lambert_pos = np.vstack(list(lambert_comparison_history.values()))[:, 0:3]
        pos_sh = np.vstack(list(sh.values()))[:, 0:3]
        norm_pos_lambert = np.linalg.norm(lambert_pos, axis=1, keepdims=True)

        lambert_vel = np.vstack(list(lambert_comparison_history.values()))[:, 3:6]
        propag_vel = np.vstack(list(sh.values()))[:, 3:6]


        lambert_acc= -mu_sun * lambert_pos / norm_pos_lambert**3
        propag_acc = np.vstack(list(dynamics_simulator.propagation_results.dependent_variable_history.values()))[:, 0:3]


        plot_diff_pos = np.linalg.norm(pos_sh-lambert_pos, axis=1, keepdims=True)
        plot_diff_vel = np.linalg.norm(propag_vel-lambert_vel, axis=1, keepdims=True)
        plot_diff_acc = np.linalg.norm(propag_acc-lambert_acc, axis=1, keepdims=True)

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        fig.suptitle(f"Lambert Arc vs Numerical Propagation - {case} Error", fontsize=16)
        axs[0].semilogy(time_arr-time_arr[0], plot_diff_pos, label="position error")
        axs[0].set_ylabel(r"$\|\mathbf{x}(t) - \mathbf{\bar{x}}(t)\| [m]$", fontsize=14)
        axs[0].tick_params(axis='both', labelsize=13)

        axs[0].legend()
        axs[1].semilogy(time_arr-time_arr[0], plot_diff_vel, label="velocity error")
        axs[1].set_ylabel(r"$\|\mathbf{v}(t) - \mathbf{\bar{v}}(t)\| [m/s]$", fontsize=13)
        axs[1].tick_params(axis='both', labelsize=13)
        axs[1].legend()
        axs[2].semilogy(time_arr-time_arr[0], plot_diff_acc, label="acceleration error")
        axs[2].set_ylabel(r"$\|\mathbf{a}(t) - \mathbf{\bar{a}}(t)\| [m/s^2]$", fontsize=13)
        axs[2].set_xlabel("Time step [s]", fontsize=14)
        axs[2].tick_params(axis='both', labelsize=13)
        axs[2].legend()
        plt.tight_layout()
        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)
        

        plt.show()
        if case != "case_iv":
            sh_array = np.vstack(list(sh.values()))
            time_keys = list(sh.keys())
            with open(output_directory_file, "a") as f:
                f.write(f"{time_keys[0]} ")
                for elem in sh_array[0]:
                    f.write(f"{elem} ")
                f.write("\n")
                f.write(f"{time_keys[-1]} ")
                for elem in sh_array[-1]:
                    f.write(f"{elem} ")
                f.write("\n")

        # for the different masses
        if case == "case_iv":
            mass_variants = [500, 250]
            sh_mass = {}

            for mass in mass_variants:
                bodies = create_simulation_bodies(sc_mass=mass)
                dynamics_simulator_mass = propagate_trajectory(
                    propag_start_iv,
                    termination_settings,
                    bodies,
                    lambert_arc_ephemeris,
                    use_perturbations=True,
                )
                sh_mass[mass] = dynamics_simulator_mass.propagation_results.state_history

           
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            fig.suptitle("Case IV: Mass scaling analysis", fontsize=16)

            # --- subplot 1: raw position differences ---
            ref_pos = np.vstack(list(sh_reference.values()))[:, 0:3]
            time_ref = np.array(list(sh_reference.keys()))

            all_diffs = {}
            all_times = {}
            for mass, sh_m in sh_mass.items():
                pos_m = np.vstack(list(sh_m.values()))[:, 0:3]
                time_m = np.array(list(sh_m.keys()))
                all_diffs[mass] = np.linalg.norm(pos_m - ref_pos, axis=1)
                all_times[mass] = time_m
                axs[0].semilogy(time_m - time_m[0], all_diffs[mass], label=f"mass = {int(mass)} kg")

            axs[0].set_ylabel(r"$\|\mathbf{x}_{m}(t) - \mathbf{x}_{1000}(t)\|$ [m]", fontsize=13)
            axs[0].tick_params(axis='both', labelsize=13)
            axs[0].legend()
            axs[0].grid(True)

            actual_ratio = all_diffs[500] / (all_diffs[250] + 1e-30)
            time_m = all_times[500]
            axs[1].plot(time_m - time_m[0], actual_ratio, label=r"$\|\Delta x_{500}\| / \|\Delta x_{250}\|$")
            axs[1].axhline(y=2/3, color='r', linestyle='--', label=f"Expected = 2/3 (linear in m)")
            axs[1].axhline(y=1/3, color='g', linestyle='--', label=f"Expected = 1/3 (linear in 1/m)")
            axs[1].set_ylabel(r"Ratio of position deviations [-]", fontsize=13)
            axs[1].set_xlabel("Time step [s]", fontsize=14)
            axs[1].tick_params(axis='both', labelsize=13)
            axs[1].legend()
            axs[1].grid(True)
            plt.tight_layout()
            plt.show()
        


        write_propagation_results_to_file(
            dynamics_simulator,
            lambert_arc_ephemeris,
            "Q2_" + str(cases.index(case)),
            output_directory,
        )



