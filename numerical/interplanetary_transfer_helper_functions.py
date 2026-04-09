""" 
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and 
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
"""

import numpy as np
import matplotlib.pyplot as plt
from tudatpy import constants, astro
from tudatpy.astro import element_conversion, two_body_dynamics
from tudatpy.data import save2txt
from tudatpy.interface import spice
from tudatpy.dynamics import environment_setup, environment, propagation_setup, propagation, parameters_setup, parameters, simulator

# Define departure/arrival epoch - in seconds since J2000


departure_epoch = 340.6905189 * constants.JULIAN_DAY
time_of_flight = 165.3067532 * constants.JULIAN_DAY
arrival_epoch = departure_epoch + time_of_flight
target_body = "Venus" #Venus
global_frame_orientation = "ECLIPJ2000"
fixed_step_size = 3600.0

################ HELPER FUNCTIONS: DO NOT MODIFY ########################################



# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def write_propagation_results_to_file(
    dynamics_simulator: simulator.SingleArcSimulator,
    lambert_arc_ephemeris: environment.Ephemeris,
    file_output_identifier: str,
    output_directory: str,
) -> None:
    """
    This function will write the results of a numerical propagation, as well as the Lambert arc states at the epochs of the
    numerical state history, to a set of files. Two files are always written when calling this function (numerical state history, a
    and Lambert arc state history). If any dependent variables are saved during the propagation, those are also saved to a file

    Parameters
    ----------
    dynamics_simulator : simulator.SingleArcSimulator
        Object that was used to propagate the dynamics, and which contains the numerical state and dependent variable results

    lambert_arc_ephemeris : environment.Ephemeris
        Lambert arc state model as returned by the get_lambert_problem_result() function

    file_output_identifier : str
        Name that will be used to correctly save the output data files

    output_directory : str
        Directory to which the files will be written

    Files written
    -------------

    <output_directory/file_output_identifier>_numerical_states.dat
    <output_directory/file_output_identifier>_dependent_variables.dat
    <output_directory/file_output_identifier>_lambert_states.dat


    Return
    ------
    None

    """

    propagation_results = dynamics_simulator.propagation_results

    # Save numerical states
    state_history = propagation_results.state_history
    save2txt(
        solution=state_history,
        filename=output_directory + file_output_identifier + "_numerical_states.dat",
        directory="./",
    )

    # Save dependent variables
    dependent_variables = propagation_results.dependent_variable_history
    if len(dependent_variables.keys()) > 0:
        save2txt(
            solution=dependent_variables,
            filename=output_directory
            + file_output_identifier
            + "_dependent_variables.dat",
            directory="./",
        )

    # Save Lambert arc states
    lambert_arc_states = get_lambert_arc_history(lambert_arc_ephemeris, state_history)

    save2txt(
        solution=lambert_arc_states,
        filename=output_directory + file_output_identifier + "_lambert_states.dat",
        directory="./",
    )

    return


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_lambert_problem_result(
    bodies: environment.SystemOfBodies,
    target_body: str,
    departure_epoch: float,
    arrival_epoch: float,
) -> environment.Ephemeris:
    """
    This function solved Lambert's problem for a transfer from Earth (at departure epoch) to
    a target body (at arrival epoch), with the states of Earth and the target body defined
    by ephemerides stored inside the SystemOfBodies object (bodies). Note that this solver
    assumes that the transfer departs/arrives to/from the center of mass of Earth and the target body

    Parameters
    ----------
    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    target_body : str
        The name (string) of the body to which the Lambert arc is to be computed

    departure_epoch : float
        Epoch at which the departure from Earth's center of mass is to take place

    arrival_epoch : float
        Epoch at which the arrival at he target body's center of mass is to take place

    Return
    ------
    Ephemeris object defining a purely Keplerian trajectory. This Keplerian trajectory defines the transfer
    from Earth to the target body according to the inputs to this function. Note that this Ephemeris object
    is valid before the departure epoch, and after the arrival epoch, and simply continues (forwards and backwards)
    the unperturbed Sun-centered orbit, as fully defined by the unperturbed transfer arc
    """

    # Gravitational parameter of the Sun
    central_body_gravitational_parameter = bodies.get_body(
        "Sun"
    ).gravitational_parameter

    # Set initial and final positions for Lambert targeter
    initial_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name="Earth",
        observer_body_name="Sun",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=departure_epoch,
    )

    final_state = spice.get_body_cartesian_state_at_epoch(
        target_body_name=target_body,
        observer_body_name="Sun",
        reference_frame_name=global_frame_orientation,
        aberration_corrections="NONE",
        ephemeris_time=arrival_epoch,
    )

    # Create Lambert targeter
    lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
        initial_state[:3],
        final_state[:3],
        arrival_epoch - departure_epoch,
        central_body_gravitational_parameter,
    )

    # Compute initial Cartesian state of Lambert arc
    lambert_arc_initial_state = initial_state
    lambert_arc_initial_state[3:] = lambertTargeter.get_departure_velocity()

    # Compute Keplerian state of Lambert arc
    lambert_arc_keplerian_elements = element_conversion.cartesian_to_keplerian(
        lambert_arc_initial_state, central_body_gravitational_parameter
    )

    # Setup Keplerian ephemeris model that describes the Lambert arc
    kepler_ephemeris = environment_setup.create_body_ephemeris(
        environment_setup.ephemeris.keplerian(
            lambert_arc_keplerian_elements,
            departure_epoch,
            central_body_gravitational_parameter,
        ),
        "",  # for keplerian ephemeris, this argument does not have an effect
    )

    return kepler_ephemeris


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_lambert_arc_history(
    lambert_arc_ephemeris: environment.Ephemeris, simulation_result: dict
) -> dict:
    """
    This function extracts the state history (as a dict with time as keys, and Cartesian states as values)
    from an Ephemeris object defined by a lambert solver. This function takes a dictionary of states (simulation_result)
    as input, iterates over the keys of this dict (which represent times) to ensure that the times
    at which this function returns the states of the lambert arcs are identical to those at which the
    simulation_result has (numerically calculated) states


    Parameters
    ----------
    lambert_arc_ephemeris : environment.Ephemeris
        Ephemeris object from which the states are to be extracted

    simulation_result : dict
        Dictionary of (numerically propagated) states, from which the keys
        are used to determine the times at which this function is to extract states
        from the lambert arc

    Return
    ------
    Dictionary of Cartesian states of the lambert arc, with the keys (epochs) being the same as those of the input
    simulation_result and the corresponding Cartesian states of the Lambert arc.
    """

    lambert_arc_states = dict()
    for epoch in simulation_result:
        lambert_arc_states[epoch] = lambert_arc_ephemeris.cartesian_state(epoch)

    return lambert_arc_states


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def propagate_trajectory(
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    bodies: environment.SystemOfBodies,
    lambert_arc_ephemeris: environment.Ephemeris,
    use_perturbations: bool,
    initial_state_correction=np.array([0, 0, 0, 0, 0, 0]),
    use_rsw_acceleration: bool = False,
    rsw_acceleration_magnitude: np.ndarray = np.array([0,0,0])
) -> simulator.SingleArcSimulator:
    """
    This function will be repeatedly called throughout the assignment. Propagates the trajectory based
    on several input parameters

    Parameters
    ----------
    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    lambert_arc_ephemeris : environment.Ephemeris
        Lambert arc state model as returned by the get_lambert_problem_result() function

    use_perturbations : bool
        Boolean to indicate whether a perturbed (True) or unperturbed (False) trajectory
        is propagated

    initial_state_correction : np.ndarray, default=np.array([0, 0, 0, 0, 0, 0])
        Cartesian state which is added to the Lambert arc state when computing the numerical initial state

    use_rsw_acceleration: Boolean defining whether an RSW acceleration (used to denote thrust) is to be used

    rsw_acceleration_magnitude: Magnitude of RSW acceleration, to be used if use_rsw_acceleration == True
                                the entries of this vector denote the acceleration in radial, normal and cross-track,
                                respectively.

    Return
    ------
    Dynamics simulator object from which the state- and dependent variable history can be extracted

    """

    # Compute initial state along Lambert arc (and apply correction if needed)
    lambert_arc_initial_state = (
        lambert_arc_ephemeris.cartesian_state(initial_time) + initial_state_correction
    )

    # Get propagator settings for perturbed/unperturbed forwards/backwards arcs
    if use_perturbations:
        propagator_settings = get_perturbed_propagator_settings(
            bodies, lambert_arc_initial_state, initial_time, termination_condition, use_rsw_acceleration, rsw_acceleration_magnitude
        )

    else:
        propagator_settings = get_unperturbed_propagator_settings(
            bodies, lambert_arc_initial_state, initial_time, termination_condition
        )

    # Propagate dynamics with required settings
    dynamics_simulator = simulator.create_dynamics_simulator(
        bodies, propagator_settings
    )

    return dynamics_simulator


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def propagate_variational_equations(
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    bodies: environment.SystemOfBodies,
    lambert_arc_ephemeris: environment.Ephemeris,
    initial_state_correction=np.array([0, 0, 0, 0, 0, 0]),
    use_rsw_acceleration: bool = False,
    rsw_acceleration_magnitude: np.ndarray = np.array([0,0,0])
) -> simulator.SingleArcVariationalSimulator:
    """
    Propagates the variational equations for a given range of epochs for a perturbed trajectory.

    Parameters
    ----------
    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    lambert_arc_ephemeris : environment.Ephemeris
        Lambert arc state model as returned by the get_lambert_problem_result() function

    initial_state_correction : np.ndarray, default=np.array([0, 0, 0, 0, 0, 0])
        Cartesian state which is added to the Lambert arc state when computing the numerical initial state

    use_rsw_acceleration: Boolean defining whether an RSW acceleration (used to denote thrust) is to be used

    rsw_acceleration_magnitude: Magnitude of RSW acceleration, to be used if use_rsw_acceleration == True
                                the entries of this vector denote the acceleration in radial, normal and cross-track,
                                respectively.

    Return
    ------
    Variational equations solver object, from which the state-, state transition matrix-, and
    sensitivity matrix history can be extracted.
    """

    # Compute initial state along Lambert arc
    lambert_arc_initial_state = (
        lambert_arc_ephemeris.cartesian_state(initial_time) + initial_state_correction
    )

    # Get propagator settings
    propagator_settings = get_perturbed_propagator_settings(
        bodies,
        lambert_arc_initial_state,
        initial_time,
        termination_condition,
        use_rsw_acceleration,
        rsw_acceleration_magnitude
    )

    # Define parameters for variational equations
    sensitivity_parameters = get_sensitivity_parameter_set(propagator_settings, bodies, use_rsw_acceleration)

    # Propagate variational equations
    variational_equations_solver = (
        simulator.create_variational_equations_solver(
            bodies, propagator_settings, sensitivity_parameters
        )
    )

    return variational_equations_solver


# DO NOT MODIFY THIS FUNCTION (OR, DO SO AT YOUR OWN RISK)
def get_sensitivity_parameter_set(
    propagator_settings: propagation_setup.propagator.PropagatorSettings,
    bodies: environment.SystemOfBodies,
    use_rsw_acceleration: bool = False
) -> parameters.EstimatableParameterSet:
    """
    Function creating the parameters for which the variational equations are to be solved.

    Parameters
    ----------
    propagator_settings : propagation_setup.propagator.PropagatorSettings
        Settings used for the propagation of the dynamics

    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    use_rsw_acceleration : Boolean denoting whether the sensitivity to an RSW acceleration is to be
                           included. Note that this can only be used (set to True) is the acceleration models
                           in propagator_settings contain an empirical acceleration

    Return
    ------
    Propagation settings of the unperturbed trajectory.
    """
    parameter_settings = parameters_setup.initial_states(
        propagator_settings, bodies
    )

    if use_rsw_acceleration:
        parameter_settings.append(
            parameters_setup.constant_empirical_acceleration_terms(
                "Spacecraft", "Sun"
            )
        )

    return parameters_setup.create_parameter_set(
        parameter_settings, bodies, propagator_settings
    )


################ HELPER FUNCTIONS: MODIFY ########################################


# STUDENT CODE TASK - full function (except signature and return)
def get_unperturbed_propagator_settings(
    bodies: environment.SystemOfBodies,
    initial_state: np.ndarray,
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
) -> propagation_setup.propagator.SingleArcPropagatorSettings:
    """
    Creates the propagator settings for an unperturbed trajectory.

    Parameters
    ----------
    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    initial_state : np.ndarray
        Cartesian initial state of the vehicle in the simulation

    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    Return
    ------
    Propagation settings of the unperturbed trajectory.
    """

    # Define variables
    acceleration_settings_on_spacecraft = {"Sun": [propagation_setup.acceleration.point_mass_gravity()]}
    acceleration_settings = {"Spacecraft": acceleration_settings_on_spacecraft}
    bodies_to_propagate = ["Spacecraft"]
    central_bodies = ["Sun"]
    dependent_variables_to_save = [propagation_setup.dependent_variable.relative_position("Earth", "Sun"),
                                   propagation_setup.dependent_variable.relative_position("Venus", "Sun")]

    # Create required models
    acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies
        )
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
    )

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies = central_bodies,
        acceleration_models=acceleration_models,
        bodies_to_integrate=bodies_to_propagate,
        initial_states=initial_state,
        initial_time=initial_time,
        integrator_settings=integrator_settings,
        termination_settings=termination_condition,
        output_variables=dependent_variables_to_save
)
    return propagator_settings


# STUDENT CODE TASK - full function (except signature and return)
def get_perturbed_propagator_settings(
    bodies: environment.SystemOfBodies,
    initial_state: np.ndarray,
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    use_rsw_acceleration: bool = False,
    rsw_acceleration_magnitude: np.ndarray = np.array([0,0,0])
) -> propagation_setup.propagator.SingleArcPropagatorSettings:
    """
    Creates the propagator settings for a perturbed trajectory.

    Parameters
    ----------
    bodies : environment.SystemOfBodies
        Body objects defining the physical simulation environment

    initial_state : np.ndarray
        Cartesian initial state of the vehicle in the simulation

    initial_time : float
        Epoch since J2000 at which the propagation starts

    termination_condition : propagation_setup.propagator.PropagationTerminationSettings
        Settings for condition upon which the propagation will be terminated

    use_rsw_acceleration: Boolean defining whether an RSW acceleration (used to denote thrust) is to be used

    rsw_acceleration_magnitude: Magnitude of RSW acceleration, to be used if use_rsw_acceleration == True
                                the entries of this vector denote the acceleration in radial, normal and cross-track,
                                respectively.

    Return
    ------
    Propagation settings of the perturbed trajectory.
    """

    # Define accelerations acting on vehicle.


    acceleration_settings_on_spacecraft = dict(
    Sun = [
        propagation_setup.acceleration.point_mass_gravity(),
        propagation_setup.acceleration.radiation_pressure()
        ],
    Moon =
        [
        propagation_setup.acceleration.point_mass_gravity(),
        ],
    Jupiter = 
        [
        propagation_setup.acceleration.point_mass_gravity(),
        ],
    Saturn = [
        propagation_setup.acceleration.point_mass_gravity()
        ],
    Earth = [
        propagation_setup.acceleration.point_mass_gravity()
        ],
    Mars= [
        propagation_setup.acceleration.point_mass_gravity()
        ],
    Venus = [
        propagation_setup.acceleration.point_mass_gravity()
        ]
)


    
    # DO NOT MODIFY, and keep AFTER creation of acceleration_settings_on_spacecraft, but before
    # call to function create_acceleration_models
    # (line is added for compatibility with question 4)
    if use_rsw_acceleration:
        acceleration_settings_on_spacecraft["Sun"].append(
            propagation_setup.acceleration.empirical(rsw_acceleration_magnitude))

    acceleration_settings = {"Spacecraft": acceleration_settings_on_spacecraft}

    # Create propagation settings.
    bodies_to_propagate = ["Spacecraft"]
    central_bodies = ["Sun"]
    dependent_variables_to_save = [propagation_setup.dependent_variable.total_acceleration("Spacecraft"),]
    

    # Create required models
    acceleration_models = propagation_setup.create_acceleration_models(
            bodies, acceleration_settings, bodies_to_propagate, central_bodies
        )
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size, coefficient_set = propagation_setup.integrator.CoefficientSets.rk_4
    )

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies = central_bodies,
        acceleration_models=acceleration_models,
        bodies_to_integrate=bodies_to_propagate,
        initial_states=initial_state,
        initial_time=initial_time,
        integrator_settings=integrator_settings,
        termination_settings=termination_condition,
        output_variables=dependent_variables_to_save
)
    return propagator_settings


# STUDENT CODE TASK - full function (except signature and return)
# NOTE: Keep this function the same for each question (it does no harm if bodies are
# added that are not used)
def create_simulation_bodies(sc_mass = 1000) -> environment.SystemOfBodies:
    """
    Creates the body objects required for the simulation, using the
    environment_setup.create_system_of_bodies for natural bodies,
    and manual definition for vehicles

    Parameters
    ----------
    none

    Return
    ------
    Body objects required for the simulation.

    """
    bodies_to_create = ["Venus", "Sun", "Earth", "Moon", "Jupiter", "Saturn", "Mars"]
    global_frame_origin = "Sun"
    Sref_rad = 20.0
    Cr = 1.2

    global_frame_orientation = "ECLIPJ2000"
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation
    )
    body_settings.add_empty_settings("Spacecraft")

    # Add radiation pressure settings to the bodies
    occulting_bodies_dict = dict()
    body_settings.get("Spacecraft").constant_mass = sc_mass
    vehicle_target_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
        Sref_rad, Cr, occulting_bodies_dict
    )
    body_settings.get("Spacecraft").radiation_pressure_target_settings = vehicle_target_settings
    bodies = environment_setup.create_system_of_bodies(body_settings)

    return bodies





def plot_orbits_3d(*trajectories, title="Lambert Targeter"):
    AU = 1.496e11
    fig = plt.figure(figsize=(9, 7), facecolor="#0d1117")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#0d1117")

    for arr, label, color in trajectories:
        x, y, z = arr[:,0]/AU, arr[:,1]/AU, arr[:,2]/AU
        ax.plot(x, y, z, color=color, lw=1.5, label=label)

    max_range = max(
        max(abs(arr[:,0]/AU).max() for arr,_,_ in trajectories),
        max(abs(arr[:,1]/AU).max() for arr,_,_ in trajectories),
    )
    z_min = min(arr[:,2].min()/AU for arr,_,_ in trajectories)
    z_max = max(arr[:,2].max()/AU for arr,_,_ in trajectories)

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(z_min, z_max)
    ax.set_box_aspect([1, 1, 1])

    x_lo = ax.get_xlim()[0]
    y_hi = ax.get_ylim()[1]
    z_lo = ax.get_zlim()[0]

    # Projections + start/end markers
    for arr, label, color in trajectories:
        x, y, z = arr[:,0]/AU, arr[:,1]/AU, arr[:,2]/AU

        # Trajectory projections
        ax.plot(x, y, zs=z_lo, zdir='z', color=color, lw=0.8, alpha=0.25)  # XY
        ax.plot(x, z, zs=y_hi, zdir='y', color=color, lw=0.8, alpha=0.25)  # XZ
        ax.plot(y, z, zs=x_lo, zdir='x', color=color, lw=0.8, alpha=0.25)  # YZ

        # Start/end on trajectory
        ax.scatter(x[0],  y[0],  z[0],  color=color, s=40, marker='o', zorder=10, depthshade=False)
        ax.scatter(x[-1], y[-1], z[-1], color=color, s=40, marker='s', zorder=10, depthshade=False)

        # Start/end on XY projection
        ax.scatter(x[0],  y[0],  z_lo, color=color, s=20, marker='o', alpha=0.25, zorder=5, depthshade=False)
        ax.scatter(x[-1], y[-1], z_lo, color=color, s=20, marker='s', alpha=0.25, zorder=5, depthshade=False)

        # Start/end on XZ projection
        ax.scatter(x[0],  y_hi, z[0],  color=color, s=20, marker='o', alpha=0.25, zorder=5, depthshade=False)
        ax.scatter(x[-1], y_hi, z[-1], color=color, s=20, marker='s', alpha=0.25, zorder=5, depthshade=False)

        # Start/end on YZ projection
        ax.scatter(x_lo, y[0],  z[0],  color=color, s=20, marker='o', alpha=0.25, zorder=5, depthshade=False)
        ax.scatter(x_lo, y[-1], z[-1], color=color, s=20, marker='s', alpha=0.25, zorder=5, depthshade=False)

    # Sun + projections
    ax.scatter(0, 0, 0,      s=120, color="#fff176", zorder=10, label="Sun", depthshade=False)
    ax.scatter(0, 0,    z_lo, color="#fff176", s=60, alpha=0.25, zorder=5, depthshade=False)
    ax.scatter(0, y_hi, 0,   color="#fff176", s=60, alpha=0.25, zorder=5, depthshade=False)
    ax.scatter(x_lo, 0, 0,   color="#fff176", s=60, alpha=0.25, zorder=5, depthshade=False)

    ax.set_xlabel("X [AU]", color="#8b949e", fontsize=8)
    ax.set_ylabel("Y [AU]", color="#8b949e", fontsize=8)
    ax.set_zlabel("Z [AU]", color="#8b949e", fontsize=8)
    ax.tick_params(colors="#8b949e", labelsize=7)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#21262d")

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo['grid'].update({'color': '#21262d', 'linewidth': 0.5})

    ax.set_title(title, color="#e6edf3", fontsize=12, pad=12)

    from matplotlib.lines import Line2D
    extra_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e6edf3', markersize=6, label='Departure', linestyle='None'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#e6edf3', markersize=6, label='Arrival',   linestyle='None'),
    ]
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=existing_handles + extra_handles,
        framealpha=0.1, edgecolor="#30363d", labelcolor="#e6edf3", fontsize=8
    )

    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    plt.show()


def find_propagation_time_soi(lambert_arc_ephemeris, bodies, departure_epoch, departure_body, arrival_body, arrival_epoch):
    # Calculate SOI departure epoch

    time_array = np.linspace(departure_epoch, arrival_epoch, int((arrival_epoch-departure_epoch)/fixed_step_size))

    # approximation semimajoraxis \sim constant
    earth_state = spice.get_body_cartesian_state_at_epoch(departure_body, "Sun", "ECLIPJ2000", "None", 892.0386735 * constants.JULIAN_DAY)
    mu_sun = spice.get_body_gravitational_parameter("Sun")
    a_earth = astro.element_conversion.cartesian_to_keplerian(earth_state, mu_sun)[0]
    venus_state = spice.get_body_cartesian_state_at_epoch(arrival_body, "Sun", "ECLIPJ2000", "None", 892.0386735 * constants.JULIAN_DAY)
    a_venus = astro.element_conversion.cartesian_to_keplerian(venus_state, mu_sun)[0]

    bool = True
    index_check = 0
    while bool:
        time_instant = time_array[index_check]
        lambert_state = lambert_arc_ephemeris.cartesian_state(time_instant)
        pos_lamb = lambert_state[0:3]
        pos_earth_dep = spice.get_body_cartesian_position_at_epoch(
            departure_body, "Sun", "ECLIPJ2000", "None", time_instant)[0:3]
        r_soi_earth = a_earth * (bodies.get(departure_body).gravitational_parameter 
                    / bodies.get("Sun").gravitational_parameter)**(2/5)
        if np.abs(np.linalg.norm(pos_lamb - pos_earth_dep)) > r_soi_earth:
            bool = False
            departure_epoch = time_instant
            departure_state = lambert_state
        index_check += 1

    # Calculate SOI arrival epoch
    bool = True
    index_check = len(time_array) - 1
    while bool:
        time_instant = time_array[index_check]
        lambert_state = lambert_arc_ephemeris.cartesian_state(time_instant)
        pos_lamb = lambert_state[0:3]
        pos_venus_arr = spice.get_body_cartesian_position_at_epoch(
            arrival_body, "Sun", "ECLIPJ2000", "None", time_instant)[0:3]
        r_soi_venus = a_venus * (bodies.get(arrival_body).gravitational_parameter 
                    / bodies.get("Sun").gravitational_parameter)**(2/5)
        if np.abs(np.linalg.norm(pos_lamb - pos_venus_arr)) > r_soi_venus:
            bool = False
            arrival_epoch = time_instant
            arrival_state = lambert_state
            time_of_flight = arrival_epoch - departure_epoch
        index_check -= 1
    return departure_epoch, departure_state, arrival_epoch, arrival_state, time_of_flight


def iterative_correction_low_thrust(bodies, lambert_arc_ephemeris, sensitivity_matrix_final_epoch, delta_x_final, termination_settings,
    start_epoch, initial_state_correction, first_low_thrust_guess, tolerance = 1):

    delta_r_final = delta_x_final[-1, 0:3]
    p = first_low_thrust_guess
    it = 0
    while np.linalg.norm(delta_r_final) > tolerance:
        it += 1


        dp = np.linalg.solve(sensitivity_matrix_final_epoch, -delta_r_final)
        p += dp

        dynamics_simulator = propagate_trajectory(
            start_epoch, termination_settings, bodies, lambert_arc_ephemeris,
            use_perturbations=True,
            initial_state_correction=initial_state_correction,
            use_rsw_acceleration=True,
            rsw_acceleration_magnitude=p,
        )
        sh = dynamics_simulator.propagation_results.state_history
        lamb_hist = get_lambert_arc_history(lambert_arc_ephemeris, sh)
        lamb_array = np.vstack(list(lamb_hist.values()))
        sh_array      = np.vstack(list(sh.values()))
        delta_x_tfinal     = sh_array - lamb_array
        delta_r_final      = delta_x_tfinal[-1, 0:3]

    print(f"Converged in {it} iterations. Final position error: {np.linalg.norm(delta_r_final):.2f} m")
    return p, delta_x_tfinal, it, sh_array, sh