"""
Standalone helper functions for Earth-Mars low-thrust transfer computation.

Contains only the functions required for:
  - Body creation (with perturbation models)
  - Lambert problem solution
  - SOI crossing detection
  - Perturbed trajectory propagation (with optional RSW empirical acceleration)
  - Variational equations propagation (for sensitivity matrix computation)

All configurable parameters are defined at the top of this file.
No external helper files are required.
"""

import numpy as np
import matplotlib.pyplot as plt
from tudatpy import constants, astro
from tudatpy.astro import element_conversion, two_body_dynamics
from tudatpy.interface import spice
from tudatpy.dynamics import (
    environment_setup, environment,
    propagation_setup, propagation,
    parameters_setup, parameters, simulator,
)

###########################################################################
# CONFIGURABLE MODULE-LEVEL PARAMETERS
###########################################################################

global_frame_orientation = "ECLIPJ2000"

# Integration step size [s].
# For 1000 sub-arcs over ~116 days the arc length is ~10 000 s.
# A 100 s step gives ~100 RK4 steps per arc — accurate and stable.
fixed_step_size = 10.0


###########################################################################
# BODY CREATION
###########################################################################

def create_simulation_bodies(sc_mass: float = 1000.0) -> environment.SystemOfBodies:
    """
    Creates Sun, planets, Moon, and a cannonball-radiation-pressure Spacecraft.

    Parameters
    ----------
    sc_mass : float
        Spacecraft dry mass [kg].

    Returns
    -------
    environment.SystemOfBodies
    """
    bodies_to_create = [
        "Sun", "Earth", "Mars"
        #, "Venus", "Jupiter", "Saturn", "Moon",
    ]
    global_frame_origin = "Sun"
    Sref_rad = 20.0   # reference area [m^2]
    Cr = 1.2           # radiation pressure coefficient

    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation
    )
    body_settings.add_empty_settings("Spacecraft")

    occulting_bodies_dict = dict()
    body_settings.get("Spacecraft").constant_mass = sc_mass
    vehicle_target_settings = (
        environment_setup.radiation_pressure.cannonball_radiation_target(
            Sref_rad, Cr, occulting_bodies_dict
        )
    )
    body_settings.get(
        "Spacecraft"
    ).radiation_pressure_target_settings = vehicle_target_settings

    bodies = environment_setup.create_system_of_bodies(body_settings)
    return bodies


###########################################################################
# LAMBERT PROBLEM
###########################################################################

def get_lambert_problem_result(
    bodies: environment.SystemOfBodies,
    target_body: str,
    departure_epoch: float,
    arrival_epoch: float,
) -> environment.Ephemeris:
    """
    Solves Lambert's problem for an Earth -> target_body transfer and
    returns a Keplerian ephemeris object representing the unperturbed arc.

    Parameters
    ----------
    bodies : environment.SystemOfBodies
        System of bodies (must contain "Sun").
    target_body : str
        Name of the arrival body (e.g. "Mars").
    departure_epoch : float
        Seconds since J2000 at departure.
    arrival_epoch : float
        Seconds since J2000 at arrival.

    Returns
    -------
    environment.Ephemeris
        Keplerian ephemeris of the Lambert arc (valid at any epoch).
    """
    mu_sun = bodies.get_body("Sun").gravitational_parameter

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

    lambert_targeter = two_body_dynamics.LambertTargeterIzzo(
        initial_state[:3],
        final_state[:3],
        arrival_epoch - departure_epoch,
        mu_sun,
    )

    lambert_arc_initial_state = initial_state.copy()
    lambert_arc_initial_state[3:] = lambert_targeter.get_departure_velocity()

    lambert_arc_keplerian_elements = element_conversion.cartesian_to_keplerian(
        lambert_arc_initial_state, mu_sun
    )

    kepler_ephemeris = environment_setup.create_body_ephemeris(
        environment_setup.ephemeris.keplerian(
            lambert_arc_keplerian_elements,
            departure_epoch,
            mu_sun,
        ),
        "",
    )
    return kepler_ephemeris


###########################################################################
# LAMBERT ARC HISTORY EXTRACTION
###########################################################################

def get_lambert_arc_history(
    lambert_arc_ephemeris: environment.Ephemeris,
    simulation_result: dict,
) -> dict:
    """
    Evaluate the Lambert arc ephemeris at every epoch present in
    *simulation_result* and return the resulting state dictionary.
    """
    lambert_arc_states = dict()
    for epoch in simulation_result:
        lambert_arc_states[epoch] = lambert_arc_ephemeris.cartesian_state(epoch)
    return lambert_arc_states


###########################################################################
# SOI CROSSING DETECTION
###########################################################################

def find_propagation_time_soi(
    lambert_arc_ephemeris: environment.Ephemeris,
    bodies: environment.SystemOfBodies,
    departure_epoch: float,
    departure_body: str,
    arrival_body: str,
    arrival_epoch: float,
):
    """
    Scans the Lambert arc from *departure_epoch* to *arrival_epoch* and
    returns the first epoch outside the departure body's SOI and the last
    epoch outside the arrival body's SOI.

    SOI radii are computed via the Hill-sphere approximation:
        r_SOI = a * (mu_body / mu_Sun)^(2/5)

    Parameters
    ----------
    lambert_arc_ephemeris : environment.Ephemeris
    bodies : environment.SystemOfBodies
    departure_epoch : float   [s since J2000]
    departure_body  : str     (e.g. "Earth")
    arrival_body    : str     (e.g. "Mars")
    arrival_epoch   : float   [s since J2000]

    Returns
    -------
    soi_departure_epoch : float
    soi_departure_state : np.ndarray (6,)
    soi_arrival_epoch   : float
    soi_arrival_state   : np.ndarray (6,)
    soi_tof             : float
    """
    mu_sun = spice.get_body_gravitational_parameter("Sun")

    # Semi-major axes (approximate — using states at departure/arrival epochs)
    dep_body_state = spice.get_body_cartesian_state_at_epoch(
        departure_body, "Sun", global_frame_orientation, "None", departure_epoch
    )
    a_dep = astro.element_conversion.cartesian_to_keplerian(dep_body_state, mu_sun)[0]

    arr_body_state = spice.get_body_cartesian_state_at_epoch(
        arrival_body, "Sun", global_frame_orientation, "None", arrival_epoch
    )
    a_arr = astro.element_conversion.cartesian_to_keplerian(arr_body_state, mu_sun)[0]

    # SOI radii
    r_soi_dep = a_dep * (
        bodies.get(departure_body).gravitational_parameter
        / bodies.get("Sun").gravitational_parameter
    ) ** (2 / 5)
    r_soi_arr = a_arr * (
        bodies.get(arrival_body).gravitational_parameter
        / bodies.get("Sun").gravitational_parameter
    ) ** (2 / 5)

    # Dense time grid for scanning
    n_samples = int((arrival_epoch - departure_epoch) / fixed_step_size)
    time_array = np.linspace(departure_epoch, arrival_epoch, max(n_samples, 10000))

    # --- Forward scan: first epoch outside departure SOI ---
    soi_departure_epoch = None
    soi_departure_state = None
    for t in time_array:
        state = lambert_arc_ephemeris.cartesian_state(t)
        pos_body = spice.get_body_cartesian_position_at_epoch(
            departure_body, "Sun", global_frame_orientation, "None", t
        )[0:3]
        if np.linalg.norm(state[0:3] - pos_body) > r_soi_dep:
            soi_departure_epoch = t
            soi_departure_state = state
            break

    if soi_departure_epoch is None:
        raise RuntimeError("Lambert arc never exits departure body SOI.")

    # --- Backward scan: last epoch outside arrival SOI ---
    soi_arrival_epoch = None
    soi_arrival_state = None
    for t in reversed(time_array):
        state = lambert_arc_ephemeris.cartesian_state(t)
        pos_body = spice.get_body_cartesian_position_at_epoch(
            arrival_body, "Sun", global_frame_orientation, "None", t
        )[0:3]
        if np.linalg.norm(state[0:3] - pos_body) > r_soi_arr:
            soi_arrival_epoch = t
            soi_arrival_state = state
            break

    if soi_arrival_epoch is None:
        raise RuntimeError("Lambert arc never enters arrival body SOI.")

    soi_tof = soi_arrival_epoch - soi_departure_epoch
    return (
        soi_departure_epoch,
        soi_departure_state,
        soi_arrival_epoch,
        soi_arrival_state,
        soi_tof,
    )


###########################################################################
# PERTURBED PROPAGATOR SETTINGS
###########################################################################

def get_perturbed_propagator_settings(
    bodies: environment.SystemOfBodies,
    initial_state: np.ndarray,
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    use_rsw_acceleration: bool = False,
    rsw_acceleration_magnitude: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> propagation_setup.propagator.SingleArcPropagatorSettings:
    """
    Propagator settings for a perturbed heliocentric trajectory.

    Acceleration model
    ------------------
    Sun   : point-mass gravity + solar radiation pressure
    Earth : point-mass gravity
    Moon  : point-mass gravity
    Mars  : point-mass gravity
    Venus : point-mass gravity
    Jupiter : point-mass gravity
    Saturn  : point-mass gravity
    (optional) constant empirical RSW acceleration on top of Sun gravity

    Parameters
    ----------
    bodies : environment.SystemOfBodies
    initial_state : np.ndarray (6,)
    initial_time : float
    termination_condition : PropagationTerminationSettings
    use_rsw_acceleration : bool
    rsw_acceleration_magnitude : np.ndarray (3,)
        Constant acceleration in [R, S, W] frame [m/s^2].

    Returns
    -------
    SingleArcPropagatorSettings
    """
    acceleration_settings_on_spacecraft = dict(
        Sun=[
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.radiation_pressure(),
        ],
        Earth=[propagation_setup.acceleration.point_mass_gravity()],
        Mars=[propagation_setup.acceleration.point_mass_gravity()],
        # Moon=[propagation_setup.acceleration.point_mass_gravity()],
        # Jupiter=[propagation_setup.acceleration.point_mass_gravity()],
        # Saturn=[propagation_setup.acceleration.point_mass_gravity()],
        # Venus=[propagation_setup.acceleration.point_mass_gravity()],
    )

    if use_rsw_acceleration:
        acceleration_settings_on_spacecraft["Sun"].append(
            propagation_setup.acceleration.empirical(rsw_acceleration_magnitude)
        )

    acceleration_settings = {"Spacecraft": acceleration_settings_on_spacecraft}
    bodies_to_propagate = ["Spacecraft"]
    central_bodies = ["Sun"]

    dependent_variables_to_save = [
        propagation_setup.dependent_variable.total_acceleration("Spacecraft"),
    ]

    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        fixed_step_size,
        coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4,
    )

    propagator_settings = propagation_setup.propagator.translational(
        central_bodies=central_bodies,
        acceleration_models=acceleration_models,
        bodies_to_integrate=bodies_to_propagate,
        initial_states=initial_state,
        initial_time=initial_time,
        integrator_settings=integrator_settings,
        termination_settings=termination_condition,
        output_variables=dependent_variables_to_save,
    )
    return propagator_settings


###########################################################################
# TRAJECTORY PROPAGATION
###########################################################################

def propagate_trajectory(
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    bodies: environment.SystemOfBodies,
    lambert_arc_ephemeris: environment.Ephemeris,
    initial_state_correction: np.ndarray = np.array([0, 0, 0, 0, 0, 0]),
    use_rsw_acceleration: bool = False,
    rsw_acceleration_magnitude: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> simulator.SingleArcSimulator:
    """
    Propagates a perturbed trajectory starting from the Lambert arc state
    (optionally corrected) at *initial_time*.

    Parameters
    ----------
    initial_time : float
    termination_condition : PropagationTerminationSettings
    bodies : environment.SystemOfBodies
    lambert_arc_ephemeris : environment.Ephemeris
    initial_state_correction : np.ndarray (6,)
        Added to the Lambert state at initial_time.
    use_rsw_acceleration : bool
    rsw_acceleration_magnitude : np.ndarray (3,)

    Returns
    -------
    simulator.SingleArcSimulator
    """
    lambert_arc_initial_state = (
        lambert_arc_ephemeris.cartesian_state(initial_time) + initial_state_correction
    )

    propagator_settings = get_perturbed_propagator_settings(
        bodies,
        lambert_arc_initial_state,
        initial_time,
        termination_condition,
        use_rsw_acceleration,
        rsw_acceleration_magnitude,
    )

    dynamics_simulator = simulator.create_dynamics_simulator(
        bodies, propagator_settings
    )
    return dynamics_simulator


###########################################################################
# VARIATIONAL EQUATIONS
###########################################################################

def get_sensitivity_parameter_set(
    propagator_settings: propagation_setup.propagator.PropagatorSettings,
    bodies: environment.SystemOfBodies,
    use_rsw_acceleration: bool = False,
) -> parameters.EstimatableParameterSet:
    """
    Defines the parameter set for the variational equations.

    When *use_rsw_acceleration* is True the sensitivity matrix will contain
    columns for the three constant empirical-acceleration components (R, S, W).
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


def propagate_variational_equations(
    initial_time: float,
    termination_condition: propagation_setup.propagator.PropagationTerminationSettings,
    bodies: environment.SystemOfBodies,
    lambert_arc_ephemeris: environment.Ephemeris,
    initial_state_correction: np.ndarray = np.array([0, 0, 0, 0, 0, 0]),
    use_rsw_acceleration: bool = False,
    rsw_acceleration_magnitude: np.ndarray = np.array([0.0, 0.0, 0.0]),
) -> simulator.SingleArcVariationalSimulator:
    """
    Propagates the variational equations for a perturbed trajectory,
    yielding the state transition matrix Phi(t, t0) and — when
    *use_rsw_acceleration* is True — the sensitivity matrix S(t)
    mapping the three RSW acceleration components to the 6-D state.

    Parameters
    ----------
    initial_time : float
    termination_condition : PropagationTerminationSettings
    bodies : environment.SystemOfBodies
    lambert_arc_ephemeris : environment.Ephemeris
    initial_state_correction : np.ndarray (6,)
    use_rsw_acceleration : bool
    rsw_acceleration_magnitude : np.ndarray (3,)

    Returns
    -------
    simulator.SingleArcVariationalSimulator
        .state_history                  — dict {epoch: state(6,)}
        .state_transition_matrix_history — dict {epoch: Phi(6,6)}
        .sensitivity_matrix_history      — dict {epoch: S(6,3)}
    """
    lambert_arc_initial_state = (
        lambert_arc_ephemeris.cartesian_state(initial_time) + initial_state_correction
    )

    propagator_settings = get_perturbed_propagator_settings(
        bodies,
        lambert_arc_initial_state,
        initial_time,
        termination_condition,
        use_rsw_acceleration,
        rsw_acceleration_magnitude,
    )

    sensitivity_parameters = get_sensitivity_parameter_set(
        propagator_settings, bodies, use_rsw_acceleration
    )

    variational_equations_solver = simulator.create_variational_equations_solver(
        bodies, propagator_settings, sensitivity_parameters
    )
    return variational_equations_solver