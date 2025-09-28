import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from base_methods import compute_dt, save, better_interpolator, save_conditions
from all_variables import *
from flow_equations import z_momentum_s, r_momentum_s, energy_s, divergence_s
from flow_equations import temperature_boundaries, velocity_boundaries, pressure_correction_iterate
from animation import animation, animation_vertical
from single_particle_solver import Particle, PARTICLE_INSERTION


def one_iteration_flow(uz_prev, ur_prev, rho_prev, p_prev, T_prev, T_next, Fz, Fr, dt):

    global current_time
    p_null = np.zeros_like(p_prev)
    rho_next = rhof(T_next)

    uz_tent, rho_uz_tent = z_momentum_s(uz_prev, ur_prev, rho_next, rho_prev, p_null, T_next, Fz, dt)
    ur_tent, rho_ur_tent = r_momentum_s(uz_prev, ur_prev, rho_next, rho_prev, p_null, T_next, Fr, dt)

    uz_next, ur_next, p_next, divs = pressure_correction_iterate(rho_prev, rho_next, p_prev,
                                                                 uz_tent, ur_tent, dt,
                                                                 N_DIVERGENCE_ITERATIONS, current_time)

    velocity_boundaries(uz_next, ur_next)

    divergence = divergence_s(rho_prev, uz_tent, ur_tent)
    divergence2 = divergence_s(rho_next, uz_next, ur_next)
    divergence[0:Ni_carrier + 1, 0:Nj_carrier+1] = 0
    divergence2[0:Ni_carrier + 1, 0:Nj_carrier+1] = 0

    return uz_next, ur_next, p_next, divs


def one_iteration_energy(uz, ur, rho, T, P, dt):

    T_new, _ = energy_s(uz, ur, rho, T, P, dt)
    temperature_boundaries(T_new)

    return T_new, rhof(T_new)


def magnetic_field_update(T, A_prev, N_MAGNETIC_FIELD_ITERATIONS=N_MAGNETIC_FIELD_ITERATIONS):
    """Returns Fr, Fz, P for all the interior cells."""
    T_regular = better_interpolator(R_original=R, Z_original=Z, R_new=R_regular, Z_new=Z_regular, Field_Original=T)
    T_regular = T_regular.T
    A_prev_regular = better_interpolator(R_original=R[1:-1, 1:-1], Z_original=Z[1:-1, 1:-1],
                                         R_new=R_regular[1:-1, 1:-1], Z_new=Z_regular[1:-1, 1:-1],
                                         Field_Original=A_prev)
    A_prev_regular = A_prev_regular.T
    A_regular, err = MagClass.magnetic_vector_solver(T_regular[1:-1, 1:-1], N_MAGNETIC_FIELD_ITERATIONS, A_prev_regular)
    Fz_regular, Fr_regular, P_regular = MagClass.FzFrP_overall(T_regular[1:-1, 1:-1], A_regular)

    Fz = better_interpolator(R_original=R_regular[1:-1, 1:-1], Z_original=Z_regular[1:-1, 1:-1],
                             R_new=R[1:-1, 1:-1], Z_new=Z[1:-1, 1:-1], Field_Original=Fz_regular.T)
    Fr = better_interpolator(R_original=R_regular[1:-1, 1:-1], Z_original=Z_regular[1:-1, 1:-1],
                             R_new=R[1:-1, 1:-1], Z_new=Z[1:-1, 1:-1], Field_Original=Fr_regular.T)
    P = better_interpolator(R_original=R_regular[1:-1, 1:-1], Z_original=Z_regular[1:-1, 1:-1],
                            R_new=R[1:-1, 1:-1], Z_new=Z[1:-1, 1:-1], Field_Original=P_regular.T)
    A = better_interpolator(R_original=R_regular[1:-1, 1:-1], Z_original=Z_regular[1:-1, 1:-1],
                            R_new=R[1:-1, 1:-1], Z_new=Z[1:-1, 1:-1], Field_Original=A_regular.T)

    return Fz, Fr, P, A


def one_iteration_simulation(uz_prev, ur_prev, p_prev, T_prev, rho_prev,
                             A_prev, N_MAGNETIC_FIELD_ITERATIONS=N_MAGNETIC_FIELD_ITERATIONS):
    dt = compute_dt(uz_prev, ur_prev, T_prev, p_prev, rho_prev)
    Fz, Fr, P, A = magnetic_field_update(T_prev, A_prev, N_MAGNETIC_FIELD_ITERATIONS)
    T_next, rho_next = one_iteration_energy(uz_prev, ur_prev, rho_prev, T_prev, P, dt)
    uz_next, ur_next, p_next, divs = one_iteration_flow(uz_prev, ur_prev, rho_prev, p_prev, T_prev, T_next, Fz, Fr, dt)
    # print()
    # print(
    #     "For velocity \n"
    #     f"The average change is {np.average(np.abs(uz_next-uz_prev))} \n "
    #     f"and the max change is {np.max(np.abs(uz_next-uz_prev))}"
    # )
    # print()
    # print(
    #     "For temperature \n"
    #     f"The average change is {np.average(np.abs(T_next - T_prev))} \n "
    #     f"and the max change is {np.max(np.abs(T_next - T_prev))}"
    # )
    # print()
    return (
        uz_next*alpha + (1-alpha)*uz_prev, ur_next*alpha + (1-alpha)*ur_prev,
        p_next, T_next*alpha_T + (1-alpha_T)*T_prev, rhof(T_next*alpha_T + (1-alpha_T)*T_prev), dt, Fz, Fr, P, A, divs
    )


def simulation(uz_prev=uz_prev, ur_prev=ur_prev, p_prev=p_prev, T_prev=T_prev, rho_prev=rho_prev):

    # Step one:
    global current_time, SHOW_FLOW

    particle = Particle(dpi=100*10**(-6), uzpi=0, urpi=0, ri=Lr_carrier/3, zi=0.04, Tpi=350)

    A_null = np.zeros_like(T_prev[1:-1, 1:-1])
    X = one_iteration_simulation(uz_prev, ur_prev, p_prev, T_prev, rho_prev, A_null, 20)
    uz_next, ur_next, p_next, T_next, rho_next, dt, Fz, Fr, P, A, divs = X
    current_time += dt

    for step in tqdm(range(Nt)):

        if step % 500 == 0 and SAVE:
            save_conditions(T_next, uz_next, ur_next,
                            f"[kW={kW},Q_in={Q_main}, "
                            f"Q_sheath={Q_sheath},Q_carrier={Q_carrier}]", file_number
                            )

        if step % 500 == 0 and SHOW_FLOW:

            X = one_iteration_simulation(uz_next, ur_next, p_next, T_next, rho_next, A)
            uz_next, ur_next, p_next, T_next, rho_next, dt, Fz, Fr, P, A, divs = X

            # if current_time > PARTICLE_INSERTION:
            #     particle.particle_movement(uz_next, ur_next, T_next, dt)

            variables = uz_next, ur_next, T_next, p_next, rho_next, P, A, Fr, Fz, divs, Z, R, current_time
            animation(variables, particle)

            current_time += dt

        else:

            X = one_iteration_simulation(uz_next, ur_next, p_next, T_next, rho_next, A)
            uz_next, ur_next, p_next, T_next, rho_next, dt, Fz, Fr, P, A, divs = X

            # if current_time > PARTICLE_INSERTION:
            #     particle.particle_movement(uz_next, ur_next, T_next, dt)

            current_time += dt

    plt.show()

    return uz_next, ur_next, p_next, T_next

