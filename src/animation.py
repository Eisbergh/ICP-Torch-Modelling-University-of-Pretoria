import matplotlib.pyplot as plt
import cmasher as cmr
import numpy as np
from all_variables import ASPECT_RATIO, Lz, Lr, Patm, SHOW_FLOW, VERTICAL_DISTANCE, DPI
from base_methods import divergence_s, better_interpolator
from parameters import muvf

Ni_animation = 40
Nj_animation = 20

dz_ani = Lz / Ni_animation
dr_ani = Lr / Nj_animation

z_coordinates_regular = np.linspace(-dz_ani/2, Lz+dz_ani/2, Ni_animation+2)
r_coordinates_regular = np.linspace(-dr_ani/2, Lr+dr_ani/2, Nj_animation+2)
Z_reg_ani, R_reg_ani = np.meshgrid(z_coordinates_regular, r_coordinates_regular, indexing="ij")


def mass_balance(rho, uz, R_reg):
    index = 1
    mass_in = 2*np.pi*np.sum(R_reg[index, 1:-1]*rho[index, 1:-1]*uz[index, 1:-1]*(R_reg[index, 2:]-R_reg[index, 1:-1]))
    mass_out = 2*np.pi*np.sum(R_reg[-1, 1:-1]*rho[-1, 1:-1]*uz[-1, 1:-1]*(R_reg[index, 2:]-R_reg[index, 1:-1]))

    return mass_in-mass_out, mass_in, mass_out


if SHOW_FLOW:
    plt.figure(figsize=(1.7 * ASPECT_RATIO, VERTICAL_DISTANCE), dpi=DPI, constrained_layout=True)


def animation(variables, particles):

    NUMBER_OF_PLOTS = 3
    Subplot = 0

    uz_next, ur_next, T_next, p_next, rho_next, P, A, Fr, Fz, divs, Z, R, current_time = variables

    uz_vertex_centered = (
            (
                    uz_next[1:, 1:-1]
                    +
                    uz_next[:-1, 1:-1]
            ) / 2
    )
    ur_vertex_centered = (
            (
                    ur_next[1:-1, 1:]
                    +
                    ur_next[1:-1, :-1]
            ) / 2
    )

    uz_vertex_centered1 = better_interpolator(R[1:-1, 1:-1], Z[1:-1, 1:-1],
                                              uz_vertex_centered, R_reg_ani[1:-1, 1:-1], Z_reg_ani[1:-1, 1:-1])
    ur_vertex_centered1 = better_interpolator(R[1:-1, 1:-1], Z[1:-1, 1:-1],
                                              ur_vertex_centered, R_reg_ani[1:-1, 1:-1], Z_reg_ani[1:-1, 1:-1])

    T_next1 = better_interpolator(R, Z, T_next, R_reg_ani, Z_reg_ani)
    # rho_next = better_interpolator(R, Z, rho_next, R_reg_ani, Z_reg_ani)
    divs = better_interpolator(R[1:-1, 1:-1], Z[1:-1, 1:-1], divs,
                               R_reg_ani[1:-1, 1:-1], Z_reg_ani[1:-1, 1:-1])
    # Fr = better_interpolator(R[1:-1, 1:-1], Z[1:-1, 1:-1], Fr, R_reg_ani[1:-1, 1:-1], Z_reg_ani[1:-1, 1:-1])
    # P = better_interpolator(R[1:-1, 1:-1], Z[1:-1, 1:-1], P, R_reg_ani[1:-1, 1:-1], Z_reg_ani[1:-1, 1:-1])

    # ================================================================ #
    # --------------------- Velocity field plot ---------------------- #
    # ================================================================ #
    Subplot += 1
    plt.subplot(NUMBER_OF_PLOTS, 1, Subplot)
    plt.contourf(
        Z[1:-1, 1:-1],
        R[1:-1, 1:-1],
        uz_vertex_centered,
        levels=8,
        cmap=cmr.viola,
        vmin=int(np.min(uz_vertex_centered) - 1),
        vmax=int(np.max(uz_vertex_centered) + 1),
    )
    plt.colorbar(label="$v_z$ [m/s]")
    plt.quiver(
        Z[1:-1, 1:-1][::2],
        R[1:-1, 1:-1][::2],
        uz_vertex_centered[::, ::][::2],
        ur_vertex_centered[::, ::][::2],
        alpha=0.4,
        color="black",
        width=0.001
    )
    # plt.title(f"Velocity field at time {current_time * 10 ** 6:.1f} microseconds.", fontsize=10)
    plt.title(f"Velocity field.", fontsize=10)
    plt.xlabel("Axial Position $z$ [m]", fontsize=10)
    plt.ylabel("Radial Position $r$ [m]", fontsize=10)
    plt.axhline(y=0, color='white', linestyle='--', linewidth=0.8, alpha=0.6)
    plt.tick_params(labelsize=10)
    plt.xlim(0, 0.2)
    plt.ylim(0, 0.025)
    for y in [0.0188, -0.0188]:
        plt.hlines(y, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    plt.hlines(-0.003, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    plt.hlines(0.003, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    # Add 3 vertical lines at x = 0.05 with different y-spans
    plt.vlines(x=0.05, ymin=-0.025, ymax=-0.0188, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    plt.vlines(x=0.05, ymin=-0.003, ymax=0.003, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    plt.vlines(x=0.05, ymin=0.0188, ymax=0.025, color='black', linestyle='--', linewidth=1.0, alpha=0.8)

    # # ================================================================ #
    # -------------------- Extra field plot -------------------- #
    # ================================================================ #
    Subplot += 1
    plt.subplot(NUMBER_OF_PLOTS, 1, Subplot)
    speed = np.sqrt(uz_vertex_centered1 ** 2 + ur_vertex_centered1 ** 2)

    print("Maximum velocity.")
    print(np.max(speed))

    stream = plt.streamplot(
        Z_reg_ani[1:-1, 1:-1].T,
        R_reg_ani[1:-1, 1:-1].T,
        uz_vertex_centered1.T,
        ur_vertex_centered1.T,
        color=speed.T,
        cmap='plasma',
        density=1,
        linewidth=0.5  # 2 * speed.T / np.max(speed)

    )
    plt.scatter(particles.z, particles.r, s=10, c='red')
    plt.colorbar(stream.lines, label="Velocity Magnitude [m/s]")
    plt.title("Velocity Stream Plot")
    plt.xlabel("Axial Position $z$ [m]")
    plt.ylabel("Radial Position $r$ [m]")
    plt.grid(True)
    plt.ylim(0, Lr)
    plt.xlim(0, Lz)
    line_width = 1.5
    # Add 3 vertical lines at x = 0.05 with different y-spans
    plt.vlines(x=0.05, ymin=-0.025, ymax=-0.0188, color='black', linestyle='--', linewidth=line_width, alpha=0.8)
    plt.vlines(x=0.05, ymin=-0.003, ymax=0.003, color='black', linestyle='--', linewidth=line_width, alpha=0.8)
    plt.vlines(x=0.05, ymin=0.0188, ymax=0.025, color='black', linestyle='--', linewidth=line_width, alpha=0.8)
    for y in [0.0188, -0.0188]:
        plt.hlines(y, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=line_width, alpha=0.8)
    plt.hlines(-0.003, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=line_width, alpha=0.8)
    plt.hlines(0.003, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=line_width, alpha=0.8)
    #
    # ================================================================ #
    # -------------------- Temperature field plot -------------------- #
    # ================================================================ #
    print("Maximum temperature")
    print(np.max(T_next1))

    Subplot += 1

    plt.subplot(NUMBER_OF_PLOTS, 1, Subplot)
    plt.contourf(
        Z_reg_ani,
        R_reg_ani,
        T_next1,
        levels=10,
        cmap=cmr.redshift,
        vmin=int(np.min(T_next1[1:-1, 1:-1]) - 100),
        vmax=int(np.max(T_next1[1:-1, 1:-1]) + 100)
    )
    plt.colorbar(label="$T$ [K]")
    plt.title("Temperature Field")
    plt.xlabel("Axial Position $z$ [m]", fontsize=10)
    plt.ylabel("Radial Position $r$ [m]", fontsize=10)
    plt.axhline(y=0, color='white', linestyle='--', linewidth=0.8, alpha=0.6)
    plt.tick_params(labelsize=10)
    plt.xlim(0, Lz)
    plt.ylim(0, Lr)
    for y in [0.0188, -0.0188]:
        plt.hlines(y, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=line_width, alpha=0.8)
    plt.hlines(-0.003, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=line_width, alpha=0.8)
    plt.hlines(0.003, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=line_width, alpha=0.8)
    # Add 3 vertical lines at x = 0.05 with different y-spans
    plt.vlines(x=0.05, ymin=-0.025, ymax=-0.0188, color='black', linestyle='--', linewidth=line_width, alpha=0.8)
    plt.vlines(x=0.05, ymin=-0.003, ymax=0.003, color='black', linestyle='--', linewidth=line_width, alpha=0.8)
    plt.vlines(x=0.05, ymin=0.0188, ymax=0.025, color='black', linestyle='--', linewidth=line_width, alpha=0.8)

    # ======================================================================================================== #

    # # --- 4. field ---
    # Subplot += 1
    # plt.subplot(NUMBER_OF_PLOTS, 1, Subplot)
    # plt.axhline(y=0, color='white', linestyle='--', linewidth=0.8, alpha=0.6)
    # plt.contourf(
    #     Z,
    #     R,
    #     p_next,
    #     levels=10,
    #     cmap=cmr.redshift,
    #     vmin=int(np.min(p_next)),
    #     vmax=int(np.max(p_next))
    # )
    # plt.colorbar(label="Pressure [kPa]")
    # plt.xlim(0, Lz)
    # plt.ylim(0, Lr)
    # plt.title(f"Pressure")
    # plt.xlabel("Axial Position $z$ [m]")
    # plt.ylabel("Radial Position $r$ [m]")

    plt.savefig("flow_fields.png", dpi=2000)
    plt.draw()
    plt.pause(0.000000005)
    plt.clf()

    print(f"The net mass flow is {mass_balance(rho_next, uz_next, R)}.")
    print(f"Current time is {current_time * 10 ** 6:.1f} microseconds")
    print(f"The average Reynolds number is: {np.average(np.abs(uz_next*rho_next[1:]*Lr/muvf(T_next[1:])))}.")

    return


if SHOW_FLOW and False:
    fig, axes = plt.subplots(1, 3, figsize=(8, 5.6), dpi=60, sharey=True)


def animation_vertical(variables, particles):

    NUMBER_OF_PLOTS = 3
    Subplot = 0

    uz_next, ur_next, T_next, p_next, rho_next, P, A, Fr, Fz, divs, Z, R, current_time = variables

    uz_vertex_centered = (
            (
                    uz_next[1:, 1:-1]
                    +
                    uz_next[:-1, 1:-1]
            ) / 2
    )
    ur_vertex_centered = (
            (
                    ur_next[1:-1, 1:]
                    +
                    ur_next[1:-1, :-1]
            ) / 2
    )

    uz_vertex_centered1 = better_interpolator(R[1:-1, 1:-1], Z[1:-1, 1:-1],
                                              uz_vertex_centered, R_reg_ani[1:-1, 1:-1], Z_reg_ani[1:-1, 1:-1])
    ur_vertex_centered1 = better_interpolator(R[1:-1, 1:-1], Z[1:-1, 1:-1],
                                              ur_vertex_centered, R_reg_ani[1:-1, 1:-1], Z_reg_ani[1:-1, 1:-1])

    T_next1 = better_interpolator(R, Z, T_next, R_reg_ani, Z_reg_ani)
    # rho_next = better_interpolator(R, Z, rho_next, R_reg_ani, Z_reg_ani)
    divs = better_interpolator(R[1:-1, 1:-1], Z[1:-1, 1:-1], divs,
                               R_reg_ani[1:-1, 1:-1], Z_reg_ani[1:-1, 1:-1])
    # Fr = better_interpolator(R[1:-1, 1:-1], Z[1:-1, 1:-1], Fr, R_reg_ani[1:-1, 1:-1], Z_reg_ani[1:-1, 1:-1])
    # P = better_interpolator(R[1:-1, 1:-1], Z[1:-1, 1:-1], P, R_reg_ani[1:-1, 1:-1], Z_reg_ani[1:-1, 1:-1])

    # ================================================================ #
    # -------------------- H O R I Z O N T A L  L A Y O U T ---------- #
    # ================================================================ #

    ax_vel, ax_stream, ax_temp = axes

    # ---------- (1) Velocity field (filled contours + quiver) ----------
    h1 = ax_vel.contourf(
        R[1:-1, 1:-1],  # x = r
        Z[1:-1, 1:-1],  # y = z
        uz_vertex_centered,
        levels=8, cmap=cmr.viola,
        vmin=int(np.min(uz_vertex_centered) - 1),
        vmax=int(np.max(uz_vertex_centered) + 1),
    )
    fig.colorbar(h1, ax=ax_vel, label="$v_z$ [m/s]")
    ax_vel.quiver(
        R[1:-1, 1:-1][::],
        Z[1:-1, 1:-1][::],
        ur_vertex_centered[::, ::],  # U along x
        uz_vertex_centered[::, ::],  # V along y
        alpha=0.4, color="black", width=0.001
    )
    ax_vel.set_title("Velocity field", fontsize=10)
    ax_vel.set_xlabel("Radial position $r$ [m]")
    ax_vel.set_ylabel("Axial position $z$ [m]")
    ax_vel.set_xlim(0, Lr);
    ax_vel.set_ylim(0, Lz)

    # guides
    for x in [0.0188, 0.003]:
        ax_vel.axvline(x=x, ymin=0, ymax=0.25, ls='--', lw=1.0, c='k', alpha=0.8)
    ax_vel.axhline(y=0.05, xmin=0, xmax=3/25, ls='--', lw=1.0, c='k', alpha=0.8)
    ax_vel.axhline(y=0.05, xmin=18.8/25, xmax=1, ls='--', lw=1.0, c='k', alpha=0.8)

    # ---------- (2) Streamlines (speed-coloured) ----------
    speed = np.sqrt(uz_vertex_centered1 ** 2 + ur_vertex_centered1 ** 2)
    strm = ax_stream.streamplot(
        R_reg_ani[1:-1, 1:-1],  # x = r
        Z_reg_ani[1:-1, 1:-1],  # y = z
        ur_vertex_centered1,  # U along x
        uz_vertex_centered1,  # V along y
        color=speed, cmap='plasma', density=1.0, linewidth=0.5
    )
    fig.colorbar(strm.lines, ax=ax_stream, label="Speed [m/s]")
    ax_stream.scatter(particles.r, particles.z, s=10, c='red')
    ax_stream.set_title("Velocity stream plot")
    ax_stream.set_xlabel("Radial position $r$ [m]")
    ax_stream.set_xlim(0, Lr);
    ax_stream.set_ylim(0, Lz);
    ax_stream.grid(True)
    # guides
    for x in [0.0188, 0.003]:
        ax_stream.axvline(x=x, ymin=0, ymax=0.25, ls='--', lw=1.0, c='k', alpha=0.8)
    ax_stream.axhline(y=0.05, xmin=0, xmax=3/25, ls='--', lw=1.0, c='k', alpha=0.8)
    ax_stream.axhline(y=0.05, xmin=18.8/25, xmax=1, ls='--', lw=1.0, c='k', alpha=0.8)

    # ---------- (3) Temperature field ----------
    h3 = ax_temp.contourf(
        R_reg_ani, Z_reg_ani, T_next1,
        levels=10, cmap=cmr.redshift,
        vmin=int(np.min(T_next1[1:-1, 1:-1]) - 100),
        vmax=int(np.max(T_next1[1:-1, 1:-1]) + 100)
    )
    fig.colorbar(h3, ax=ax_temp, label="$T$ [K]")
    ax_temp.set_title("Temperature field")
    ax_temp.set_xlabel("Radial position $r$ [m]")
    ax_temp.set_xlim(0, Lr);
    ax_temp.set_ylim(0, Lz)
    # guides
    for x in [0.0188, 0.003]:
        ax_temp.axvline(x=x, ymin=0, ymax=0.25, ls='--', lw=1.0, c='k', alpha=0.8)
    ax_temp.axhline(y=0.05, xmin=0, xmax=3/25, ls='--', lw=1.0, c='k', alpha=0.8)
    ax_temp.axhline(y=0.05, xmin=18.8/25, xmax=1, ls='--', lw=1.0, c='k', alpha=0.8)

    fig.tight_layout()
    fig.savefig("flow_fields.png", dpi=2000)

    plt.draw()
    plt.pause(0.000000005)
    plt.clf()

    print(f"The net mass flow is {mass_balance(rho_next, uz_next, R)}.")
    print(f"Current time is {current_time * 10 ** 6:.1f} microseconds")
    print(f"The average Reynolds number is: {np.average(np.abs(uz_next*rho_next[1:]*Lr/muvf(T_next[1:])))}.")

    return


