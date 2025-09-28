from particle_parameters import *
from all_variables import (
    muvf, rhof, kf, Lz_carrier, Lz_sheath, Lr_carrier, Lr_sheath, Lr, Lz, ASPECT_RATIO, DPI, R_regular, Z_regular
)
from base_methods import interpolator, extract_conditions
import matplotlib.pyplot as plt
import cmasher as cmr


"""
I need to solve for a single particle inserted into the plasma torch.
This can be done by solving the Basset–Boussinesq–Oseen equation.
This equation is given in Nam et al.

I have a temperature field and a velocity field for the plasma in the torch.
I am going to assume that this remains fixed when the particle is inserted.  
So it is like the particle is flowing on top of a pre-determined velocity field.

"""


PARTICLE_INSERTION = 0.0002   # seconds


class Particle:

    def __init__(self, dpi, uzpi, urpi, ri, zi, Tpi, g=9.81):
        self.dpi = dpi
        self.dp = dpi  # will store the instantaneous particle diameter.
        self.uzp = uzpi
        self.urp = urpi
        self.r = ri
        self.z = zi
        self.Tp = Tpi
        self.g = g
        self.x = 0  # amount of liquid.
        self.dp_cutoff = 1*10**(-9)
        self.dp_history = [dpi]
        self.uzp_history = [uzpi]
        self.urp_history = [urpi]
        self.r_history = [ri]
        self.z_history = [zi]
        self.Tp_history = [Tpi]
        self.x_history = [0]
        self.t = PARTICLE_INSERTION
        self.t_history = [PARTICLE_INSERTION]
        self.end = False
        self.rhop_solid = 4500
        self.rhop_liquid = 4200
        self.mp = np.pi*dpi**3/6*self.rhop_solid
        self.spheroid = 0
        self.spheroid_history = [0]
        self.exit_inlet = False

    def Up(self, uz, ur):
        return np.sqrt((self.uzp-uz)**2+(self.urp-ur)**2)

    def emissivity(self, T):
        return 0.45

    def rhopf(self):
        if self.Tp < Tmp:
            return self.rhop_solid
        else:
            return self.rhop_liquid

    def Repf(self, rho, muv, uz, ur):
        """Use point values to find the Reynolds number of a single particle"""
        return self.dp * self.Up(uz, ur) * rho / muv

    def Cdf(self, rho, muv, uz, ur):
        """Returns drag coefficient for a given Reynolds number of the particle."""
        Rep = self.Repf(rho, muv, uz, ur)
        if Rep <= 0.2:
            return 24 / Rep
        elif Rep <= 2:
            return 24 / Rep * (1 + 3 / 16 * Rep)
        elif Rep <= 21:
            return 24 / Rep * (1 + 0.11 * Rep ** 0.81)
        else:
            return 24 / Rep * (1 + 0.189 * Rep ** 0.62)

    def Nup(self, rho, muv, uz, ur):
        """Nusselt number of the particle."""
        return 2 + 0.515 * np.sqrt(self.Repf(rho, muv, uz, ur))

    def hc(self, rho, muv, uz, ur, k):
        """Heat transfer coefficient for convection."""
        return self.Nup(rho, muv, uz, ur) * k / self.dp

    def Cpp(self):
        """Returns the Cp value for the particle at a specific temperature, whether liquid or metal. J / kg K"""

        Tp = self.Tp / 1000
        index = np.round(np.arctan(self.Tp - 700) / (np.pi) + 0.5)
        index2 = np.round(np.arctan(self.Tp - Tmp + 0.0001) / (np.pi) + 0.5)

        Solid = (
                        (1 - index) * 22.61942 + (1 - index) * 18.98795 * Tp +
                        (1 - index) * (-18.18735) * Tp ** 2 + (1 - index) * 7.080792 * Tp ** 3 + (1 - index) * (
                            -0.143457) / Tp ** 2
                        +
                        index * 44.37174 + index * (-44.09225) * Tp +
                        index * 31.70602 * Tp ** 2 + index * 0.0052209 * Tp ** 3 + index * 0.036168 / Tp ** 2
                ) * 1000 / 47.867

        Liquid = 33.51 * 1000 / 47.867

        return Solid * (1 - index2) + Liquid * index2

    def reflect_vertical(self, normal=np.array([1, 0]), e_n=0.99, friction=0.99):
        """
        :param normal: (0, 1)
        :param e_n: normal restitution coefficient
        :param friction: friction effect on the wall
        :return:
        """
        v = np.array([self.urp, self.uzp])
        n = np.array(normal)
        t = np.array([-n[1], n[0]])

        v_n = np.dot(v, n) * n
        v_t = np.dot(v, t) * t

        v_n_new = -e_n*v_n
        v_t_new = friction * v_t

        v_new = v_n_new+v_t_new

        return v_new[0], v_new[1]

    def reflect_horizontal(self, normal=np.array([0, 1]), e_n=0.99, friction=1):
        """
        :param normal: (0, 1)
        :param e_n: normal restitution coefficient
        :param friction: friction effect on the wall
        :return:
        """
        return -self.urp*friction, -self.uzp*e_n

    def collision_checker(self, dt):
        """Assume perfectly elastic collisions"""
        z = self.z + dt * self.uzp
        r = self.r + dt * self.urp
        if z <= 0:
            self.urp, self.uzp = 0, 0
            self.exit_inlet = True
        elif z <= Lz_sheath and r >= Lr_sheath:
            """Collides with the wall.  Assume z velocity stays the same and r velocity flips. """
            self.urp, self.uzp = self.reflect_vertical()
        elif z <= Lz_carrier < self.z and r <= Lr_carrier:
            self.urp, self.uzp = self.reflect_horizontal()
        # elif z <= Lz_carrier and r <= Lr_carrier:
        #     self.urp, self.uzp = self.reflect_horizontal()
        elif z <= Lz_carrier and r <= Lr_carrier:
            self.urp, self.uzp = self.reflect_vertical()
        elif r <= 0:
            """Just goes on normally due to reflection."""
            self.urp = -self.urp
        elif r >= Lr:
            """Collides with top wall.  Assume z velocity stays the same and r velocity flips."""
            self.urp, self.uzp = self.reflect_vertical()
        elif z >= Lz:
            self.end = True

        return

    def update_velocity(self, uz, ur, dt, muv, rho, T):

        # buoyancy-corrected gravity
        g_eff = self.g * (1.0 - rho / self.rhopf())

        Cd = self.Cdf(rho, muv, uz, ur)
        self.uzp = (
            self.uzp + dt*(
                -3/4*Cd*(self.uzp-uz)*self.Up(uz, ur)*rho/self.rhopf()/self.dp + g_eff
            )
        )
        self.urp = (
            self.urp + dt*(
                -3/4*Cd*(self.urp-ur)*self.Up(uz, ur)*rho/self.rhopf()/self.dp
            )
        )
        self.collision_checker(dt)    # Ensures urp is in the correct direction.
        self.uzp_history.append(self.uzp)
        self.urp_history.append(self.urp)
        return

    def update_position(self, dt):
        if self.dp <= self.dp_cutoff:
            self.r_history.append(self.r)
            self.z_history.append(self.z)
            return
        elif self.exit_inlet:
            self.spheroid = 0
            return
        self.z = self.z + dt*self.uzp
        self.r = self.r + dt*self.urp
        self.r_history.append(self.r)
        self.z_history.append(self.z)
        return

    def energy_balance(self, T, Text, Tmp, Tbp, Hm, Hb, rho, muv, uz, ur, k, dt):
        hc = self.hc(rho, muv, uz, ur, k)
        convection = np.pi*self.dp**2*hc*(T-self.Tp)  ## Deel met vier?
        radiation = np.pi*self.dp**2*sigma_s*self.emissivity(self.Tp)*(self.Tp**4-Text**4)

        # for the radiation part I think they just do it incorrectly.  It is assumed that the plasma is optically thin

        if self.Tp < Tmp:    # Remember that the diameter changes.
            divider = self.rhopf()*np.pi/6*self.dp**3*self.Cpp()
            T_new = (
                self.Tp + dt*(
                    convection/divider - radiation/divider
                )
            )
            if T_new >= Tmp:
                self.Tp = Tmp
            else:
                self.Tp = T_new

        elif self.Tp == Tmp:
            divider = self.mp*Hm
            x_new = (
                self.x + dt*(
                    convection/divider - radiation/divider
                )
            )
            if x_new >= 1:
                self.spheroid = 1
                self.x = 1
                self.Tp += 0.1
            elif x_new <= 0:
                self.x = 0
                self.Tp -= 0.1
            else:
                self.x = x_new

            Vp = self.mp*((1-self.x)/self.rhop_solid+self.x/self.rhop_liquid)
            self.dp = (6/np.pi*Vp)**(1/3)

        elif self.Tp == Tbp and self.Tp <= T:
            divider = -self.rhopf()*np.pi/2*self.dp**2*Hb
            dp_new = (
                self.dp + dt*(
                    convection/divider - radiation/divider
                )
            )
            if dp_new <= self.dp_cutoff:
                self.dp = self.dp_cutoff

            elif dp_new <= self.dp:
                self.dp = dp_new

            self.mp = self.rhop_liquid*np.pi/6*self.dp**3

        elif Tmp <= self.Tp <= Tbp:
            divider = self.rhopf()*np.pi/6*self.dp**3*self.Cpp()
            T_new = (
                self.Tp + dt*(
                    convection/divider - radiation/divider
                )
            )
            if T_new >= Tbp:
                self.Tp = Tbp
            elif T_new <= Tmp:
                self.Tp = Tmp
            else:
                self.Tp = T_new

        return

    def particle_movement(self, uz, ur, T, dt, R, Z):

        if self.end:
            self.energy_balance(350, 350, Tmp, Tbp, Hm, Hb, rhof(350), muvf(350), 0, 0, kf(350), dt)
            self.Tp_history[-1] = self.Tp
            self.x_history[-1] = self.x
            self.dp_history[-1] = self.dp

            return

        uz_grid = (
                uz[1:-1, 1:-1]
        )
        ur_grid = (
                ur[1:-1, 1:-1]
        )
        T_grid = T[1:-1, 1:-1]
        T = interpolator(self.z, self.r, T_grid, R[1:-1, 1:-1], Z[1:-1, 1:-1])
        uz = interpolator(self.z, self.r, uz_grid, R[1:-1, 1:-1], Z[1:-1, 1:-1])
        ur = interpolator(self.z, self.r, ur_grid, R[1:-1, 1:-1], Z[1:-1, 1:-1])
        self.t = self.t+dt
        self.t_history.append(self.t)
        self.spheroid_history.append(self.spheroid)
        self.update_position(dt)
        self.update_velocity(uz, ur, dt, muvf(T), rhof(T), T)
        self.energy_balance(T, 350, Tmp, Tbp, Hm, Hb, rhof(T), muvf(T), uz, ur, kf(T), dt)
        self.Tp_history.append(self.Tp)
        self.x_history.append(self.x)
        self.dp_history.append(self.dp)

        return


def simulate_particle(dpi, uzpi, urpi, ri, zi, Tpi, R, Z):
    particle = Particle(dpi=dpi, uzpi=uzpi, urpi=urpi, ri=ri, zi=zi, Tpi=Tpi)

    name = "[kW=3,Q_in=3, Q_sheath=31,Q_carrier=1]"
    T, uz, ur = extract_conditions(R=R, Z=Z, Rur=R, Zur=Z, Ruz=R, Zuz=Z, name=name, num=4)

    dt = 0.0002
    time = 0

    plt.figure(figsize=(1.3 * ASPECT_RATIO, 2), dpi=100)

    for _ in range(10000):
        time += dt
        particle.particle_movement(uz=uz, ur=ur, T=T, dt=dt, R=R_regular, Z=Z_regular)

        # plt.scatter(particle.z, particle.r, s=10, c='red')
        # plt.title(
        #     (f"Particle uz = {particle.uzp:.3f}, ur={particle.urp:.3f}, T={particle.Tp:.1f}, "
        #      f"dp={particle.dp*10**6:.1f}, x={particle.x:.2f}, r={particle.r:.3f}"),
        #     fontsize=8)
        # plt.xlim(0, 0.2)
        # plt.ylim(0, 0.025)
        # plt.tight_layout()
        # plt.draw()
        # plt.pause(0.000000005)
        # plt.clf()

    return particle, T, uz, ur


if __name__ == "__main__":

    from particle_viz import (
        history_from_particle, plot_trajectory_over_fields, plot_summary_panels, overlay_summary_panels_for_sizes,
        plot_trajectory_over_fields3, get_final_values, plot_dp_change_vs_initial
    )
    #
    # particle, T, uz, ur = simulate_particle(
    #     dpi=80e-6, uzpi=1.8, urpi=0.01, ri=3.7 / 1000 / 3, zi=0.051, Tpi=350, R=R_regular, Z=Z_regular
    # )
    #
    # hist = history_from_particle(particle)

    # # 1) Trajectory over the fields (color by time)
    # plot_trajectory_over_fields(
    #     hist=hist, R=R_regular, Z=Z_regular, T=T, uz_field=uz, ur_field=ur,
    #     color_by="time", streamline_density=1.8, figscale=2.0, dpi=60,
    #     savepath="traj_over_fields_time.png",
    #     title="Trajectory over T and streamlines (colored by time)"
    # )
    #
    # # 2) Trajectory over the fields (color by particle temperature)
    # plot_trajectory_over_fields(
    #     hist=hist, R=R_regular, Z=Z_regular, T=T, uz_field=uz, ur_field=ur,
    #     color_by="Tp", streamline_density=1.2, figscale=2.0, dpi=60,
    #     savepath="traj_over_fields_Tp.png",
    #     title="Trajectory over T and streamlines (colored by $T_p$)"
    # )
    #
    # 3) Summary panels (z, r, velocities, T_p & d_p)
    # If you have Tmp and Tbp in scope, pass them for reference lines.
    # plot_summary_panels(
    #     hist=hist, Tmp=Tmp, Tbp=Tbp, dpi=60, savepath="summary_panels.png",
    #     title="Single-particle time histories"
    # )

    # plt.show()
    #
    # overlay_summary_panels_for_sizes(
    #     simulate_particle, R_regular, Z_regular, Tmp, Tbp, microns=[20, 40, 60, 80, 100, 1000],
    #     outdir="figs_overlays",
    #     dpi=400
    # )

    particle, T, uz, ur = simulate_particle(
        dpi=100e-6, uzpi=1.8, urpi=0, ri=3.7/1000/3, zi=0.051, Tpi=350, R=R_regular, Z=Z_regular
    )
    hist40 = history_from_particle(particle)

    particle, T, uz, ur = simulate_particle(
        dpi=200e-6, uzpi=1.8, urpi=0.1, ri=3.7/1000/3, zi=0.051, Tpi=350, R=R_regular, Z=Z_regular
    )
    hist60 = history_from_particle(particle)

    particle, T, uz, ur = simulate_particle(
        dpi=300e-6, uzpi=1.8, urpi=0, ri=3.7/1000/3, zi=0.051, Tpi=350, R=R_regular, Z=Z_regular
    )
    hist80 = history_from_particle(particle)

    particle, T, uz, ur = simulate_particle(
        dpi=400-6, uzpi=1.8, urpi=0, ri=3.7/1000/3, zi=0.051, Tpi=350, R=R_regular, Z=Z_regular
    )
    hist100 = history_from_particle(particle)

    particle, T, uz, ur = simulate_particle(
        dpi=500e-6, uzpi=1.8, urpi=0, ri=3.7/1000/3, zi=0.051, Tpi=350, R=R_regular, Z=Z_regular
    )
    hist1000 = history_from_particle(particle)

    # Multiple trajectories, shared colorbar, custom labels
    plot_trajectory_over_fields3(
        [hist40, hist60, hist80, hist100, hist1000],
        R_regular, Z_regular, T=T, uz_field=uz, ur_field=ur,
        color_by="time",
        labels=["100 μm", "200 μm", "300 μm", "400 μm", "500 μm"],
        savepath="Hallo"
    )

    values = np.linspace(10, 700, 50)/1000000
    extra = np.linspace(20, 700, 50)*0

    histories = []

    for val, ext in zip(values, extra):
        particle, T, uz, ur = simulate_particle(
            dpi=val, uzpi=1.8, urpi=ext, ri=3.7/1000/3, zi=0.051, Tpi=350, R=R_regular, Z=Z_regular
        )
        hist = history_from_particle(particle)
        histories.append(hist)

    plot_dp_change_vs_initial(histories)



