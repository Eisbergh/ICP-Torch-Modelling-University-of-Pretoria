from single_particle_solver import Particle
from simulation_file import simulation
from all_variables import *


class MultiParticle(Particle):

    def __init__(self, mass, dpi, uzpi, urpi, ri, zi, Tpi, g=9.81, Lr=Lr, Lz=Lz, ni=30, nj=20):
        super().__init__(dpi, uzpi, urpi, ri, zi, Tpi, g=g)
        self.mass = mass
        self.z_edges = np.linspace(0, Lz, ni+1)
        self.r_edges = np.linspace(0, Lr, nj+1)
        self.residence_time = np.zeros((ni, nj))
        self.concentration = np.zeros((ni, nj))

    def volumes(self):
        Z, R = np.meshgrid(self.z_edges, self.r_edges, indexing="ij")
        return (Z[1:, 1:]-Z[:-1, 1:])*np.pi*(R[1:, 1:]**2-R[:-1, :-1]**2)

    def trajectory_lk(self, ri, dpi, N0lk, uz, ur, T, dt):
        # l is the particle size group and ri is the particle insertion radial position
        # Works out what the residence time is for the particles in each of the cells.
        # Can only do it for one particle size and one projection.
        self.r = ri
        self.dpi = dpi
        for _ in range(int(2000)):
            self.particle_movement(uz, ur, T, dt)
            z_idx = np.searchsorted(self.z_edges, self.z)-1
            r_idx = np.searchsorted(self.r_edges, self.r)-1
            if self.z > self.z_edges[-1] or self.r > self.r_edges[-1]:
                break
            self.residence_time[z_idx, r_idx] += dt
        self.concentration = N0lk*self.residence_time/self.volumes()
        return


if __name__ == "__main__":

    part = MultiParticle(mass=10, dpi=60*10**(-6), uzpi=0, urpi=0, ri=3.7/1000/3, zi=0.04, Tpi=350)
    print("Hello")
    print(np.shape(part.volumes()))
    dt = 0.0001
    part.trajectory_lk(3.7/1000/3, 60*10**(-6), 10000, uz=np.load('uz_3.npy'), ur=np.load('ur_3.npy'), T=np.load('T_3.npy'), dt=dt)
    print(part.residence_time)
