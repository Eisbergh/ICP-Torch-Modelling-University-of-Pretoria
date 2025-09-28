import scipy as sc
import numpy as np
from all_variables import mu0, omega, Ic, Coils
from parameters import sigmaf
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
# So you have a boundary condition that you can use.  This boundary condition is given by Bernardi et al.
# Everywhere in the interior of the boundary you will find that the differential equation holds.


class ElectroMagnetic:

    def __init__(self, dr, dz, Lr, Lz, Ni, Nj, Coils=Coils, omega=omega, mu0=mu0, Ic=Ic, sigmaf=sigmaf):
        self.dr = dr
        self.dz = dz
        self.Ni = Ni
        self.Nj = Nj
        self.Coils = Coils
        self.omega = omega
        self.mu0 = mu0
        self.Ic = Ic
        self.sigmaf = sigmaf
        self.r = np.linspace(0, Lr, Ni)
        self.z = np.linspace(0, Lz, Nj)
        self.R, self.Z = np.meshgrid(self.r, self.z, indexing="ij")
        self.Coils = Coils[:, ::-1]

    def k1(self, index):
        return 1 / 2 / self.R[index[0], index[1]] / self.dr + 1 / self.dr ** 2

    def k2(self, index):
        return - 1 / 2 / self.dr / self.R[index[0], index[1]] + 1 / self.dr ** 2

    def k3(self, index):
        return 1 / self.dz ** 2

    def k4(self, index):
        return 1 / self.dz ** 2

    def k5(self, index, sigma):  # Just watch out for sigma
        return (
                - 2 / self.dr ** 2 - 2 / self.dz ** 2 -
                1 / self.R[index[0]][index[1]] ** 2 - 1j * self.mu0 * self.omega
                * sigma[index[0], index[1]]
        )

    @staticmethod
    def Gf(m):
        return ((2 - m ** 2) * sc.special.ellipk(m ** 2) - 2 * sc.special.ellipe(m ** 2)) / m

    @staticmethod
    def kjf(rj, Rb, zj, Zb):
        return np.sqrt(4 * Rb * rj / ((rj + Rb) ** 2 + (Zb - zj) ** 2))

    @staticmethod
    def kif(Ri, Rb, Zi, Zb):
        return np.sqrt(4 * Rb * Ri / ((Ri + Rb) ** 2 + (Zb - Zi) ** 2))

    def matrix_setup(self, sigma):

        """
        Only need to run it once to set up this matrix.  But if temp changes need to run it again. Then only update the
        inside.
        :return:
        """

        mat = np.zeros([self.Ni * self.Nj, self.Nj * self.Ni], dtype=np.complex128)

        for i in range(self.Ni * self.Nj):
            mat[i, i] = 1

        for i in range(self.Nj + 1, self.Nj * self.Ni - self.Nj):
            if i % self.Nj != 0 and i % self.Nj != self.Nj-1:
                index = [int(np.floor((i + 1) / self.Nj)), (i + 1) % self.Nj - 1]
                mat[i, i] = self.k5(index, sigma)
                mat[i, i - 1] = self.k4(index)
                mat[i, i + 1] = self.k3(index)
                mat[i, i - self.Nj] = self.k2(index)
                mat[i, i + self.Nj] = self.k1(index)

        return mat

    def b_vector(self, A, sigma):

        mock = np.zeros([self.Ni, self.Nj], dtype=np.complex128)
        inn = A.copy()[1:-1, 1:-1]
        prod1 = (
                -1j * self.omega * self.mu0 /
                2 / np.pi * self.dr * self.dz *
                inn * sigma[1:-1, 1:-1] * np.sqrt(self.R[1:-1, 1:-1])
                )
        prod2 = self.mu0 * self.Ic / 2 / np.pi * np.sqrt(self.Coils[:, 0])

        for j in range(0, self.Nj):
            Rb = self.R[-1, j]
            Zb = self.Z[-1, j]
            kj = self.kjf(self.R[1:-1, 1:-1], Rb, self.Z[1:-1, 1:-1], Zb)
            ki = self.kif(self.Coils[:, 0], Rb, self.Coils[:, 1], Zb)
            mock[-1, j] = (np.sum(prod1 * np.sqrt(1 / Rb) * self.Gf(kj)) +
                           np.sum(prod2 * np.sqrt(1 / Rb) * self.Gf(ki)))

        for i in range(1, self.Ni):
            Rb = self.R[i, -1]
            Zb = self.Z[i, -1]
            kj = self.kjf(self.R[1:-1, 1:-1], Rb, self.Z[1:-1, 1:-1], Zb)
            ki = self.kif(self.Coils[:, 0], Rb, self.Coils[:, 1], Zb)
            mock[i, -1] = (np.sum(prod1 * np.sqrt(1 / Rb) * self.Gf(kj)) +
                           np.sum(prod2 * np.sqrt(1 / Rb) * self.Gf(ki)))
            Rb = self.R[i, 0]
            Zb = self.Z[i, 0]
            kj = self.kjf(self.R[1:-1, 1:-1], Rb, self.Z[1:-1, 1:-1], Zb)
            ki = self.kif(self.Coils[:, 0], Rb, self.Coils[:, 1], Zb)
            mock[i, 0] = (np.sum(prod1 * np.sqrt(1 / Rb) * self.Gf(kj)) +
                          np.sum(prod2 * np.sqrt(1 / Rb) * self.Gf(ki)))

        return mock.reshape(-1)

    def magnetic_vector_solver(self, temp, iterations, A_guess):
        sigma = self.sigmaf(temp)
        B0 = self.b_vector(A_guess, sigma)
        D0 = self.matrix_setup(sigma)
        A_guess = A_guess.reshape(-1)
        A0 = splinalg.bicgstab(D0, B0, x0=A_guess)[0]
        A = A0.reshape(self.Ni, self.Nj)
        A_prev = np.copy(A)  # Store previous iteration matrix

        for j in range(iterations):
            # start_time = time.time()
            B = self.b_vector(A, sigma)
            D = self.matrix_setup(sigma)
            A_new = splinalg.bicgstab(D, B, x0=A_prev.reshape(-1))[0].reshape(self.Ni, self.Nj)
            # error = np.max(np.abs(A - A_prev) / (np.abs(A)+1e-12))
            # error2 = np.max(np.abs(A - A_prev))
            #
            # elapsed_time = time.time() - start_time
            #
            # print(j+1, error*100, error2, elapsed_time)
            A_prev, A = A, A_new  # Shift matrices

        max_error = np.max(np.abs(A - A_prev)/(np.abs(A)+1e-12))  # Compute max error after final iteration

        return A, max_error

    def Electric_Field(self, A):

        return -1j * self.omega * A

    def BzBr(self, A):

        Bz = np.zeros([self.Ni, self.Nj], dtype=np.complex128)
        Br = np.zeros([self.Ni, self.Nj], dtype=np.complex128)

        AR = A*self.R

        Bz[1:-1, :] = (
                1 / self.R[1:-1, :] *
                1 / 2 / self.dr * (AR[2:, :] - AR[:-2, :])
        )

        # Bz[0, :] = (-3*AR[0, :]+4*AR[1, :]-AR[2, :]) / (2*dr) / (self.R[0, :]+dr/14)   # I fine tuned the 14.
        Bz[-1, :] = (3*AR[-1, :]-4*AR[-2, :]+AR[-3, :]) / (2*self.dr) / self.R[-1, :]
        Bz[0, :] = Bz[1, :]
        Br[1:-1, 1:-1] = - (
                1 / 2 / self.dz * (A[1:-1, 2:] - A[1:-1, :-2])
        )
        Br[0, :] = Br[1, :]
        Br[-1, :] = Br[-2, :]
        Br[:, 0] = Br[:, 1]
        Br[:, -1] = Br[:, -2]

        # Calculating magnetic field intensity.
        Hz = Bz/self.mu0
        Hr = Br/self.mu0

        return Bz, Br, Hz, Hr

    def FzFrP(self, T, E, Br, Bz):
        one, two, three = E * np.conjugate(Br), E * np.conjugate(Bz), E * np.conjugate(E)

        sigma = self.sigmaf(T)

        Fz = -1 / 2 * sigma * one.real
        Fr = 1 / 2 * sigma * two.real
        P = 1 / 2 * sigma * three.real

        return Fz, Fr, P

    def overall_maxwell(self, T, iterations, A_guess):

        """
        :param A_guess:
        :param T: current temperature of the grid
        :param iterations: number of iterations
        :return: Fz, Fr, P
        """

        A, err = self.magnetic_vector_solver(T, iterations, A_guess)
        Bz, Br, _, _ = self.BzBr(A)
        E = self.Electric_Field(A)
        Fz, Fr, P = self.FzFrP(T, E, Br, Bz)

        return Fz, Fr, P, A

    def FzFrP_overall(self, T, A):

        Bz, Br, _, _ = self.BzBr(A)
        E = self.Electric_Field(A)
        Fz, Fr, P = self.FzFrP(T, E, Br, Bz)

        return Fz, Fr, P

