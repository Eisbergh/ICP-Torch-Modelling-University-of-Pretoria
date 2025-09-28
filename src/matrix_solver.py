import numpy as np
from all_variables import dz, dr, R
import scipy.sparse as sp


"""Solving the laplacian equation."""


def pressure_poisson_matrix(Ni, Nj):

    """
    Unpacking is in the Nj then Ni method.

    [N00, N01, N02, N03 ... N10, N11, N12, N13, ... N20, N21, N22, N23 ...]

    So for the b-vector, make it np.reshape(-1).

    Then np.reshape(Ni, Nj) at the end.
    """

    Nj = Nj+2
    Ni = Ni+2

    A = np.zeros([Ni*Nj, Ni*Nj])

    def idx(i, j):
        return (Nj)*i+j

    for i in range(Ni):
        for j in range(Nj):
            ind = idx(i, j)
            if j == Nj-1:
                A[ind, ind] = 1
                A[ind, ind-1] = -1
            elif j == 0:
                A[ind, ind] = 1     # Dirichlet p=0
                A[ind, ind] = -1
            elif i == Ni-1:
                A[ind, ind] = 1

            elif i == 0:
                A[ind, ind] = 1
                A[ind, ind - Nj] = -1
            else:
                A[ind, ind] = -2*(1/dz**2+1/dr**2)
                A[ind, ind+1] = 1/R[i, j]/dr/2+1/dr**2
                A[ind, ind-1] = -1/R[i, j]/dr/2+1/dr**2
                A[ind, ind-Nj] = 1/dz**2
                A[ind, ind+Nj] = 1/dz**2

    A = sp.csr_matrix(A)

    return A
