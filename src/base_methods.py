import numpy as np
from scipy.interpolate import RegularGridInterpolator
from all_variables import *
from scipy.sparse.linalg import spsolve
from numba import njit


""" Base functions for the solver."""


@njit
def phi_faces(phi):
    """
    phi_faces_inner
    """
    z_faces = (phi[1:, 1:-1] + phi[:-1, 1:-1]) / 2
    r_faces = (phi[1:-1, 1:] + phi[1:-1, :-1]) / 2

    return z_faces, r_faces


@njit
def phi_faces_upwind(phi, uz, ur):
    """
    phi_faces_inner
    """
    z_faces = np.where(uz[:, 1:-1] > 0, phi[:-1, 1:-1], phi[1:, 1:-1])
    r_faces = np.where(ur[1:-1, :] > 0, phi[1:-1, :-1], phi[1:-1, 1:])
    return z_faces, r_faces


@njit
def harmonic_phi_faces(k):
    """
    Harmonic mean at interior cell faces
    """
    z_face = 2 * k[1:, 1:-1] * k[:-1, 1:-1] / (k[1:, 1:-1] + k[:-1, 1:-1])
    r_face = 2 * k[1:-1, 1:] * k[1:-1, :-1] / (k[1:-1, 1:] + k[1:-1, :-1])
    return z_face, r_face


@njit
def uz__center(uz):
    """
    Gives the values at the center interior of the grid.
    Uses normal velocity averaging not that deep.
    """
    return 1 / 2 * (uz[1:, 1:-1] + uz[:-1, 1:-1])


@njit
def phi_faces_uz_cells(phi):
    """Only for the interior ux cells you are solving for. Simple averaging."""
    z_faces = phi[1:-1, 1:-1]
    r_faces = 0.25 * (phi[2:-1, 1:] + phi[1:-2, 1:] + phi[2:-1, 0:-1] + phi[1:-2, 0:-1])
    return z_faces, r_faces


@njit
def u_faces_uz_cells(uz, ur):
    """
    Returns average velocity values at faces as follows: WEux, NSux, WEuy, NSuy
    Only for the interior ux cells."""
    WEuz, NSuz = 0.5 * (uz[1:, 1:-1] + uz[:-1, 1:-1]), 0.5 * (uz[1:-1, 1:] + uz[1:-1, :-1])
    NSur, WEur = 0.5 * (ur[1:, 1:-1] + ur[:-1, 1:-1]), 0.5 * (ur[:, 2:-1] + ur[:, 1:-2])
    return WEuz, NSuz, WEur, NSur


@njit
def ur_center_uz_cells(uz, ur):
    """
    Returns the average uy velocity at all the ux locations.
    On the edges uses neumann boundary conditions.
    """
    average = np.zeros_like(uz)
    average[:, 1:-1] = 0.25 * (
            ur[1:, 1:] + ur[0:-1, 1:] + ur[0:-1, 0:-1] + ur[1:, 0:-1]
    )
    average[:, 0] = average[:, 1]
    average[:, -1] = average[:, -2]

    return average


@njit
def ur__center(ur):
    """
    NB ur must have the correct shape!
    Gives the values at the center of the grid.  Shape = (Ni+2, Nj+2)
    Uses normal velocity averaging not that deep.
    """
    return 1 / 2 * (ur[1:-1, 1:] + ur[1:-1, :-1])


@njit
def phi_faces_ur_cells(phi):
    """Only for the interior cells you are solving for.  Simple averaging."""
    r_faces = phi[1:-1, 1:-1]
    z_faces = 0.25 * (phi[1:, 2:-1] + phi[1:, 1:-2] + phi[:-1, 2:-1] + phi[:-1, 1:-2])
    return z_faces, r_faces


@njit
def u_faces_ur_cells(uz, ur):
    """
    Returns average velocity values at faces as follows: WEux, NSux, WEuy, NSuy
    Only for the interior uy cells."""
    NSuz, WEuz = 0.5 * (uz[1:-1, 1:] + uz[1:-1, :-1]), 0.5 * (uz[2:-1, :] + uz[1:-2, :])
    WEur, NSur = 0.5 * (ur[1:, 1:-1] + ur[:-1, 1:-1]), 0.5 * (ur[1:-1, 1:] + ur[1:-1, :-1])
    return WEuz, NSuz, WEur, NSur


@njit
def uz_at_ur(uz, ur):
    """
    Returns the average ux velocity at all the uy locations.
    Uses neumann at the edges where 4 point average not possible.
    """
    average = np.zeros_like(ur)
    average[1:-1, :] = 0.25 * (
            uz[1:, 1:] + uz[0:-1, 1:] + uz[0:-1, 0:-1] + uz[1:, 0:-1]
    )
    average[0, :] = average[1, :]
    average[-1, :] = average[-2, :]

    return average


@njit
def convective_flux(uz, ur, phi):
    """Has shape of ux and of uy."""
    phi_face = phi_faces(phi)
    z_flux = phi_face[0] * uz[:, 1:-1]
    r_flux = phi_face[1] * ur[1:-1, :]

    return z_flux, r_flux


@njit
def convective_flux_upwind(uz, ur, phi):
    """
    Upwind scheme for convective fluxes for interior cells.
    """
    phi_face = phi_faces_upwind(phi, uz, ur)
    z_flux = phi_face[0] * uz[:, 1:-1]
    r_flux = phi_face[1] * ur[1:-1, :]
    return z_flux, r_flux


@njit
def flux_uz_cells(uz, ur, rho):
    """
    Returns the flux for the uz velocity cells.
    Only interior.
    Only interior.
    """
    flux_phi_cells_we, flux_phi_cells_ns = convective_flux(uz, ur, rho)
    flux_we = 0.5 * (flux_phi_cells_we[1:, :] + flux_phi_cells_we[:-1, :])

    flux_ns = 0.5 * (flux_phi_cells_ns[1:, :] + flux_phi_cells_ns[:-1, :])

    return flux_we, flux_ns


@njit
def flux_uz_cells_upwind(uz, ur, rho):
    """
    Returns the flux for the ux velocity cells.
    Only interior.
    Only interior.
    """
    flux_phi_cells_we, flux_phi_cells_ns = convective_flux_upwind(uz, ur, rho)
    flux_we = 0.5 * (flux_phi_cells_we[1:, :] + flux_phi_cells_we[:-1, :])
    # flux_we = np.where(ux[:-1, 1:-1] > 0, flux_phi_cells_we[:-1, :], flux_phi_cells_we[1:, :])

    flux_ns = 0.5 * (flux_phi_cells_ns[1:, :] + flux_phi_cells_ns[:-1, :])
    # flux_ns = np.where(uy[1:-2, :] > 0, flux_phi_cells_ns[:-1, :], flux_phi_cells_ns[1:, :])

    return flux_we, flux_ns


@njit
def flux_ur_cells(uz, ur, rho):
    """
    Returns the flux for the uy velocity cells.
    Only interior.
    Only interior.
    """
    flux_phi_cells_we, flux_phi_cells_ns = convective_flux(uz, ur, rho)
    flux_ns = 0.5 * (flux_phi_cells_ns[:, 1:] + flux_phi_cells_ns[:, :-1])
    flux_we = 0.5 * (flux_phi_cells_we[:, 1:] + flux_phi_cells_we[:, :-1])

    return flux_we, flux_ns


@njit
def flux_ur_cells_upwind(uz, ur, rho):
    """
    Returns the flux for the uy velocity cells.
    Only interior.
    Only interior.
    """
    flux_phi_cells_we, flux_phi_cells_ns = convective_flux_upwind(uz, ur, rho)
    flux_ns = 0.5 * (flux_phi_cells_ns[:, 1:] + flux_phi_cells_ns[:, :-1])
    # flux_ns = np.where(uy[:, :-1] > 0, flux_phi_cells_ns[:, :-1], flux_phi_cells_ns[:, 1:])

    flux_we = 0.5 * (flux_phi_cells_we[:, 1:] + flux_phi_cells_we[:, :-1])
    # flux_we = np.where(ux[:, 1:-2] > 0, flux_phi_cells_we[:, :-1], flux_phi_cells_we[:, 1:])

    return flux_we, flux_ns


@njit
def divergence_s(rho, uz, ur):
    rho_z_faces, rho_r_faces = phi_faces(phi=rho)

    div = (
            (uz[1:, 1:-1]*rho_z_faces[1:, :]*z_area[1:, :] - uz[:-1, 1:-1]*rho_z_faces[:-1, :]*z_area[:-1, :]) +
            (ur[1:-1, 1:]*rho_r_faces[:, 1:]*r_area[:, 1:] - ur[1:-1, :-1]*rho_r_faces[:, 1:]*r_area[:, :-1])
    )

    return div / volume


def compute_dt(uz, ur, T, p, rho):
    """
    Compute time step based on CFL condition
    Only on the basis of the interior of the torch.
    """

    uz = uz__center(uz)
    ur = ur__center(ur)

    # Convective time step limit
    dt_conv_x = (Zuz[1:, 1:-1]-Zuz[:-1, 1:-1]) / (np.abs(uz) + 1e-10)
    dt_conv_y = (Rur[1:-1, 1:]-Rur[1:-1, :-1]) / (np.abs(ur) + 1e-10)

    # Viscous time step limit (diffusion)
    mu = muvf(T[1:-1, 1:-1])
    dt_visc_z = 0.5 * rho[1:-1, 1:-1] * (Rur[1:-1, 1:]-Rur[1:-1, :-1])**2 / (mu + 1e-10)
    dt_visc_r = 0.5 * rho[1:-1, 1:-1] * (Zuz[1:, 1:-1]-Zuz[:-1, 1:-1])**2 / (mu + 1e-10)

    # Thermal diffusion time step limit
    k = kf(T[1:-1, 1:-1])
    cp = Cpf(T[1:-1, 1:-1])
    dt_thermal_z = 0.5 * rho[1:-1, 1:-1] * (Rur[1:-1, 1:]-Rur[1:-1, :-1])**2 * cp / (k + 1e-10)
    dt_thermal_r = 0.5 * rho[1:-1, 1:-1] * (Zuz[1:, 1:-1]-Zuz[:-1, 1:-1])**2 * cp / (k + 1e-10)

    # Combined time step
    dt_min = min(
        np.min(dt_conv_x),
        np.min(dt_conv_y),
        np.min(dt_visc_z),
        np.min(dt_visc_r),
        np.min(dt_thermal_z),
        np.min(dt_thermal_r)
    )

    dt = CFL * dt_min

    return dt


def pressure_poisson_matrix(Ni, Nj):
    import scipy.sparse as sp
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
        return Nj*i+j

    for i in range(Ni):
        for j in range(Nj):
            ind = idx(i, j)
            if i == 0:
                A[ind, ind] = 1
            elif i == Ni-1:
                A[ind, ind] = 1
            elif j == 0:
                A[ind, ind] = 1     # Dirichlet p=0
                A[ind, ind+1] = -1
            elif j == Nj-1:
                A[ind, ind] = 1
                A[ind, ind-1] = -1
            else:
                A[ind, ind] = - 2 * (1/(Zuz[i, j]-Zuz[i-1, j-1])**2+1/(Rur[i, j]-Rur[i-1, j-1])**2)
                A[ind, ind+1] = 1 / R[i+1, j+1]/(Rur[i, j]-Rur[i-1, j-1])/2 + 1/(Rur[i, j]-Rur[i-1, j-1])**2
                A[ind, ind-1] = - 1 / R[i+1, j+1]/(Rur[i, j]-Rur[i-1, j-1])/2 + 1/(Rur[i, j]-Rur[i-1, j-1])**2
                A[ind, ind-Nj] = 1 / (Zuz[i, j]-Zuz[i-1, j-1])**2
                A[ind, ind+Nj] = 1 / (Zuz[i, j]-Zuz[i-1, j-1])**2

    A_sparse = sp.csr_matrix(A)

    return A_sparse


POISSON = pressure_poisson_matrix(Ni, Nj)


def matrix_pressure(pressure_poisson_rhs):
    """Pressure poisson right hand side is (Ni, Nj) and p_prev is (Ni+2, Nj+2)"""
    # p_next = np.linalg.solve(LHS_Poisson_Matrix, pressure_poisson_rhs.reshape(-1)).reshape(Ni+2, Nj+2)
    p_next = spsolve(POISSON, pressure_poisson_rhs.reshape(-1)).reshape(Ni + 2, Nj + 2)

    return p_next


def velocity_pressure_correction(uz_tent, ur_tent, rho, rho_prev, pressure_correction_next, dt, alpha_p):
    uz_next = np.zeros_like(uz_tent)
    ur_next = np.zeros_like(ur_tent)
    # Update the pressure
    p_next = pressure_correction_next
    # Correct the velocities to be incompressible

    rho_z_faces, rho_r_faces = phi_faces(rho)  # Remove net hierdie deel

    p_correction_grad_z = (
            (pressure_correction_next[2:-1, 1:-1] - pressure_correction_next[1:-2, 1:-1])
            /
            (Z[2:-1, 1:-1]-Z[1:-2, 1:-1])
    )

    uz_next[1:-1, 1:-1] = (
            uz_tent[1:-1, 1:-1] * rho_z_faces[1:-1, :] -
            dt   # en die deel met die rho.

            * p_correction_grad_z
    ) / rho_z_faces[1:-1, :]

    p_correction_grad_r = (
            (pressure_correction_next[1:-1, 2:-1] -
             pressure_correction_next[1:-1, 1:-2]
             )
            /
            (R[1:-1, 2:-1]-R[1:-1, 1:-2])
    )

    ur_next[1:-1, 1:-1] = (
            ur_tent[1:-1, 1:-1] * rho_r_faces[:, 1:-1] -
            dt   # en die deel met die rho.

            * p_correction_grad_r
    ) / rho_r_faces[:, 1:-1]

    return (
        uz_next*alpha_p + (1-alpha_p)*uz_prev,
        ur_next*alpha_p + (1-alpha_p)*ur_prev,
        p_next
    )


def interpolator(z, r, field, R, Z):

    r_coords = R[0, :]
    z_coords = Z[:, 0]
    interpolate = RegularGridInterpolator((z_coords, r_coords), field, bounds_error=False, fill_value=None)

    return interpolate((z, r))


def better_interpolator(R_original, Z_original, Field_Original, R_new, Z_new):
    """
    Interpolate field from original (R, Z) grid to new (R, Z) points.

    Inputs:
        R_original, Z_original: 2D meshgrids (shape [Ni, Nj])
        Field_Original: 2D array of values (shape [Ni, Nj])
        R_new, Z_new: meshgrids or 1D arrays of new coordinates

    Output:
        Field_new: interpolated values at new coordinates
    """

    # Extract 1D grid coordinates (assumes structured mesh)
    r_coords = R_original[0, :]
    z_coords = Z_original[:, 0]

    # Interpolator expects shape (Z, R)
    interpolator_func = RegularGridInterpolator(
        (z_coords, r_coords),
        Field_Original,
        bounds_error=False,
        fill_value=None
    )

    # Flatten and combine new coordinates
    points = np.column_stack((Z_new.ravel(), R_new.ravel()))

    # Interpolate
    Field_new_flat = interpolator_func(points)

    return Field_new_flat.reshape(R_new.shape)


def save(T, uz, ur):
    """R first and then Z"""
    np.save("Storage/T_all.npy", np.array([T, R, Z]))
    np.save("Storage/uz_all.npy", np.array([uz, Ruz, Zuz]))
    np.save("Storage/ur_all.npy", np.array([ur, Rur, Zur]))
    return


def save_conditions(T, uz, ur, name, num):
    """R first and then Z"""
    T_name = "Storage"+str(num)+"/T" + "_" + name + ".npy"
    uz_name = "Storage"+str(num)+"/uz" + "_" + name + ".npy"
    ur_name = "Storage"+str(num)+"/ur" + "_" + name + ".npy"
    np.save(T_name, np.array([T, R, Z]))
    np.save(uz_name, np.array([uz, Ruz, Zuz]))
    np.save(ur_name, np.array([ur, Rur, Zur]))
    return


def extract(R, Z, Ruz, Zuz, Rur, Zur):
    """R first then Z"""
    T_load, R_load, Z_load = np.load("Storage/T_all.npy")
    T = better_interpolator(R_load, Z_load, T_load, R, Z)
    uz_load, Ruz_load, Zuz_load = np.load("Storage/uz_all.npy")
    uz = better_interpolator(Ruz_load, Zuz_load, uz_load, Ruz, Zuz)
    ur_load, Rur_load, Zur_load = np.load("Storage/ur_all.npy")
    ur = better_interpolator(Rur_load, Zur_load, ur_load, Rur, Zur)

    return T, uz, ur


def extract_conditions(R, Z, Ruz, Zuz, Rur, Zur, name, num):
    """R first then Z"""
    T_name = "Storage"+str(num)+"/T" + "_" + name + ".npy"
    uz_name = "Storage"+str(num)+"/uz" + "_" + name + ".npy"
    ur_name = "Storage"+str(num)+"/ur" + "_" + name + ".npy"
    T_load, R_load, Z_load = np.load(T_name)
    T = better_interpolator(R_load, Z_load, T_load, R, Z)
    uz_load, Ruz_load, Zuz_load = np.load(uz_name)
    uz = better_interpolator(Ruz_load, Zuz_load, uz_load, Ruz, Zuz)
    ur_load, Rur_load, Zur_load = np.load(ur_name)
    ur = better_interpolator(Rur_load, Zur_load, ur_load, Rur, Zur)

    return T, uz, ur
