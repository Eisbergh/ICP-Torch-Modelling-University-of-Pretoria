import numpy as np

from base_methods import *


def azimuthal_u(ur, T):
    u_theta = np.zeros_like(ur)

    rho350 = rhof(350)
    rhos = rhof(T[:, 1:])
    factor = rho350/rhos
    u_theta[:, int(np.ceil(Nj_sheath+(Nj-Nj_sheath)*SHEATH_FACTOR)):] = AZIMUTHAL
    u_theta[:, Nj_sheath:] = AZIMUTHAL
    return u_theta


def divergence_s(rho, uz, ur):
    rho_z_faces, rho_r_faces = phi_faces(phi=rho)

    div = (
            (uz[1:, 1:-1]*rho_z_faces[1:, :]*z_area[1:, :] - uz[:-1, 1:-1]*rho_z_faces[:-1, :]*z_area[:-1, :]) +
            (ur[1:-1, 1:]*rho_r_faces[:, 1:]*r_area[:, 1:] - ur[1:-1, :-1]*rho_r_faces[:, 1:]*r_area[:, :-1])
    )

    return div / volume


def hybrid(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn):
    """Hybrid scheme as done in Versreeg pg. 124"""
    aW = np.maximum.reduce([Fw, Dw + Fw / 2, np.zeros_like(Fw)])
    aE = np.maximum.reduce([-Fe, De - Fe / 2, np.zeros_like(Fe)])
    aS = np.maximum.reduce([Fs, Ds + Fs / 2, np.zeros_like(Fs)])
    aN = np.maximum.reduce([-Fn, Dn - Fn / 2, np.zeros_like(Fn)])
    dF = Fe-Fw+Fn-Fs
    return aW, aE, aS, aN, dF


def exponential(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn):
    """
    Exponential scheme (exact for 1D steady CD), robust and accurate.
    A(Pe) = Pe / (exp(Pe) - 1) with A(0)=1 by limit.
    """
    eps = 1e-30
    Pe_w = Fw / np.maximum(Dw, eps)
    Pe_e = Fe / np.maximum(De, eps)
    Pe_s = Fs / np.maximum(Ds, eps)
    Pe_n = Fn / np.maximum(Dn, eps)

    def A(Pe):
        # Stable evaluation near zero
        out = np.empty_like(Pe)
        small = np.abs(Pe) < 1e-6
        out[small] = 1.0 - Pe[small]/2.0 + Pe[small]**2/12.0  # series
        out[~small] = Pe[~small] / (np.exp(Pe[~small]) - 1.0)
        # Ensure nonnegative
        return np.maximum(out, 0.0)

    aW = Dw * A(Pe_w) + np.maximum(Fw, 0.0)
    aE = De * A(Pe_e) + np.maximum(-Fe, 0.0)
    aS = Ds * A(Pe_s) + np.maximum(Fs, 0.0)
    aN = Dn * A(Pe_n) + np.maximum(-Fn, 0.0)
    dF = Fe - Fw + Fn - Fs
    return aW, aE, aS, aN, dF


def upwind(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn):
    """Pure upwind differencing (bounded, diffusive)."""
    aW = Dw + np.maximum(Fw, 0.0)
    aE = De + np.maximum(-Fe, 0.0)
    aS = Ds + np.maximum(Fs, 0.0)
    aN = Dn + np.maximum(-Fn, 0.0)
    dF = Fe - Fw + Fn - Fs
    return aW, aE, aS, aN, dF


def central(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn):
    """Central differencing (accurate for |Pe|≲2, may be unbounded for high Pe)."""
    aW = Dw - 0.5 * Fw
    aE = De + 0.5 * Fe
    aS = Ds - 0.5 * Fs
    aN = Dn + 0.5 * Fn
    # Clip negatives to preserve boundedness if you want a safer CDS
    aW = np.maximum(aW, 0.0)
    aE = np.maximum(aE, 0.0)
    aS = np.maximum(aS, 0.0)
    aN = np.maximum(aN, 0.0)
    dF = Fe - Fw + Fn - Fs
    return aW, aE, aS, aN, dF


def power_law(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn):
    """
    Patankar power-law scheme (good accuracy, bounded).
    A(Pe) = max(0, (1 - 0.1*|Pe|)^5)
    """
    # Avoid divide-by-zero: where D==0, set Pe=0 so A=1
    eps = 1e-30
    Pe_w = Fw / np.maximum(Dw, eps)
    Pe_e = Fe / np.maximum(De, eps)
    Pe_s = Fs / np.maximum(Ds, eps)
    Pe_n = Fn / np.maximum(Dn, eps)

    def A(Pe):
        return np.maximum(0.0, (1.0 - 0.1 * np.abs(Pe))**5)

    aW = Dw * A(Pe_w) + np.maximum(Fw, 0.0)
    aE = De * A(Pe_e) + np.maximum(-Fe, 0.0)
    aS = Ds * A(Pe_s) + np.maximum(Fs, 0.0)
    aN = Dn * A(Pe_n) + np.maximum(-Fn, 0.0)
    dF = Fe - Fw + Fn - Fs
    return aW, aE, aS, aN, dF


scheme = exponential


def energy_s(uz, ur, rho, T, P, dt):

    T_new = np.zeros_like(T)
    h_new = np.zeros_like(T)

    h = hf(T)

    k = kf(T)
    Cp = Cpf(T)

    Fwe, Fns = convective_flux(uz, ur, rho)
    _, k_r_faces = phi_faces_upwind(k, uz, ur)
    k_z_faces, k_r_faces = phi_faces(k)
    _, Cp_r_faces = phi_faces_upwind(Cp, uz, ur)
    Cp_z_faces, Cp_r_faces = phi_faces(Cp)

    Fe = Fwe[1:, :] * z_area[1:, :]
    Fw = Fwe[:-1, :] * z_area[:-1, :]
    Fn = Fns[:, 1:] * r_area[:, 1:]
    Fs = Fns[:, :-1] * r_area[:, :-1]
    De = (k_z_faces/Cp_z_faces)[1:, :] * z_area[1:, :] / (Z[2:, 1:-1]-Z[1:-1, 1:-1])
    Dw = (k_z_faces/Cp_z_faces)[:-1, :] * z_area[:-1, :] / abs(Z[:-2, 1:-1]-Z[1:-1, 1:-1])
    Dn = (k_r_faces/Cp_r_faces)[:, 1:] * r_area[:, 1:] / (R[1:-1, 2:]-R[1:-1, 1:-1])
    Ds = (k_r_faces/Cp_r_faces)[:, :-1] * r_area[:, :-1] / abs(R[1:-1, :-2]-R[1:-1, 1:-1])

    # Dn[0:Ni_carrier + 1, :Nj_carrier+1] = 0                # Boundary condition vir temperatuur!
    # Ds[0:Ni_carrier+1, :Nj_carrier+1] = 0

    aW, aE, aS, aN, dF = scheme(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn)

    B = 0 + (P - Qrf(T[1:-1, 1:-1])-Qrf2(T[1:-1, 1:-1], Nj_safe, Nj)) * volume  # source.  For Qr and P.
    # Is actually dependent on temperature.  Maal volume.

    # B = 0 + (P - Qrf(T[1:-1, 1:-1])) * volume  # source.  For Qr and P.

    for _ in range(3):

        ap0 = volume * rho[1:-1, 1:-1] / dt

        ap = aN+aS+aW+aE-ap0+dF

        h_new[1:-1, 1:-1] = (
            1 / ap0 * (
                - ap * h[1:-1, 1:-1]
                + aN * h[1:-1, 2:]
                + aS * h[1:-1, :-2]
                + aE * h[2:, 1:-1]
                + aW * h[:-2, 1:-1]
                + B
            )
        )

        T_new[1:-1, 1:-1] = Tf(h_new[1:-1, 1:-1])

        temperature_boundaries(T_new)

        rho = rhof(T_new)

    return T_new, rhof(T_new)


def z_momentum_s(uz, ur, rho, rho_prev, p, T, Fz, dt):
    """ z momentum coefficients as using discretisation and hybrid scheme."""

    # Viscosity:
    mu = muvf(T)
    Fwe, Fns = flux_uz_cells(uz=uz, ur=ur, rho=rho_prev)

    mu_z_faces, mu_r_faces = phi_faces_uz_cells(mu)

    Fe = Fwe[1:, :]*z_area_uz[1:, :]
    Fw = Fwe[:-1, :]*z_area_uz[:-1, :]
    Fn = Fns[:, 1:]*r_area_uz[:, 1:]
    Fs = Fns[:, :-1]*r_area_uz[:, :-1]
    De = 2*mu_z_faces[1:, :]*z_area_uz[1:, :]/(Zuz[2:, 1:-1]-Zuz[1:-1, 1:-1])
    Dw = 2*mu_z_faces[:-1, :]*z_area_uz[:-1, :]/abs(Zuz[:-2, 1:-1]-Zuz[1:-1, 1:-1])
    Dn = mu_r_faces[:, 1:]*r_area_uz[:, 1:]/(Ruz[1:-1, 2:]-Ruz[1:-1, 1:-1])
    Ds = mu_r_faces[:, :-1] * r_area_uz[:, :-1]/abs(Ruz[1:-1, :-2]-Ruz[1:-1, 1:-1])

    aW, aE, aS, aN, dF = scheme(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn)

    dur_dz = (ur[2:-1, :] - ur[1:-2, :]) / (Z[2:-1, 1:]-Z[1:-2, 1:])
    diff_source = (
            mu_r_faces[:, 1:]*dur_dz[:, 1:]*r_area_uz[:, 1:] -
            mu_r_faces[:, :-1] * dur_dz[:, :-1] * r_area_uz[:, :-1]
    ) / volume_uz
    # diff_source = 0
    B = diff_source * volume_uz + 0.5*(Fz[1:, :]+Fz[:-1, :]) * volume_uz

    dp = (
        (p[2:-1, 1:-1] - p[1:-2, 1:-1]) / (Z[2:-1, 1:-1]-Z[1:-2, 1:-1])
    ) * volume_uz

    rho_z_faces, rho_r_faces = phi_faces(rho_prev)   # phi_faces_upwind(rho_prev, uz, ur)

    ap0 = volume_uz * rho_z_faces[1:-1, :] / dt  # Constant since rho is constant.

    ap = aW+aE+aS+aN-ap0+dF    # NB watch net vir die twee maal terme!

    rho_uz_next = np.zeros_like(uz)
    uz_next = np.zeros_like(uz)

    rho_uz_next[1:-1, 1:-1] = (
        1 / (volume_uz / dt) * (
            -ap*uz[1:-1, 1:-1]
            + aN * uz[1:-1, 2:]
            + aS * uz[1:-1, 0:-2]
            + aE * uz[2:, 1:-1]
            + aW * uz[:-2, 1:-1]
            + B
            + dp
        )
    )
    rho_z_faces, rho_r_faces = phi_faces(rho)  # phi_faces_upwind(rho, uz, ur)
    uz_next[1:-1, 1:-1] = rho_uz_next[1:-1, 1:-1] / rho_z_faces[1:-1, :]

    return uz_next, rho_uz_next


def r_momentum_s(uz, ur, rho, rho_prev, p, T, Fr, dt):
    """r momentum coefficients"""
    # Viscosity
    mu = muvf(T)
    rho_z_faces, rho_r_faces = phi_faces_upwind(rho_prev, uz, ur)

    Fwe, Fns = flux_ur_cells(uz=uz, ur=ur, rho=rho_prev)

    mu_z_faces, mu_r_faces = phi_faces_ur_cells(mu)

    Fe = Fwe[1:, :]*z_area_ur[1:, :]
    Fw = Fwe[:-1, :]*z_area_ur[:-1, :]
    Fn = Fns[:, 1:]*r_area_ur[:, 1:]
    Fs = Fns[:, :-1] * r_area_ur[:, :-1]
    Dn = 2 * mu_r_faces[:, 1:] * r_area_ur[:, 1:] / (Rur[1:-1, 2:]-Rur[1:-1, 1:-1])
    Ds = 2 * mu_r_faces[:, :-1] * r_area_ur[:, :-1] / abs(Rur[1:-1, :-2]-Rur[1:-1, 1:-1])
    De = mu_z_faces[1:, :] * z_area_ur[1:, :] / (Zur[2:, 1:-1]-Zur[1:-1, 1:-1])
    Dw = mu_z_faces[:-1, :] * z_area_ur[:-1, :] / abs(Zur[:-2, 1:-1]-Zur[1:-1, 1:-1])

    aW, aE, aS, aN, dF = scheme(Fw, Dw, Fe, De, Fs, Ds, Fn, Dn)

    duz_dr = (uz[:, 2:-1] - uz[:, 1:-2]) / dr
    diff_source = (
            mu_z_faces[1:, :] * duz_dr[1:, :] * z_area_ur[1:, :] -
            mu_z_faces[:-1, :] * duz_dr[:-1, :] * z_area_ur[:-1, :]
    ) / volume_ur

    # Source terms
    u_theta = azimuthal_u(ur, T)

    Fr = np.roll(Fr, 0, 1)
    B = (
        diff_source*volume_ur
        +
        0.5*(Fr[:, 1:]+Fr[:, :-1]) * volume_ur
        +
        u_theta[1:-1, 1:-1]**2*volume_ur*rho_r_faces[:, 1:-1]/Rur[1:-1, 1:-1]
    )

    dp = (
             (p[1:-1, 2:-1] - p[1:-1, 1:-2]) / (R[1:-1, 2:-1]-R[1:-1, 1:-2])
    ) * volume_ur

    Sp = (
            -4*np.pi*phi_faces(mu)[1][:, 1:-1] * (Zur[2:, 1:-1]-Zur[1:-1, 1:-1]) *
            np.log((R[1:-1, 2:-1])/(R[1:-1, 1:-2]))
    )

    # Sp = 0     # Kyk hierso!

    ap0 = volume_ur * rho_r_faces[:, 1:-1] / dt

    ap = aN+aS+aW+aE-ap0+dF-Sp  # Watch net vir *2!

    ur_next = np.zeros_like(ur)
    rho_ur_next = np.zeros_like(ur)

    rho_ur_next[1:-1, 1:-1] = (
        1 / (volume_ur/dt) * (
            -ap*ur[1:-1, 1:-1]
            + aN * ur[1:-1, 2:]
            + aS * ur[1:-1, :-2]
            + aE * ur[2:, 1:-1]
            + aW * ur[:-2, 1:-1]
            + B
            + dp
        )
    )

    rho_z_faces, rho_r_faces = phi_faces_upwind(rho, uz, ur)

    ur_next[1:-1, 1:-1] = rho_ur_next[1:-1, 1:-1] / rho_r_faces[:, 1:-1]

    return ur_next, rho_ur_next


def continuity_s(uz, ur, rho, dt):

    rho_z_faces, rho_r_faces = phi_faces(rho)
    divergence = (
        rho_z_faces[1:]*z_area[1:]*uz[1:, 1:-1] - rho_z_faces[:-1]*z_area[:-1]*uz[:-1, 1:-1]
        + rho_r_faces[:, 1:]*r_area[:, :1]*ur[1:-1, 1:] - rho_r_faces[:, :-1]*r_area[:, :-1]*ur[1:-1, :-1]
    )

    rho_new = np.zeros_like(rho)
    rho_new[1:-1, 1:-1] = rho[1:-1, 1:-1] - dt*divergence
    return rho_new


@njit
def velocity_boundaries(uz_next, ur_next):
    # Apply boundaries on uz_next
    uz_next[0, 1:-1] = U_inlet_main  # Inlet flow
    uz_next[1:3, 1:-1] = U_inlet_main  # Inlet flow
    # uz_next[1, 1:-1] = U_inlet_main  # Inlet flow
    uz_next[-1, 1:-1] = uz_next[-2, 1:-1]  # * inflow_mass_rate_next / outflow_mass_rate_next  # Neumann at the outlet
    uz_next[:, 0] = uz_next[:, 1]  # Neumann at r=0 at the axis.
    uz_next[:, -1] = -uz_next[:, -2]  # Zero at the top boundary by anti-symmetry.  Maak weer anti!
    # uz_next[:, -1] = 0   Hulle was aan gewees.
    # uz_next[:, -2] = 0

    # Apply boundaries on ur_next
    # ur_next[0, 1:-1] = -ur_next[1, 1:-1]  # Inlet flow
    # ur_next[1, 1:-1] = 0  # -ur_next[1, 1:-1]  # Inlet flow
    ur_next[-1, 1:-1] = ur_next[-2, 1:-1]  # Neumann at the outlet
    ur_next[:, 0] = 0  # Zero at the axis.
    ur_next[:, -1] = 0  # Zero at the top.

    # For the ICP torch with sheath gas:
    uz_next[0:Ni_sheath + 1, Nj_sheath:] = 0
    uz_next[0:Ni_sheath+3, int(np.ceil(Nj_sheath+(Nj-Nj_sheath)*SHEATH_FACTOR)):] = U_sheath
    ur_next[0:Ni_sheath+1, Nj_sheath-1:] = 0
    uz_next[0:Ni_carrier + 2, 0:Nj_carrier+2] = 0
    uz_next[0:Ni_carrier + 1, 0:Nj_carrier] = U_carrier
    ur_next[0:Ni_carrier + 2, 0:Nj_carrier+2] = 0

    # Cool
    uz_next[1:-1, 1:-1] = np.clip(uz_next[1:-1, 1:-1], a_min=-40, a_max=40)

    # For stability.
    ur_next[1:-1, 1:-1] = np.clip(ur_next[1:-1, 1:-1], a_min=-10, a_max=10)     # Maak weer a_min=-1

    uz_next[int(Ni/9*5):-1, 1:-5] = np.clip(
        uz_next[int(Ni/9*5):-1, 1:-5], U_sheath, None
    )

    uz_next[1:-1, int(np.ceil(Nj/3*2)):-1] = (
        np.clip(uz_next[1:-1, int(Nj/3*2):-1], U_sheath, None)
    )

    uz_next[0:Ni_sheath + 2, Nj_sheath:] = 0
    uz_next[0:Ni_sheath+3, int(np.ceil(Nj_sheath+(Nj-Nj_sheath)*SHEATH_FACTOR)):] = U_sheath

    uz_next[0:2, int(np.ceil(Nj_sheath+(Nj-Nj_sheath)*SHEATH_FACTOR)):] = U_sheath     # Ook hierdie ene!

    return


@njit
def temperature_boundaries(T_new):
    # Apply boundaries on T_next
    # Left inlet
    global current_time
    global temp_boundary

    T_new[0, :] = temp_boundary
    T_new[1, :] = temp_boundary
    # Right outlet
    T_new[-1, :] = T_new[-2, :]  # Neumann
    # Top wall
    T_new[:, -1] = temp_boundary/1
    T_new[:, -2] = temp_boundary/1
    # T_new[:, -3] = temp_boundary
    # T_new[:, -4] = temp_boundary
    # Bottom radius = 0
    T_new[:, 0] = T_new[:, 1]  # Neumann

    # For the ICP torch with sheath gas:
    T_new[0:Ni_sheath+1, int(np.ceil(Nj_sheath+(Nj-Nj_sheath)*SHEATH_FACTOR)):] = temp_boundary
    T_new[0:Ni_carrier + 1, 0:int(np.ceil(Nj_carrier*0.6))] = temp_boundary
    T_new[0:Ni_carrier + 1, 0:Nj_carrier+1] = temp_boundary

    np.clip(T_new, temp_boundary, None, out=T_new)

    return


def pressure_correction_iterate(rho_prev, rho, p_prev, uz_tent, ur_tent, dt, N_DIVERGENCE_ITERATIONS, current_time):

    """
    Does the pressure correction more than once.

    Solves ∇²p = div / dt iteratively to reduce the divergence.
    """

    dummy = np.zeros_like(p_prev)
    velocity_boundaries(uz_tent, ur_tent)
    div = divergence_s(rho, uz_tent, ur_tent)
    uz_next = uz_tent.copy()
    ur_next = ur_tent.copy()

    drho_dt = (rho-rho_prev)[1:-1, 1:-1] / dt
    drho_dt[0:Ni_carrier + 1, 0:Nj_carrier + 1] = 0
    drho_dt[:Ni_sheath + 1, Nj_sheath:] = 0

    alpha_alpha = 0

    if current_time > SWITCH_TIME:
        alpha_alpha = 0.75

    for n in range(N_DIVERGENCE_ITERATIONS):
        pressure_poisson_rhs = (
                div
                +
                drho_dt*alpha_alpha
        )/dt

        pressure_poisson_rhs[:Ni_sheath, Nj_sheath:] = 0    # NB outside domain make the poisson RHS = 0.
        pressure_poisson_rhs[:Ni_carrier, :Nj_carrier] = 0

        dummy[1:-1, 1:-1] = pressure_poisson_rhs
        dummy[0, :] = Patm
        dummy[-1, :] = Patm-10
        pressure_correction_next = matrix_pressure(dummy)

        uz_next, ur_next, p_next = velocity_pressure_correction(uz_next, ur_next, rho, rho_prev,
                                                                pressure_correction_next, dt, alpha_p)

        velocity_boundaries(uz_next, ur_next)
        div = divergence_s(rho, uz_next, ur_next)
        div[0:Ni_carrier + 1, 0:Nj_carrier + 1] = 0
        div[:Ni_sheath + 1, Nj_sheath:] = 0
        # print(np.max(np.abs(drho_dt*alpha_alpha+div)))
        # print(np.average(np.abs(drho_dt*alpha_alpha+div)))

    return uz_next, ur_next, p_next, drho_dt*alpha_alpha+div


def add_remove_fr(ur, Fr, rho, uz, dt):

    ur_next_remove = np.copy(ur)
    ur_next_add = np.copy(ur)
    rho_z_faces, rho_r_faces = phi_faces_upwind(rho, uz, ur)

    ur_next_remove[1:-1, 1:-1] -= dt*0.5*(Fr[:, 1:]+Fr[:, :-1]) / rho_r_faces[:, 1:-1]
    ur_next_add[1:-1, 1:-1] += dt*0.5*(Fr[:, 1:]+Fr[:, :-1]) / rho_r_faces[:, 1:-1]

    return ur_next_remove, ur_next_add
