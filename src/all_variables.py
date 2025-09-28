import numpy as np


# =============================================================== #
#                        ICP Torch Layout
# =============================================================== #

Lz = 200/1000
Lr = 25/1000
Lr_sheath = 18.8/1000
t_sheath = 2.2/1000
Lz_sheath = 50/1000
Lr_carrier = 3.7/1000
t_carrier = 2/1000
Lz_carrier = 50/1000

Q_main = 3
Q_sheath = 31
Q_carrier = 1

# =============================================================== #
#                       Boundary Values
# =============================================================== #

U_inlet_main = Q_main / 1000 / 60 / (np.pi*(Lr_sheath**2-Lr_carrier**2))
U_sheath = Q_sheath / 1000 / 60 / (np.pi*(Lr**2-(Lr_sheath+t_sheath)**2))
U_carrier = Q_carrier / 1000 / 60 / (np.pi*(Lr_carrier-t_carrier)**2)

temp_boundary = 350

p_atm = 101325

U_inlet = U_inlet_main

SHEATH_FACTOR = 0.25

SWITCH_TIME = 0.000

SHOW_FLOW = True
VERTICAL_DISTANCE = 6

SAVE = False
file_number = 4

# =============================================================== #
#                          dt calcs
# =============================================================== #
acoustic_weight = 0.001
CFL = 0.5
# N_PRESSURE_POISSON_ITERATIONS = 100
# PRESSURE_POISSON_ERROR = 0.05
current_time = 0
N_DIVERGENCE_ITERATIONS = 9    # 3   7 werk baie goed.
DIVERGENCE_ERROR = 0.0001
N_MAGNETIC_FIELD_ITERATIONS = 6
alpha = 0.6  # 0.6
alpha_T = 0.6  # 0.6
alpha_p = 1     # Hierdie werk nie - nie die moeite werd nie.

DPI = 60

# =============================================================== #
#                      Fluid Properties
# =============================================================== #


from parameters import hf, Cpf, kf, rhof, EquationOfState, tempf, muvf, sigmaf, Qrf, d_dT_rhof, Qrf2, Prf
from parameters import muvf as muvf2


def muvf(temp):
    a = muvf2(temp)
    a[:, 14:] = 0.000001
    return a


muvf = muvf2
hf = hf
Cpf = Cpf
kf = kf
rhof = rhof
eos = EquationOfState()
Tf = tempf
sigmaf = sigmaf
Qrf = Qrf
d_dT_rhof = d_dT_rhof
pf = eos.pressure_eos
Qrf2 = Qrf2
Prf = Prf


# =============================================================== #
#                       Magnetic Field:
# =============================================================== #

from temperature_field import currentf

"""
For 7.5 kW use 6
For 5 kW use 5
For 10 kW use 9
F0r 15 kW use 15
For 3 kW (100 A) use 2.5
"""

AZIMUTHAL = 5
kW = 5  # kW
Ic = currentf(kW)
# Ic = 100   for 3 kW

print(f"Current is {Ic} A")
Patm = 101.325 * 1000
omega = 2*np.pi*3*10**6  # Angular frequency (adjust as needed)
mu0 = 4 * np.pi * 1e-7  # Permeability of free space

Coils = np.array(
    [[63 / 1000, 33 / 1000],
     [92 / 1000, 33 / 1000],
     [121 / 1000, 33 / 1000]
     ]
)


# =============================================================== #
#                 Grid initialization variables
# =============================================================== #

# Inner cell faces.   Ni is the axial direction and Nj the radial direction.
Nt = 2000000
Ni = 50
Nj = 30

ASPECT_RATIO = Lz/Lr

Ni_regular = 20
Nj_regular = 20

dz = Lz / Ni_regular
dr = Lr / Nj_regular

z_coordinates_regular = np.linspace(-dz/2, Lz+dz/2, Ni_regular+2)
r_coordinates_regular = np.linspace(-dr/2, Lr+dr/2, Nj_regular+2)
Z_regular, R_regular = np.meshgrid(z_coordinates_regular, r_coordinates_regular, indexing="ij")


from magnetic_field import ElectroMagnetic


MagClass = ElectroMagnetic(dr=dr, dz=dz, Lr=Lr, Lz=Lz, Ni=Nj_regular, Nj=Ni_regular,
                           Coils=Coils, omega=omega, mu0=mu0, Ic=Ic, sigmaf=sigmaf)


# =============================================================== #
#                       Grid Initialization
# =============================================================== #

z_coordinates = np.linspace(-dz/2, Lz+dz/2, Ni+2)

# r_coordinates = np.array([
#     -0.0005, 0.0005,                    # ghost points
#     0.002, 0.0035, 0.005,              # coarse center
#     0.0065, 0.008, 0.0092, 0.0105,     # getting finer
#     0.0118, 0.0132, 0.0146, 0.016,     # transition
#     0.0172, 0.0184, 0.0196,            # fine near tip
#     0.0203, 0.0209, 0.0215, 0.022,     # finer
#     0.0224, 0.0228, 0.0232, 0.0236, 0.0239,
#     0.0241, 0.0243, 0.0245, 0.0247, 0.0248, 0.0249, 0.0251
# ])


# z_coordinates = np.array([
#     -0.001, 0.001,
#     0.0025,
#     0.0037,
#     0.007,
#     0.013,
#     0.018,
#     0.022,
#     0.026,
#     0.031,
#     0.035,
#     0.039,
#     0.042,
#     0.045,
#     0.047,
#     0.049,
#     0.05,
#     0.051,
#     0.053,
#     0.055,
#     0.059,
#     0.064,
#     0.069,
#     0.074,
#     0.08,
#     0.084,
#     0.089,
#     0.095,
#     0.101,
#     0.105,
#     0.109,
#     0.112,
#     0.115,
#     0.118,
#     0.121,
#     0.124,
#     0.127,
#     0.13,
#     0.134,
#     0.138,
#     0.142,
#     0.146,
#     0.15,
#     0.154,
#     0.158,
#     0.162,
#     0.168, 0.174, 0.182, 0.19, 0.199,
#     0.201
# ])


r_coordinates = np.array([
    -0.0005, 0.0005,
    0.002, 0.0035, 0.005,
    0.0065, 0.008, 0.0092, 0.0105,
    0.0118, 0.014, 0.016, 0.0172, 0.018,
    0.0185, 0.0193, 0.02,
    0.0207, 0.0212, 0.0216, 0.022,
    0.0224, 0.0228, 0.0232, 0.0234, 0.0236,
    0.0238, 0.0242, 0.0245, 0.0247, 0.0249, 0.0251
])

# r_coordinates = np.array([
#     -0.0005, 0.0005, 0.00125, 0.002, 0.00275, 0.0035, 0.00425, 0.005, 0.00575, 0.0065,
#     0.00725, 0.008, 0.0086, 0.0092, 0.00985, 0.0105, 0.01115, 0.0118, 0.0129, 0.014,
#     0.015, 0.016, 0.0166, 0.0172, 0.0176, 0.018, 0.01825, 0.0185, 0.0189, 0.0193,
#     0.01965, 0.02, 0.02035, 0.0207, 0.02095, 0.0212, 0.0214, 0.0216, 0.0218, 0.022,
#     0.0222, 0.0224, 0.0226, 0.0228, 0.023, 0.0232, 0.0233, 0.0234, 0.0235, 0.0236,
#     0.0237, 0.0238, 0.024, 0.0242, 0.02435, 0.0245, 0.0246, 0.0247, 0.0248, 0.0249,
#     0.025, 0.0251
# ])

z_coordinates_uz = 1/2*(z_coordinates[1:]+z_coordinates[:-1])   # Sits in the middle.
r_coordinates_ur = 1/2*(r_coordinates[1:]+r_coordinates[:-1])   # Sits in the middle.

Z, R = np.meshgrid(z_coordinates, r_coordinates, indexing="ij")

Zuz, Ruz = np.meshgrid(z_coordinates_uz, r_coordinates, indexing="ij")
Zur, Rur = np.meshgrid(z_coordinates, r_coordinates_ur, indexing="ij")

z_area = (
        np.ones_like(Z[1:, 1:-1]) * np.pi *
        ((Rur[1:, 1:])**2 -
         np.maximum.reduce([(Rur[1:, :-1]), np.zeros_like(Rur[1:, :-1])])**2)
)

r_area = np.ones_like(R[1:-1, 1:]) * (Zuz[1:, 1:]-Zuz[:-1, 1:]) * 2 * np.pi * (Rur[1:-1, :])

volume = (
        np.pi * ((Rur[1:-1, 1:])**2-(Rur[1:-1, :-1])**2) * (Zuz[1:, 1:-1]-Zuz[:-1, 1:-1])
)

# x_area, y_area and volume is defined only for the full cells.
z_area_uz = (
        np.ones_like(Zuz[1:, 1:-1]) * np.pi *
        ((Rur[1:-1, 1:])**2-np.maximum.reduce([(Rur[1:-1, :-1]), np.zeros_like(Rur[1:-1, :-1])])**2)
)
r_area_uz = np.ones_like(Ruz[1:-1, 1:]) * (Z[2:-1, 1:]-Z[1:-2, 1:]) * 2 * np.pi * (Rur[2:-1, :])

volume_uz = (
        np.pi * ((Rur[2:-1, 1:])**2-(Rur[1:-2, :-1])**2) * dz
)

# x_area, y_area and volume is defined only for the full cells.
z_area_ur = (
        np.ones_like(Zur[1:, 1:-1]) * np.pi *
        ((R[1:, 2:-1])**2-np.maximum.reduce([(R[1:, 1:-2]), np.zeros_like(R[1:, 1:-2])])**2)
)

r_area_ur = np.ones_like(Rur[1:-1, 1:]) * (Zuz[1:, 1:-1]-Zuz[:-1, 1:-1]) * 2 * np.pi * (R[1:-1, 1:-1])

volume_ur = (
        np.pi * ((R[1:-1, 2:-1])**2-(R[1:-1, 1:-2])**2) * (Zuz[1:, 2:-1]-Zuz[:-1, 2:-1])
)


Nj_sheath = 14  # 14    # in the middle of the sheath thingy.

Nj_safe = 13  # 13  # 15   # Die belangrikste parameter.

Nj_carrier = 2  # 2

Ni_regular_sheath = int((Ni+2)*Lz_sheath/Lz)

Ni_sheath = Ni_regular_sheath  # 16
Ni_carrier = Ni_regular_sheath  # 16

Lr_sheath = 18.8/1000
t_sheath = 2.2/1000

# Therefore inlet space is 21 mm

Lz_sheath = 50/1000
Lr_carrier = 3.7/1000        # 0.0037
t_carrier = 2/1000           # 0.002

# Therefore inside is at 1.7 mm

Lz_carrier = 50/1000


# =============================================================== #
#                       Initial Conditions:
# =============================================================== #

uz_prev = np.ones((Ni+1, Nj+2))*U_inlet_main*0
# uz_prev[:, int(np.ceil(Nj_sheath + (Nj - Nj_sheath) * 0.4)):] = U_sheath

uz_prev[:, 0] = -uz_prev[:, 1]
uz_prev[:, -1] = -uz_prev[:, -2]

ur_prev = np.zeros((Ni+2, Nj+1))

pr_prev = np.zeros((Ni+2, Nj+2))

T_prev = np.ones_like(pr_prev) * temp_boundary

rho_prev = rhof(T_prev)
p_prev = np.ones_like(pr_prev) * p_atm
