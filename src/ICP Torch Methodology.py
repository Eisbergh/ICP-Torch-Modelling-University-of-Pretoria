import numpy as np

from all_variables import *
from base_methods import save, better_interpolator, extract, extract_conditions
from simulation_file import simulation
from temperature_field import temp
from flow_equations import temperature_boundaries
from plotting import plot_contour

# T_prev_regular = np.zeros_like(T_prev) + temp_boundary
# temperature_boundaries(T_prev_regular)

# uz_prev = np.load('Storage/uz_good.npy')
# T_prev_regular = temp(Ni+2, Nj+2, np.load('Storage/T_good.npy'))
# ur_prev = np.load('Storage/ur_good.npy')

# T_prev = better_interpolator(R_original=R_regular, Z_original=Z_regular,
#                              Z_new=Z, R_new=R, Field_Original=T_prev_regular)
name = "[kW=5,Q_in=3, Q_sheath=31,Q_carrier=1]"
T_prev, uz_prev, ur_prev = extract_conditions(R=R, Z=Z, Rur=Rur, Zur=Zur, Ruz=Ruz, Zuz=Zuz, name=name, num=4)
# plot_contour(Rur, Zur, ur_prev)
# T_prev, uz_prev, ur_prev = extract(R=R, Z=Z, Rur=Rur, Zur=Zur, Ruz=Ruz, Zuz=Zuz)
#
# T_prev = np.zeros_like((T_prev))+temp_boundary
# uz_prev = np.zeros_like(uz_prev)
# ur_prev = np.zeros_like(ur_prev)

uz_next, ur_next, p_next, T_next = simulation(uz_prev, ur_prev, p_prev, T_prev, rhof(T_prev))


