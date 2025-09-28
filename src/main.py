from simulation_file import simulation
from all_variables import r_coordinates_uz, z_coordinates, Lr, dr, U_inlet_main, Ni
import matplotlib.pyplot as plt
import numpy as np


uz_next, p_next, T_next = simulation()


def uz(r, uz_max):
    uz = uz_max*(1-(r/Lr)**2)
    return uz


# Plot setup
plt.figure(figsize=(8, 6))

# Line plot of the axial velocity profile
plt.plot(r_coordinates_uz[1:-1], uz_next[Ni-2, 1:-1], label='Numerical Result', color='blue', linewidth=2)

# Analytical or fitted profile (if applicable)
plt.scatter(r_coordinates_uz[1:-1], uz(r_coordinates_uz[1:-1], max(uz_next[Ni-2, 1:-1])),
            label='Analytical Fit', color='red', marker='o', s=40)

# Axis labels with units
plt.xlabel('Radial Position $r$ [m]', fontsize=14)
plt.ylabel('Axial Velocity $v_z$ [m/s]', fontsize=14)

# Title and legend
# plt.title('Radial Profile of Axial Velocity in Pipe Flow', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Axis limits (optional, adjust based on your data)
# plt.xlim(0, 0.025)
plt.ylim(0, 1.1 * np.max(uz_next[Ni-2, 1:-1]))

# Ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Layout and display
plt.tight_layout()
plt.savefig("axial_velocity_profile.png", dpi=300, bbox_inches='tight')  # you can also use .pdf, .svg, etc.

plt.show()


print(uz_next[Ni-2, 1:-1]-uz(r_coordinates_uz[1:-1], max(uz_next[Ni-2, 1:-1])))
print(np.average(np.abs(uz_next[Ni-2, 1:-1]-uz(r_coordinates_uz[1:-1],
                                               max(uz_next[Ni-2, 1:-1])))), max(uz_next[Ni-2, 1:-1]))

"""
For heat profile:

"""


def bulk_temperature(uz, T, r):
    integrand = uz * T * r
    weight = uz * r

    numerator = np.trapezoid(integrand, r)
    denominator = np.trapezoid(weight, r)

    return numerator / denominator


P = 2*np.pi*Lr
Nu = 3.66
k = 0.5
h = Nu*k/(2*Lr)
Cp = 518.8
m = 1.3906257032299385*np.pi*Lr**2*U_inlet_main


def LHS(T_mean, T_i):
    return (350*3-T_mean) / (350*3-T_i)


def RHS(z_coordinates):
    return np.exp(-P*z_coordinates/m/Cp*h)


T_mean = []

for i in range(Ni):
    T_mean.append(bulk_temperature((uz_next[i+1]+uz_next[i])/2, T_next[i+1], r_coordinates_uz))

T_mean = np.array(T_mean)

plt.figure(figsize=(8, 6))

# Plot LHS of the energy equation
plt.plot(z_coordinates[1:-1], LHS(T_mean, T_mean[0]),
         label='Numerical Results', color='blue', linewidth=2)

# Plot RHS (analytical or fitted function)
plt.scatter(z_coordinates[1:-1], RHS(z_coordinates[1:-1]),
            label='Analytical Fit', color='red', marker='o', s=40)

# Axis labels with units
plt.xlabel('Axial Position $z$ [m]', fontsize=14)
plt.ylabel('Dimensionless Temperature Profile', fontsize=14)

# Title and legend
# plt.title('Axial Variation of Energy Equation Terms', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Optional axis limits
# plt.xlim(0, z_coordinates[-1])
# plt.ylim(...)

# Ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Layout and save
plt.tight_layout()
plt.savefig("energy_balance_comparison.png", dpi=300, bbox_inches='tight')  # Optional

plt.show()


print(RHS(z_coordinates[1:-1])-LHS(T_mean, T_mean[0]))
print(np.average(np.abs(RHS(z_coordinates[1:-1])-LHS(T_mean, T_mean[0]))))
