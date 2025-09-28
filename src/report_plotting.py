from all_variables import *


import matplotlib.pyplot as plt
import numpy as np

# === Base grid plot ===
fig, ax = plt.subplots(figsize=(7, 6))

# Plot the full grid in light grey
ax.pcolormesh(Z, R, np.zeros_like(Z),
              edgecolors='lightgrey', linewidth=0.5, facecolors='none')

# === Highlight sheath region ===
# Axial extent
z_sheath_min = 0
z_sheath_max = Lz_sheath
# Radial extent
r_sheath_min = Lr_sheath - t_sheath
r_sheath_max = Lr_sheath

ax.fill_betweenx(
    [r_sheath_min, r_sheath_max],
    z_sheath_min, z_sheath_max,
    color='tab:blue', alpha=0.2, label='Sheath Region'
)

# === Highlight carrier region ===
z_carrier_min = 0
z_carrier_max = Lz_carrier
r_carrier_min = Lr_carrier - t_carrier
r_carrier_max = Lr_carrier

ax.fill_betweenx(
    [r_carrier_min, r_carrier_max],
    z_carrier_min, z_carrier_max,
    color='tab:orange', alpha=0.2, label='Carrier Region'
)

# === Labels and formatting ===
ax.set_xlabel("z (m)", fontsize=9)
ax.set_ylabel("r (m)", fontsize=9)
# ax.set_title("Non-Uniform Computational Grid with Sheath & Carrier Regions", fontsize=10)
ax.legend(loc='upper right', fontsize=10)
ax.set_aspect('equal')
ax.set_xlim([0, Lz])
ax.set_ylim([0, r_coordinates[-1]])
ax.grid(False)

plt.tight_layout()
plt.savefig("grid_with_sheath_carrier.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure()
plt.plot()
plt.show()

# Assuming Z, R, Lz_sheath, Lr_sheath, t_sheath, Lz_carrier, Lr_carrier, t_carrier, Lz, r_coordinates are defined
# If not, you would need to provide their values for a complete example

# === Base grid plot ===
fig, ax = plt.subplots(figsize=(6, 7))  # Adjusted figure size for vertical orientation

# Plot the full grid in light grey
# Swap Z and R for pcolormesh to reflect r on x-axis and z on y-axis
ax.pcolormesh(R, Z, np.zeros_like(Z),
              edgecolors='lightgrey', linewidth=0.5, facecolors='none')

# === Highlight sheath region ===
# Axial extent (now on y-axis)
z_sheath_min = 0
z_sheath_max = Lz_sheath
# Radial extent (now on x-axis)
r_sheath_min = Lr_sheath - t_sheath
r_sheath_max = Lr_sheath

# Use fill_between (horizontal fill) instead of fill_betweenx
ax.fill_between(
    [r_sheath_min, r_sheath_max],
    z_sheath_min, z_sheath_max,
    color='tab:blue', alpha=0.2, label='Sheath Region'
)

# === Highlight carrier region ===
z_carrier_min = 0
z_carrier_max = Lz_carrier
r_carrier_min = Lr_carrier - t_carrier
r_carrier_max = Lr_carrier

ax.fill_between(
    [r_carrier_min, r_carrier_max],
    z_carrier_min, z_carrier_max,
    color='tab:orange', alpha=0.2, label='Carrier Region'
)

# === Labels and formatting ===
ax.set_xlabel("r (m)", fontsize=9)  # r is now on x-axis
ax.set_ylabel("z (m)", fontsize=9)  # z is now on y-axis
ax.legend(loc='upper right', fontsize=10)
ax.set_aspect('equal')
ax.set_xlim([0, r_coordinates[-1]])  # r limits on x-axis
ax.set_ylim([0, Lz])  # z limits on y-axis
ax.grid(False)

plt.tight_layout()
plt.savefig("grid_with_sheath_carrier_swapped_axes.png", dpi=300, bbox_inches="tight")
plt.show()

# Second plot (empty plot from original code)
plt.figure()
plt.plot()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Data
power = np.array([5, 7.5, 10, 12.5, 15, 17.5])
max_temp = np.array([10214, 10655, 10947, 11108, 11230, 11317])
current = currentf(power)

# Fit a linear (degree 1) and quadratic (degree 2) polynomial
linear_coeffs = np.polyfit(power, max_temp, 1)
quadratic_coeffs = np.polyfit(power, max_temp, 2)

# Create polynomial functions
linear_fit = np.poly1d(linear_coeffs)
quadratic_fit = np.poly1d(quadratic_coeffs)

# Generate smooth points for plotting the fitted curves
power_smooth = np.linspace(min(power), max(power), 100)
linear_temp = linear_fit(power_smooth)
quadratic_temp = quadratic_fit(power_smooth)

# Calculate R² for both fits
linear_pred = linear_fit(power)
quadratic_pred = quadratic_fit(power)

# Set a professional style
plt.style.use('seaborn-v0_8')

# Create figure
plt.figure(figsize=(8, 6), dpi=100)

# Plot data points
plt.scatter(current, max_temp, label='Simulation Data', color='crimson',
            s=80, marker='o', edgecolors='black', alpha=0.8)

# Plot fitted curves
# plt.plot(power_smooth, linear_temp, label=f'Linear Fit',
#          color='navy', linestyle='--', linewidth=2)
# plt.plot(power_smooth, quadratic_temp, label=f'Quadratic Fit',
#          color='green', linestyle='-', linewidth=2)

# Customize axes
plt.xlabel('Current (A)', fontsize=14, labelpad=10)
plt.ylabel('Maximum Temperature (K)', fontsize=14, labelpad=10)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(fontsize=12, loc='best')

# Adjust tick parameters
plt.tick_params(axis='both', which='major', labelsize=12, direction='in',
                length=5, width=1)

# Ensure tight layout
plt.tight_layout()
plt.savefig("Temperature and power.png", dpi=1000)
# Show plot
plt.show()

# Print fit parameters
print(f"Linear Fit: T = {linear_coeffs[0]:.1f}P + {linear_coeffs[1]:.1f}")
print(f"Quadratic Fit: T = {quadratic_coeffs[0]:.1f}P² + {quadratic_coeffs[1]:.1f}P + {quadratic_coeffs[2]:.1f}")


