import matplotlib.pyplot as plt
import numpy as np


ratio_plot = 0.8


def plot2(R, Z, field, title="Contour Plot"):
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(R, Z, field, levels=50, cmap="plasma")  # Contour plot

    # Add color bar
    cbar = plt.colorbar(contour)
    cbar.set_label("Field Value")

    # Labels and formatting
    plt.xlabel("r (m)")
    plt.ylabel("z (m)")
    plt.title(title, pad=20)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().set_aspect(1.05)  # Adjust aspect ratio

    plt.show()


def plot(X, Y, title="XY-plot", xlabel="x", ylabel="y"):
    plt.figure(figsize=(8, 6))

    plt.plot(X, Y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    return


def plot_contour(R, Z, field, title="Contour Plot", levels=10):
    plt.figure(figsize=(8, 6))

    # Create filled contours
    contour = plt.contourf(R, Z, field, levels=50, cmap="plasma")

    # Add contour lines (using the same levels as contourf)
    contour_lines = plt.contour(R, Z, field, levels=levels,
                                colors='black', linewidths=0.5, alpha=0.5)

    # Optionally add labels to the contour lines
    plt.clabel(contour_lines, inline=True, fontsize=2, fmt='%1.1f')

    # Add color bar
    cbar = plt.colorbar(contour)
    cbar.set_label("Field Value")

    # Labels and formatting
    plt.xlabel("r (m)")
    plt.ylabel("z (m)")
    plt.title(title, pad=20)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().set_aspect(ratio_plot)  # Adjust aspect ratio

    plt.show()


def plot_contour_save(R, Z, field, title="Contour Plot", levels=10, ratio_plot=ratio_plot, filename=None,
                      colourbar="Field Value"):
    """
    Plots a contour of the given field and optionally saves it to a PNG file.

    Parameters:
        R, Z        : 2D arrays for meshgrid coordinates (r, z)
        field       : 2D array of field values
        title       : Plot title
        levels      : Number of contour lines
        ratio_plot  : Aspect ratio (e.g., 1.0 for square)
        filename    : If provided, saves the figure as a PNG
    """
    plt.figure(figsize=(8, 6))

    # Filled contour
    contour = plt.contourf(R, Z, field, levels=100, cmap="plasma")

    # Contour lines
    contour_lines = plt.contour(R, Z, field, levels=levels, colors='black', linewidths=0.6, alpha=0.7)
    plt.clabel(contour_lines, inline=True, fontsize=6, fmt='%1.1f')

    # Color bar
    cbar = plt.colorbar(contour, pad=0.01)
    cbar.set_label(colourbar, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Labels, title, and grid
    plt.xlabel("Radial Position $r$ (m)", fontsize=10)
    plt.ylabel("Axial Position $z$ (m)", fontsize=10)
    plt.title(title, fontsize=12, pad=15)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    plt.gca().set_aspect(ratio_plot)

    # Tight layout for clean figure
    plt.tight_layout()

    # Save figure if a filename is given
    if filename:
        plt.savefig(filename+'.png', format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")

    plt.show()

    return


def plot_3d(R, Z, field, title="3D Surface Plot"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create surface plot
    surf = ax.plot_surface(R, Z, field, cmap="plasma", edgecolor='none')

    # Add color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label("Field Value")

    # Labels and formatting
    ax.set_xlabel("r (m)")
    ax.set_ylabel("z (m)")
    ax.set_zlabel("Field Value")
    ax.set_title(title, pad=20)

    # Show plot
    plt.show()


def quiver_plot(u, v, R, Z, scale=100):
    import matplotlib.pyplot as plt

    # Create a quiver plot
    plt.figure(figsize=(8, 6))
    plt.quiver(Z, R, u, v, scale=scale, pivot="middle", color="blue")

    # Labels and formatting
    plt.xlabel("Axial Direction (Z)")
    plt.ylabel("Radial Direction (R)")
    plt.title("Velocity Field")
    plt.grid()
    plt.axhline(y=0.025, color="red", linestyle="--", linewidth=2, label="x = 0.025")

    plt.axis("equal")  # Ensures aspect ratio is correct

    plt.show()


def plot_and_save(x, y_line, x_scatter, y_scatter, xlabel, ylabel, filename, line_color, scatter_color):
    plt.figure()
    plt.plot(x, y_line, color=line_color, label='Simulation')
    plt.scatter(x_scatter, y_scatter, color=scatter_color, edgecolor='black', zorder=5,
                label='Bernardi et al. (2003)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    plt.close()


def reflect_vertical(data, sign=1):
    """
    Reflects a 2D array vertically along its second axis (i.e., left-right),
    then stacks the reflected part on top of the original data.

    Args:
        data (2D np.ndarray): array to reflect.
        sign (int): +1 for even symmetry (mirror), -1 for odd symmetry (antisymmetric).

    Returns:
        np.ndarray: vertically reflected array.
    """
    return np.concatenate((sign * np.flip(data, axis=0), data), axis=0) # np.vstack([sign * data[::-1, :], data])
