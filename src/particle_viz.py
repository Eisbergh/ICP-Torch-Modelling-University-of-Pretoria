# particle_viz.py
# Nice plots for single-particle RF-ICP torch simulations.
# Requirements: numpy, matplotlib, (optional) cmasher for perceptually uniform colormaps.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import cmasher as cmr
    CMR_AVAILABLE = True
except Exception:
    CMR_AVAILABLE = False


# ---------------------------
# Data container for one run
# ---------------------------

@dataclass
class ParticleHistory:
    """Container for a single particle's time history."""
    t: np.ndarray             # shape (N,)
    z: np.ndarray             # axial positions (m), shape (N,)
    r: np.ndarray             # radial positions (m), shape (N,)
    uz: np.ndarray            # particle axial velocity (m/s), shape (N,)
    ur: np.ndarray            # particle radial velocity (m/s), shape (N,)
    Tp: np.ndarray            # particle temperature (K), shape (N,)
    dp: np.ndarray            # particle diameter (m), shape (N,)
    spheroid: np.ndarray
    x: Optional[np.ndarray] = None  # melt fraction (0–1), shape (N,) if available

    def first_melt_time(self, Tmp: float, tol: float = 1e-2) -> Optional[float]:
        if self.Tp is None:
            return None
        idx = np.where(self.Tp >= Tmp - tol)[0]
        return float(self.t[idx[0]]) if idx.size else None

    def fully_molten_time(self) -> Optional[float]:
        if self.x is None:
            return None
        idx = np.where(self.x >= 1.0)[0]
        return float(self.t[idx[0]]) if idx.size else None


# -----------------------------------
# Internal helpers for colored lines
# -----------------------------------

def _colored_path(ax, x, y, c_vals, cmap='viridis', linewidth=2.0, alpha=1.0, zorder=10):
    """Draw a line whose color varies along its length according to c_vals."""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=Normalize(vmin=float(np.min(c_vals)),
                                                        vmax=float(np.max(c_vals))))
    lc.set_array(c_vals[:-1])
    lc.set_linewidth(linewidth)
    lc.set_alpha(alpha)
    lc.set_zorder(zorder)
    ax.add_collection(lc)
    return lc


def _set_style(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, aspect=None):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title)
    if xlim:   ax.set_xlim(*xlim)
    if ylim:   ax.set_ylim(*ylim)
    if aspect: ax.set_aspect(aspect)
    ax.grid(True, alpha=0.25)


# ---------------------------------------
# 1) Trajectory over field visualization
# ---------------------------------------

def plot_trajectory_over_fields(
    hist: ParticleHistory,
    R: np.ndarray,
    Z: np.ndarray,
    T: Optional[np.ndarray] = None,
    uz_field: Optional[np.ndarray] = None,
    ur_field: Optional[np.ndarray] = None,
    color_by: str = "time",     # "time" or "Tp"
    streamline_density: float = 1.3,
    figscale: float = 20,
    dpi: int = 3000,
    savepath: Optional[str] = None,
    title: Optional[str] = None,
    zlim: Optional[Tuple[float, float]] = (0, 0.2),
    rlim: Optional[Tuple[float, float]] = (0, 0.025),
):
    """
    Draw the particle trajectory on top of optional temperature contours and velocity streamlines.

    Parameters
    ----------
    hist : ParticleHistory
        Particle time history.
    R, Z : 2D arrays
        Meshgrids (r, z) for fields.
    T : 2D array, optional
        Temperature field (K).
    uz_field, ur_field : 2D arrays, optional
        Gas-phase velocities (m/s) on the same grid as T.
    color_by : {"time", "Tp"}
        Color the trajectory by time or particle temperature.
    """

    # Figure & axis
    w = 6.5 * figscale
    h = 1.8 * figscale
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)

    # Background temperature
    if T is not None:
        cmap_bg = cmr.ember if CMR_AVAILABLE else 'viridis'
        im = ax.pcolormesh(Z, R, T/1000, shading='auto', cmap=cmap_bg, alpha=0.5)
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label("T (*1000 K)")

    # Streamlines
    if (uz_field is not None) and (ur_field is not None):
        try:
            strm = ax.streamplot(Z, R, uz_field, ur_field, density=streamline_density, linewidth=0.5, arrowsize=0.3)
        except Exception:
            # For non-rectangular meshes, try transposes
            strm = ax.streamplot(Z.T, R.T, uz_field.T, ur_field.T,
                                 density=streamline_density, linewidth=0.5, arrowsize=0.3)

    # Trajectory colored by "time" or "Tp"
    if color_by.lower() == "tp":
        cmap_line = cmr.flamingo if CMR_AVAILABLE else 'plasma'
        cvals = hist.Tp/1000
        clabel = "$T_p$ (*1000 K)"
    else:
        cmap_line = cmr.pride if CMR_AVAILABLE else 'viridis'
        cvals = hist.t
        clabel = "Time, t (s)"

    lc = _colored_path(ax, hist.z, hist.r, cvals, cmap=cmap_line, linewidth=2.2, alpha=0.95)
    cbar2 = fig.colorbar(lc, ax=ax, pad=0.01)
    cbar2.set_label(clabel)

    # Start/end markers
    ax.scatter(hist.z[0], hist.r[0], s=30, marker='o', facecolor='white', edgecolor='k', zorder=20, label='Start')
    ax.scatter(hist.z[-1], hist.r[-1], s=30, marker='X', facecolor='k', edgecolor='white', zorder=20, label='End')

    # --- Custom reference lines ---
    for y in [0.0188, -0.0188]:
        ax.hlines(y, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.hlines(-0.003, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.hlines(0.003, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=1.0, alpha=0.8)

    ax.vlines(x=0.05, ymin=-0.025, ymax=-0.0188, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.vlines(x=0.05, ymin=-0.003, ymax=0.003, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.vlines(x=0.05, ymin=0.0188, ymax=0.025, color='black', linestyle='--', linewidth=1.0, alpha=0.8)

    # Axes, limits, labels
    _set_style(ax,
               xlabel="Axial position, z (m)",
               ylabel="Radial position, r (m)",
               title=title or "Particle trajectory over fields",
               xlim=zlim or (float(np.min(Z)), float(np.max(Z))),
               ylim=rlim or (float(np.min(R)), float(np.max(R))),
               aspect='auto')

    # ax.legend(loc='upper right', fontsize=9, frameon=True)

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
    return fig, ax


# ---------------------------------------
# 2) Summary panels for one trajectory
# ---------------------------------------

def plot_summary_panels(
    hist: ParticleHistory,
    Tmp: Optional[float] = None,
    Tbp: Optional[float] = None,
    dpi: int = 400,
    savepath: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Make a 3-panel figure: (a) z(t), r(t); (b) velocities uz, ur; (c) Tp and dp vs time.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 3.3), dpi=dpi)

    # # (a) Positions
    # ax = axes[0]
    # ax.plot(hist.t, hist.z, label='z(t)', lw=2.0)
    # ax.plot(hist.t, hist.r, label='r(t)', lw=2.0)
    # _set_style(ax, xlabel='Time, t (s)', ylabel='Position (m)', title='Positions vs time')
    # ax.legend()
    #
    # # (b) Velocities
    # ax = axes[1]
    # ax.plot(hist.t, hist.uz, label='u_z(t)', lw=2.0)
    # ax.plot(hist.t, hist.ur, label='u_r(t)', lw=2.0)
    # _set_style(ax, xlabel='Time, t (s)', ylabel='Velocity (m/s)', title='Velocities vs time')
    # ax.legend()

    # (c) Temperature & diameter
    ax2 = ax.twinx()
    line_Tp, = ax.plot(hist.t, hist.Tp, lw=2.0, label='T_p (K)')
    line_dp, = ax2.plot(hist.t, hist.dp * 1e6, lw=2.0, ls='--', label='d_p (µm)')

    hlines = []
    hlabels = []
    if Tmp is not None:
        hl_tmp = ax.axhline(Tmp, color='k', lw=1.1, alpha=0.5, label='T_m (K)')
        hlines.append(hl_tmp)
        hlabels.append('T_m (K)')
    if Tbp is not None:
        hl_tbp = ax.axhline(Tbp, color='k', lw=1.1, alpha=0.5, ls=':', label='T_b (K)')
        hlines.append(hl_tbp)
        hlabels.append('T_b (K)')

    ax.set_xlabel('Time, t (s)')
    ax.set_ylabel('Temperature, $T_p$ (K)')
    ax2.set_ylabel('Diameter, $d_p$ (µm)')

    # Legends merged
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2 + hlines, labels1 + labels2, loc='best')

    # fig.suptitle(title or 'Particle time histories', y=1.02)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=1000)
    return fig, ax


# ------------------------------------------------
# 3) Small helper to bundle your Particle object
# ------------------------------------------------

def history_from_particle(particle) -> ParticleHistory:
    """Build ParticleHistory from your Particle instance after a run."""
    # Ensure arrays
    t  = np.asarray(particle.t_history, dtype=float)
    z  = np.asarray(particle.z_history, dtype=float)
    r  = np.asarray(particle.r_history, dtype=float)
    uz = np.asarray(particle.uzp_history, dtype=float)
    ur = np.asarray(particle.urp_history, dtype=float)
    Tp = np.asarray(particle.Tp_history, dtype=float)
    dp = np.asarray(particle.dp_history, dtype=float)
    x  = np.asarray(particle.x_history, dtype=float) if getattr(particle, "x_history", None) is not None else None
    spheroid = np.asarray(particle.spheroid_history, dtype=float)
    return ParticleHistory(t=t, z=z, r=r, uz=uz, ur=ur, Tp=Tp, dp=dp, x=x, spheroid=spheroid)


# ---- overlay_summary.py ----
import os
import numpy as np
import matplotlib.pyplot as plt


def overlay_summary_panels(
    histories, labels, Tmp=None, Tbp=None, dpi=300, figsize=(12, 3.6),
    savepath=None, title="Overlay: particle time histories by initial size"
):
    """
    Make a single 3-panel figure overlaying multiple particle runs.
      (a) z(t) and r(t)
      (b) u_z(t) and u_r(t)
      (c) T_p(t) and d_p(t) [right axis]
    histories : list[ParticleHistory]
    labels    : list[str]  (e.g., ["20 µm", "40 µm", ...])
    """
    n = len(histories)
    colors = plt.cm.viridis(np.linspace(0, 1, n))  # distinct colors

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    # (a) positions
    ax = axes[0]
    for h, lab, c in zip(histories, labels, colors):
        ax.plot(h.t, h.z, lw=2.0, color=c, alpha=0.95)
        ax.plot(h.t, h.r, lw=1.4, color=c, alpha=0.6, ls='--')
    _set_style(ax, xlabel='Time, t (s)', ylabel='Position (m)', title='Positions vs time')

    # (b) velocities
    ax = axes[1]
    for h, lab, c in zip(histories, labels, colors):
        ax.plot(h.t, h.uz, lw=2.0, color=c, alpha=0.95)
        ax.plot(h.t, h.ur, lw=1.4, color=c, alpha=0.6, ls='--')
    _set_style(ax, xlabel='Time, t (s)', ylabel='Velocity (m/s)', title='Velocities vs time')

    # (c) temperature & diameter
    ax = axes[2]
    ax2 = ax.twinx()
    for h, lab, c in zip(histories, labels, colors):
        ax.plot(h.t, h.Tp, lw=2.0, color=c, label=lab)
        ax2.plot(h.t, h.dp * 1e6, lw=1.6, color=c, ls='--', alpha=0.8)
    if Tmp is not None:
        ax.axhline(Tmp, color='k', lw=1.0, alpha=0.5)
    if Tbp is not None:
        ax.axhline(Tbp, color='k', lw=1.0, alpha=0.5, ls=':')

    ax.set_xlabel('Time, t (s)')
    ax.set_ylabel('Temperature, $T_p$ (K)')
    ax2.set_ylabel('Diameter, $d_p$ (µm)')
    ax.set_title('Thermal state and size')

    # Build a single legend (use panel (c) labels)
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper center', ncol=min(len(labels_), 6), frameon=True, fontsize=9)

    fig.suptitle(title, y=1.08)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
    return fig, axes


def overlay_summary_panels_for_sizes(
    simulate_particle, R_regular, Z_regular, Tmp, Tbp, microns=(20, 40, 60, 80, 100, 1000),
    uz0=1.8, ur0=0.0, ri=3.7/1000/3, zi=0.051, Tpi=350.0,
    outdir="figs_overlays", dpi=300
):
    """
    Runs each size once using your simulate_particle(), collects histories, and makes a single overlay figure.
    """
    os.makedirs(outdir, exist_ok=True)

    histories, labels = [], []
    for um in microns:
        dpi_m = float(um) * 1e-6
        particle, T, uz, ur = simulate_particle(
            dpi=dpi_m, uzpi=uz0, urpi=ur0, ri=ri, zi=zi, Tpi=Tpi, R=R_regular, Z=Z_regular
        )
        histories.append(history_from_particle(particle))
        labels.append(f"{um} µm")

        # --------- make overlay figure ----------
    colors = plt.cm.viridis(np.linspace(0, 1, len(histories)))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=dpi)

    # (a) Positions
    ax = axes[0]
    for h, lab, c in zip(histories, labels, colors):
        ax.plot(h.t, h.z, lw=2, color=c, label=lab)
        ax.plot(h.t, h.r, lw=1.2, ls='--', color=c)
    ax.set_xlabel("Time, t (s)")
    ax.set_ylabel("Position (m)")
    ax.set_title("Positions vs time")
    ax.grid(alpha=0.3)

    # (b) Velocities
    ax = axes[1]
    for h, lab, c in zip(histories, labels, colors):
        ax.plot(h.t, h.uz, lw=2, color=c)
        ax.plot(h.t, h.ur, lw=1.2, ls='--', color=c)
    ax.set_xlabel("Time, t (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Velocities vs time")
    ax.grid(alpha=0.3)

    # (c) Temperature & diameter
    ax = axes[2]
    ax2 = ax.twinx()
    for h, lab, c in zip(histories, labels, colors):
        ax.plot(h.t, h.Tp, lw=2, color=c, label=lab)
        ax2.plot(h.t, h.dp * 1e6, lw=1.2, ls='--', color=c)
    if Tmp is not None:
        ax.axhline(Tmp, color='k', lw=1.0, alpha=0.5)
    if Tbp is not None:
        ax.axhline(Tbp, color='k', lw=1.0, alpha=0.5, ls=':')
    ax.set_xlabel("Time, t (s)")
    ax.set_ylabel("Temperature, $T_p$ (K)")
    ax2.set_ylabel("Diameter, $d_p$ (µm)")
    ax.set_title("Thermal state and size")
    ax.grid(alpha=0.3)

    # Legend across top (based on sizes)
    fig.legend(labels, loc='upper center', ncol=min(len(labels), 6), frameon=True, fontsize=9)

    fig.suptitle("Overlay: particle time histories by size", y=1.05)
    fig.tight_layout()

    # Save
    savepath = os.path.join(outdir, "overlay_summary_by_size.png")
    fig.savefig("hello.png", bbox_inches='tight')
    plt.close(fig)

    print(f"[OK] Wrote {savepath}")

    return savepath


from typing import Optional, Sequence, Tuple, Union, Iterable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

def plot_trajectory_over_fields2(
    hists: Union["ParticleHistory", Sequence["ParticleHistory"]],
    R: np.ndarray,
    Z: np.ndarray,
    T: Optional[np.ndarray] = None,
    uz_field: Optional[np.ndarray] = None,
    ur_field: Optional[np.ndarray] = None,
    color_by: str = "time",       # "time" or "Tp"
    streamline_density: float = 1.3,
    figscale: float = 3,
    dpi: int = 10,
    savepath: Optional[str] = True,
    title: Optional[str] = None,
    zlim: Optional[Tuple[float, float]] = (0, 0.2),
    rlim: Optional[Tuple[float, float]] = (0, 0.025),
    labels: Optional[Sequence[str]] = None,
    normalize_across: bool = True,   # share a common color scale across trajectories
):
    """
    Draw one or more particle trajectories on top of optional temperature contours and velocity streamlines.

    Parameters
    ----------
    hists : ParticleHistory or sequence[ParticleHistory]
        One or many time histories to overlay.
    color_by : {"time", "Tp"}
        Color the trajectory by time or particle temperature.
    labels : optional sequence[str]
        Legend labels for each trajectory (start/end markers).
    normalize_across : bool
        If True, all trajectories share a common colorbar scale for `color_by`.
        :param savepath:
    """

    # --- Normalize input to a list of histories
    if isinstance(hists, Iterable) and not hasattr(hists, "t"):
        histories = list(hists)
    else:
        histories = [hists]
    n = len(histories)
    if labels is None:
        labels = [f"traj {i+1}" for i in range(n)]

    # --- Choose colormap and compute normalization
    if color_by.lower() == "tp":
        cmap_line = cmr.flamingo if CMR_AVAILABLE else "plasma"
        get_vals = lambda h: h.Tp / 1000.0
        clabel = "$T_p$ (*1000 K)"
    else:
        cmap_line = cmr.pride if CMR_AVAILABLE else "viridis"
        get_vals = lambda h: h.t
        clabel = "Time, t (s)"

    all_vals = np.concatenate([get_vals(h) for h in histories])
    if normalize_across:
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
    else:
        # fall back to per-trajectory normalization; colorbar will still reflect global (all) range
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
    norm = Normalize(vmin=vmin, vmax=vmax)

    # --- Figure & axes
    w = 6.5 * figscale
    h = 1.8 * figscale
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)

    # --- Background temperature
    if T is not None:
        cmap_bg = cmr.ember if CMR_AVAILABLE else "viridis"
        im = ax.pcolormesh(Z, R, T / 1000, shading="auto", cmap=cmap_bg, alpha=0.5)
        cbar_bg = fig.colorbar(im, ax=ax, pad=0.01)
        cbar_bg.set_label("T (*1000 K)")

    # --- Streamlines
    if (uz_field is not None) and (ur_field is not None):
        try:
            ax.streamplot(Z, R, uz_field, ur_field, density=streamline_density, linewidth=0.5, arrowsize=0.3)
        except Exception:
            ax.streamplot(Z.T, R.T, uz_field.T, ur_field.T, density=streamline_density, linewidth=0.5, arrowsize=0.3)

    # --- Trajectories (each colored by `color_by`)
    start_handles, end_handles = [], []
    for i, h in enumerate(histories):
        cvals = get_vals(h)
        # If your _colored_path supports `norm=`, pass it; if not, it simply ignores.
        lc = _colored_path(ax, h.z, h.r, cvals, cmap=cmap_line, linewidth=2.2, alpha=0.95)
        # Start/end markers (used for legend labels)
        start = ax.scatter(h.z[0], h.r[0], s=30, marker='o', facecolor='white', edgecolor='k', zorder=20)
        end = ax.scatter(h.z[-1], h.r[-1], s=30, marker='X', facecolor='k', edgecolor='white', zorder=20)
        start_handles.append(start)
        end_handles.append(end)

    # Shared colorbar for all trajectories
    sm = plt.cm.ScalarMappable(cmap=cmap_line, norm=norm)
    sm.set_array([])  # dummy for colorbar
    cbar2 = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar2.set_label(clabel)

    # --- Custom reference lines (kept as-is)
    for y in [0.0188, -0.0188]:
        ax.hlines(y, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.hlines(-0.003, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.hlines(0.003, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.vlines(x=0.05, ymin=-0.025, ymax=-0.0188, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.vlines(x=0.05, ymin=-0.003,  ymax=0.003,  color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.vlines(x=0.05, ymin=0.0188,   ymax=0.025, color='black', linestyle='--', linewidth=1.0, alpha=0.8)

    # --- Axes, limits, labels
    _set_style(ax,
               xlabel="Axial position, z (m)",
               ylabel="Radial position, r (m)",
               title=title or "Particle trajectories over fields",
               xlim=zlim or (float(np.min(Z)), float(np.max(Z))),
               ylim=rlim or (float(np.min(R)), float(np.max(R))),
               aspect='auto')

    # Legend: one label per trajectory (use start markers as proxies)
    legend_handles = [Line2D([], [], marker='o', linestyle='None', markerfacecolor='white', markeredgecolor='k')]
    legend_labels = ["Start"]
    legend_handles.append(Line2D([], [], marker='X', linestyle='None', markerfacecolor='k', markeredgecolor='white'))
    legend_labels.append("End")

    # Also attach trajectory-specific labels mapped to start markers
    for i, sh in enumerate(start_handles):
        sh.set_label(labels[i])
    ax.legend(loc='upper right', fontsize=9, frameon=True)

    plt.tight_layout()
    if savepath:
        fig.savefig("summary_trajectories.png", bbox_inches='tight', dpi=1000)
    return fig, ax


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from typing import Optional, Sequence, Tuple, Union, Iterable

# distinct, readable marker shapes to cycle through
_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]


def plot_trajectory_over_fields3(
    hists: Union["ParticleHistory", Sequence["ParticleHistory"]],
    R: np.ndarray,
    Z: np.ndarray,
    T: Optional[np.ndarray] = None,
    uz_field: Optional[np.ndarray] = None,
    ur_field: Optional[np.ndarray] = None,
    color_by: str = "time",       # "time" or "Tp"
    streamline_density: float = 1.3,
    figscale: float = 4,
    dpi: int = 1000,
    savepath: Optional[str] = None,
    title: Optional[str] = None,
    zlim: Optional[Tuple[float, float]] = (0, 0.2),
    rlim: Optional[Tuple[float, float]] = (0, 0.025),

    # ---- NEW ID/legend helpers ----
    labels: Optional[Sequence[str]] = None,
    place_end_labels: bool = True,
    end_label_offset: Tuple[float, float] = (0.002, 0.0),   # (Δz, Δr)
    marker_every: Optional[int] = 100,    # place small markers every N points; None to disable
    marker_size: float = 16.0,
    legend_loc: str = "center right",
):
    """
    Draw one or more particle trajectories on top of optional temperature contours and velocity streamlines,
    with clear per-trajectory identification.
    """
    # --- Normalize input
    if isinstance(hists, Iterable) and not hasattr(hists, "t"):
        histories = list(hists)
    else:
        histories = [hists]
    n = len(histories)
    if labels is None:
        labels = [f"traj {i+1}" for i in range(n)]

    # --- Choose colormap and normalization for color-by
    if color_by.lower() == "tp":
        cmap_line = cmr.flamingo if CMR_AVAILABLE else "plasma"
        get_vals = lambda h: h.Tp / 1000.0
        clabel = "$T_p$ (*1000 K)"
    else:
        cmap_line = cmr.pride if CMR_AVAILABLE else "viridis"
        get_vals = lambda h: h.t
        clabel = "Time, t (s)"

    all_vals = np.concatenate([get_vals(h) for h in histories])
    norm = Normalize(vmin=float(all_vals.min()), vmax=float(all_vals.max()))

    # --- Size (guard against gigantic figures)
    def _cap_pixels(w_in, h_in, dpi, max_megapixels=20):
        pixels = w_in * h_in * dpi * dpi
        mx = max_megapixels * 1e6
        if pixels <= mx: return w_in, h_in, dpi
        s = (mx / pixels) ** 0.5
        return w_in*s, h_in*s, int(dpi*s)

    w = 5.5 * figscale
    h = 1.8 * figscale
    w, h, dpi = _cap_pixels(w, h, dpi, max_megapixels=20)

    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)

    # --- Background temperature
    if T is not None:
        cmap_bg = cmr.ember if CMR_AVAILABLE else "viridis"
        im = ax.pcolormesh(Z, R, T/1000, shading="auto", cmap=cmap_bg, alpha=0.5)
        # cbar_bg = fig.colorbar(im, ax=ax, pad=0.0000001); cbar_bg.set_label("T (*1000 K)")

    # --- Streamlines
    if (uz_field is not None) and (ur_field is not None):
        try:
            ax.streamplot(Z, R, uz_field, ur_field, density=streamline_density, linewidth=0.5, arrowsize=0.3)
        except Exception:
            ax.streamplot(Z.T, R.T, uz_field.T, ur_field.T, density=streamline_density, linewidth=0.5, arrowsize=0.3)

    # --- Trajectories
    proxy_handles = []  # for legend (marker shape per traj)
    proxy_labels  = []

    # for i, (h, lab) in enumerate(zip(histories, labels)):
    #     cvals = get_vals(h)
    #     # Colored path (assumes your _colored_path supports norm=; if not, it ignores)
    #     lc = _colored_path(ax, h.z, h.r, cvals, cmap=cmap_line, linewidth=2.2, alpha=0.95)
    #
    #     # Small checkpoint markers along the path with a distinct marker symbol per trajectory
    #     if marker_every is not None and marker_every > 0:
    #         mk = _MARKERS[i % len(_MARKERS)]
    #         idx = np.arange(0, len(h.z), marker_every)
    #         ax.scatter(h.z[idx], h.r[idx],
    #                    s=marker_size, marker=mk, facecolor="none", edgecolor="k",
    #                    linewidths=0.9, zorder=15)
    #
    #         # Add a proxy handle for the legend
    #         proxy_handles.append(Line2D([], [], marker=mk, linestyle="None",
    #                                     markerfacecolor="none", markeredgecolor="k",
    #                                     markersize=max(marker_size**0.5, 5)))
    #         # proxy_labels.append(lab)
    #
    #     # Start & end markers
    #     ax.scatter(h.z[0],  h.r[0],  s=28, marker='o', facecolor='white', edgecolor='k', zorder=20)
    #     ax.scatter(h.z[-1], h.r[-1], s=36, marker='X', facecolor='k',     edgecolor='white', zorder=20)
    #
    #     # Text label at the end (most readable)
    #     if place_end_labels:
    #         txt = ax.text(h.z[-1] + end_label_offset[0], h.r[-1] + end_label_offset[1], lab,
    #                       fontsize=9, weight='bold', va='center', ha='left', zorder=30,
    #                       color='k')
    #         # White halo to keep text visible over any background
    #         txt.set_path_effects([pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])

    # Choose a discrete colormap for trajectories
    colors = plt.cm.Dark2(np.linspace(0, 1, n))
    # --- Trajectories
    line_handles = []   # collect only line handles for the legend

    colors = ["navy", "#FF1493", "gold", "lime", "black"]

    for i, (h, lab) in enumerate(zip(histories, labels)):
        color = colors[i % len(colors)]

        # Simple line for the trajectory
        ax.plot(h.z, h.r, color=color, linewidth=2.2, alpha=0.95, label=lab)

        # Simple line for the trajectory — keep the handle
        line, = ax.plot(h.z, h.r, color=color, linewidth=2.2, alpha=0.95, label=lab)
        line_handles.append(line)

        # Small checkpoint markers
        if marker_every is not None and marker_every > 0:
            mk = _MARKERS[i % len(_MARKERS)]
            idx = np.arange(0, len(h.z), marker_every)
            ax.scatter(h.z[idx], h.r[idx],
                       s=marker_size, marker=mk, facecolor="none", edgecolor=color,
                       linewidths=0.9, zorder=15)

        # Start & end markers
        ax.scatter(h.z[0],  h.r[0],  s=28, marker='o', facecolor='white', edgecolor=color, zorder=20)
        ax.scatter(h.z[-1], h.r[-1], s=36, marker='X', facecolor=color, edgecolor='white', zorder=20)

        # End label text
        if place_end_labels:
            txt = ax.text(h.z[-1] + end_label_offset[0], h.r[-1] + end_label_offset[1], lab,
                          fontsize=9, weight='bold', va='center', ha='left', zorder=30,
                          color=color)
            txt.set_path_effects([pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])

    # Add the legend for trajectories

    ax.legend(
        line_handles, labels,
        loc="center left",  # anchor legend to the left-center
        bbox_to_anchor=(1.02, 0.3),  # shift it just outside the axes (x=1.02 means just right of axis)
        fontsize=7,
        frameon=True,
        title="Radial Position"
    )

    # Shared colorbar for the scalar along the path
    sm = plt.cm.ScalarMappable(cmap=cmap_line, norm=norm); sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, pad=0.1); cbar.set_label(clabel)

    # Reference lines (your originals)
    for y in [0.0188, -0.0188]:
        ax.hlines(y, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.hlines(-0.003, xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.hlines(0.003,  xmin=0, xmax=0.05, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.vlines(x=0.05, ymin=-0.025, ymax=-0.0188, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.vlines(x=0.05, ymin=-0.003,  ymax=0.003,  color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.vlines(x=0.05, ymin=0.0188,   ymax=0.025, color='black', linestyle='--', linewidth=1.0, alpha=0.8)

    # Axes, limits, labels
    _set_style(ax,
               xlabel="Axial position, z (m)",
               ylabel="Radial position, r (m)",
               title=title,
               xlim=zlim or (float(np.min(Z)), float(np.max(Z))),
               ylim=rlim or (float(np.min(R)), float(np.max(R))),
               aspect='auto')

    # Legend linking marker shapes to labels (colorbar already explains the colors)
    if proxy_handles:
        ax.legend(proxy_handles, proxy_labels, loc=legend_loc, fontsize=9, frameon=True)

    plt.tight_layout()
    if savepath:
        # If you render headless, consider switching backend to Agg earlier (as suggested before)
        fig.savefig(savepath, bbox_inches='tight', dpi=1000)
        plt.close(fig)
    return fig, ax


from typing import Sequence, Union, Dict, List


def get_final_values(hists: Union["ParticleHistory", Sequence["ParticleHistory"]]) -> List[Dict[str, float]]:
    """
    Return the final values (last entry) of one or more ParticleHistory objects.
    Each result is a dict with keys: t, z, r, uz, ur, Tp, dp, x (if present).
    """
    if not isinstance(hists, Sequence) or hasattr(hists, "t"):
        histories = [hists]
    else:
        histories = list(hists)

    results = []
    for h in histories:
        vals = {}
        for attr in ["t", "z", "r", "uz", "ur", "Tp", "dp", "x", "spheroid"]:
            if hasattr(h, attr):
                arr = getattr(h, attr)
                if len(arr) > 0:
                    vals[attr] = float(arr[-1])   # take final entry
        results.append(vals)

    return results


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def plot_dp_change_vs_initial(histories):
    """
    Scatter: initial dp (µm) vs % change in dp (final - initial)/initial*100.
    Color encodes spheroidization using fixed colors (no cmap):
      - Not spheroidized (0): color A
      - Spheroidized   (1): color B
    """
    finals = get_final_values(histories)

    # Data arrays
    dp0_um   = np.array([float(h.dp[0]) * 1e6 for h in histories])
    dpf_um   = np.array([float(v.get("dp", np.nan)) * 1e6 for v in finals])
    delta_um = (dpf_um - dp0_um) / dp0_um * 100.0

    # Spheroid flag (0/1), default 0
    sph = []
    for h, v in zip(histories, finals):
        if "spheroid" in v:
            sph.append(int(v["spheroid"]))
        elif hasattr(h, "spheroid"):
            sph.append(int(getattr(h, "spheroid")))
        elif hasattr(h, "is_spheroid"):
            sph.append(int(getattr(h, "is_spheroid")))
        else:
            sph.append(0)
    sph = np.array(sph, dtype=int)

    # Masks
    m0 = (sph == 0)  # not spheroidized
    m1 = (sph == 1)  # spheroidized

    # Fixed colors (change if you like)
    col0 = "#1f77b4"  # blue  (not spheroidized)
    col1 = "#d62728"  # red   (spheroidized)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot each group separately (no cmap)
    if np.any(m0):
        ax.scatter(dp0_um[m0], delta_um[m0], s=50, edgecolors="k",
                   facecolor=col0, label="_nolegend_")
    if np.any(m1):
        ax.scatter(dp0_um[m1], delta_um[m1], s=50, edgecolors="k",
                   facecolor=col1, label="_nolegend_")

    # Manual legend with same fixed colors
    handles = []
    if np.any(m0):
        handles.append(mlines.Line2D([], [], marker='o', linestyle='None',
                                     markerfacecolor=col0, markeredgecolor='k',
                                     markersize=8, label="Not spheroidized"))
    if np.any(m1):
        handles.append(mlines.Line2D([], [], marker='o', linestyle='None',
                                     markerfacecolor=col1, markeredgecolor='k',
                                     markersize=8, label="Spheroidized"))

    if handles:
        ax.legend(handles=handles, title="Particle type", loc="best")

    ax.set_xlabel("Initial particle diameter, $d_{p,0}$ (µm)")
    ax.set_ylabel("Change in diameter, %")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("diameter plots.png", dpi=1000)
    return fig, ax

