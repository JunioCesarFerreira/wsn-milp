import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Callable

# ======================
# Helper: trajectories with breaks
# ======================

def _traj_with_breaks(
    r_mobile: Callable[[str, int], tuple[float, float]],
    name: str,
    T: int,
    close: bool = False,
    jump_factor: float = 5.0,
) -> np.array:
    """
    Generates a trajectory of shape (N, 2) with 'broken' lines (NaNs) at large jumps.

    - close=False by default: does not wrap the path (avoids risk between last and first points).
    - jump_factor controls sensitivity: a break is inserted when step > jump_factor * median step.
    """
    pts = np.array([r_mobile(name, t) for t in range(1, T + 1)], dtype=float)

    # Optionally close the trajectory (generally NOT recommended for open paths)
    if close:
        pts = np.vstack([pts, pts[0]])

    # Distances between consecutive points
    if len(pts) >= 2:
        diffs = np.diff(pts, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        med = np.median(dists) if np.any(dists > 0) else 0.0
        thr = max(1e-9, med * jump_factor)  # robust threshold

        # Build trajectory inserting NaNs where jumps occur
        rows = [pts[0]]
        for k in range(1, len(pts)):
            if dists[k - 1] > thr:
                rows.append([np.nan, np.nan])  # break the line here
            rows.append(pts[k])
        pts = np.array(rows, dtype=float)
    return pts

# ======================
# Palette / sizes (visual defaults)
# ======================

COLOR_SINK      = "blue"
COLOR_CANDIDATE = "black"
COLOR_FIXED0    = "gray"    # not installed
COLOR_FIXED1    = "red"     # installed
COLOR_MOBILE    = "green"
COLOR_TARGET    = "magenta"

S_SINK      = 260
S_FIXED0    = 40
S_FIXED1    = 120
S_MOBILE    = 18   # trajectory points
S_TARGET    = 60   # target size

# ================
# FIGURE 1
# ================

def plot_candidates_and_paths(
    F,
    q_fixed,
    q_sink,
    R_comm,
    mob_names,
    r_mobile,
    T,
    region,
    out_path="./pic1.jpg",
    targets=None,
    q_target=None,
    R_cov=None,
):
    """
    Plots:
      - candidate positions (F)
      - sink
      - mobile trajectories (if any)
      - (optional) targets and their coverage radii
    """
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # Candidates + communication radii
    for j in F:
        q = q_fixed[j]
        ax.add_patch(
            Circle(
                (q[0], q[1]),
                R_comm,
                fill=False,
                linewidth=1,
                ls="--",
                edgecolor="gray",
            )
        )
        ax.scatter([q[0]], [q[1]], marker="s", s=S_FIXED0, c=COLOR_CANDIDATE)

    # Sink + communication radius
    ax.scatter(
        [q_sink[0]],
        [q_sink[1]],
        marker="*",
        s=S_SINK,
        c=COLOR_SINK,
        label="sink",
    )
    ax.add_patch(
        Circle(
            (q_sink[0], q_sink[1]),
            R_comm,
            fill=False,
            linewidth=1,
            ls="--",
            edgecolor=COLOR_SINK,
            alpha=0.6,
        )
    )

    # (Optional) targets + coverage radii
    if targets is not None and q_target is not None and len(targets) > 0:
        for h in targets:
            p = q_target[h]
            # target point
            ax.scatter(
                [p[0]],
                [p[1]],
                marker="^",
                s=S_TARGET,
                c=COLOR_TARGET,
                alpha=0.9,
                label=None,
            )

        # Single legend entry for targets
        ax.scatter([], [], marker="^", s=S_TARGET, c=COLOR_TARGET, label="target")

    # Trajectories (do NOT close; break large jumps with NaNs)
    for name in mob_names:
        traj = _traj_with_breaks(
            r_mobile, name, T, close=False, jump_factor=5.0
        )
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            c=COLOR_MOBILE,
            label=None,
        )
        ax.scatter(
            traj[:, 0],
            traj[:, 1],
            marker="o",
            s=S_MOBILE,
            c=COLOR_MOBILE,
            alpha=0.7,
            label=None,
        )

    ax.set_title("Candidates, sink, trajectories and targets")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")
    if region and len(region) == 4:
        ax.set_xlim(region[0], region[2])
        ax.set_ylim(region[1], region[3])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ================
# FIGURE 2
# ================

def plot_solution(
    F: list[int],
    installed: list[int],
    q_fixed: dict[int, tuple[float, float]],
    q_sink: tuple[float, float],
    R_comm: float,
    R_inter: float,
    mob_names: list[str],
    T: int,
    r_mobile: Callable[[str, int], tuple[float, float]],
    region: list[float],
    out_path: str = "./pic2.png",
    targets: list[int] | None = None,
    q_target: dict[int, tuple[float, float]] | None = None,
    R_cov: float | None = None,
) -> None:
    """
    Plots:
      - non-installed candidates
      - installed candidates (with R_comm and R_inter)
      - sink (with radii)
      - mobile trajectories (if any)
      - (optional) targets with coverage radii
    """
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    installed_set = set(installed)

    # Non-installed candidates
    for j in F:
        if j not in installed_set:
            q = q_fixed[j]
            ax.scatter(
                [q[0]],
                [q[1]],
                marker="s",
                s=S_FIXED0,
                c=COLOR_FIXED0,
                alpha=0.9,
            )

    # Installed candidates
    for j in installed:
        q = q_fixed[j]
        ax.scatter(
            [q[0]],
            [q[1]],
            marker="s",
            s=S_FIXED1,
            c=COLOR_FIXED1,
            label=None,
        )
        ax.add_patch(
            Circle(
                (q[0], q[1]),
                R_comm,
                fill=False,
                linewidth=1,
                ls="--",
                edgecolor=COLOR_FIXED1,
                alpha=0.6,
            )
        )
        ax.add_patch(
            Circle(
                (q[0], q[1]),
                R_inter,
                fill=False,
                linewidth=1,
                ls="--",
                edgecolor="orange",
                alpha=0.6,
            )
        )

    # sink
    ax.scatter(
        [q_sink[0]],
        [q_sink[1]],
        marker="*",
        s=S_SINK,
        c=COLOR_SINK,
        label="sink",
    )
    ax.add_patch(
        Circle(
            (q_sink[0], q_sink[1]),
            R_comm,
            fill=False,
            linewidth=1,
            ls="--",
            edgecolor=COLOR_SINK,
            alpha=0.6,
        )
    )
    ax.add_patch(
        Circle(
            (q_sink[0], q_sink[1]),
            R_inter,
            fill=False,
            linewidth=1,
            ls="--",
            edgecolor="orange",
            alpha=0.6,
        )
    )

    # (Optional) targets + covered radius
    if targets is not None and q_target is not None and len(targets) > 0:
        for h in targets:
            p = q_target[h]
            ax.scatter(
                [p[0]],
                [p[1]],
                marker="^",
                s=S_TARGET,
                c=COLOR_TARGET,
                alpha=0.9,
                label=None,
            )
        ax.scatter([], [], marker="^", s=S_TARGET, c=COLOR_TARGET, label="target")

    # Trajectories (context)
    for name in mob_names:
        traj = _traj_with_breaks(
            r_mobile, name, T, close=False, jump_factor=5.0
        )
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            c=COLOR_MOBILE,
        )
        ax.scatter(
            traj[:, 0],
            traj[:, 1],
            marker="o",
            s=S_MOBILE,
            c=COLOR_MOBILE,
            alpha=0.7,
        )

    ax.set_title("Solution (candidates, installed, sink and targets)")
    ax.axis("equal")
    ax.grid(True)
    # ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    ax.legend(loc="best")
    if region and len(region) == 4:
        ax.set_xlim(region[0], region[2])
        ax.set_ylim(region[1], region[3])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_installed_graph(
    installed: list[int],
    q_fixed: dict[int, tuple[float, float]],
    q_sink: tuple[float, float],
    R_comm: float,
    region: list[float],
    out_path: str = "./pic_installed_graph.png",
    targets: list[int] | None = None,
    q_target: dict[int, tuple[float, float]] | None = None,
) -> None:
    """
    Plots only the graph formed by the INSTALLED FIXED motes.

    - Vertices: installed motes (COLOR_FIXED1) and sink (COLOR_SINK).
    - Edges: between installed motes (and sink to installed) when ||u - v|| <= R_comm.
    - (Optional) plots targets as points, without edges.
    """
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # Installed nodes (red)
    for j in installed:
        q = q_fixed[j]
        ax.scatter([q[0]], [q[1]], marker="s", s=S_FIXED1, c=COLOR_FIXED1)

    # sink (blue star)
    ax.scatter([q_sink[0]], [q_sink[1]], marker="*", s=S_SINK, c=COLOR_SINK)

    # (optional) targets
    if targets is not None and q_target is not None and len(targets) > 0:
        for h in targets:
            p = q_target[h]
            ax.scatter(
                [p[0]],
                [p[1]],
                marker="^",
                s=S_TARGET,
                c=COLOR_TARGET,
                alpha=0.9,
            )
        ax.scatter([], [], marker="^", s=S_TARGET, c=COLOR_TARGET, label="target")

    # Edges between installed nodes
    def _dist(p, q):
        return float(np.hypot(p[0] - q[0], p[1] - q[1]))

    for idx in range(len(installed)):
        for jdx in range(idx + 1, len(installed)):
            a = installed[idx]
            b = installed[jdx]
            pa, pb = q_fixed[a], q_fixed[b]
            if _dist(pa, pb) <= R_comm + 1e-9:
                ax.plot(
                    [pa[0], pb[0]],
                    [pa[1], pb[1]],
                    linewidth=2.0,
                    linestyle="-",
                    c="red",
                )

    # Edges sink ↔ installed
    for j in installed:
        p = q_fixed[j]
        if _dist(q_sink, p) <= R_comm + 1e-9:
            ax.plot(
                [q_sink[0], p[0]],
                [q_sink[1], p[1]],
                linewidth=2.0,
                linestyle="-",
                c="red",
            )

    ax.set_title("Installed fixed-node graph (edges ≤ R_comm)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(loc="best")
    if region and len(region) == 4:
        ax.set_xlim(region[0], region[2])
        ax.set_ylim(region[1], region[3])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
