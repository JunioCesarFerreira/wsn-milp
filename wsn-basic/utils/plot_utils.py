import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ================
# FIGURAS
# ================

def plot_candidates_and_paths(F, q_fixed, q_sink, R_comm, mob_names, r_mobile, T, region, out_path="./pic1.jpg"):
    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    # candidatos + raios
    for j in F:
        q = q_fixed[j]
        ax.scatter([q[0]], [q[1]], marker='s', s=60)
        ax.add_patch(Circle((q[0], q[1]), R_comm, fill=False, linewidth=1, ls='--'))

    # sink + raio
    ax.scatter([q_sink[0]], [q_sink[1]], marker='*', s=180, label="sink")
    ax.add_patch(Circle((q_sink[0], q_sink[1]), R_comm, fill=False, linewidth=1, ls='--'))

    # trajetórias
    for name in mob_names:
        traj = np.array([r_mobile(name, t) for t in range(1, T + 1)] + [r_mobile(name, 1)])
        ax.plot(traj[:, 0], traj[:, 1], linestyle='-', label=f"traj {name}")
        ax.scatter(traj[:, 0], traj[:, 1], marker='o', s=12)

    ax.set_title("Candidatos, sink e trajetórias (raios de comunicação)")
    ax.axis('equal')
    ax.grid(True)
    ax.legend(loc="best")
    if region and len(region) == 4:
        ax.set_xlim(region[0], region[2])
        ax.set_ylim(region[1], region[3])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_solution(F, installed, q_fixed, q_sink, R_comm, mob_names, r_mobile, T,
                  E_t, x_val, pos_node, t_plot, region, out_path="./pic2.png"):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # todos candidatos
    for j in F:
        q = q_fixed[j]
        ax.scatter([q[0]], [q[1]], marker='s', s=40, alpha=0.6)

    # instalados
    for j in installed:
        q = q_fixed[j]
        ax.scatter([q[0]], [q[1]], marker='s', s=120, label=f"instalado {j[1]}")
        ax.add_patch(Circle((q[0], q[1]), R_comm, fill=False, linewidth=1, ls='--'))

    # sink
    ax.scatter([q_sink[0]], [q_sink[1]], marker='*', s=180, label="sink")
    ax.add_patch(Circle((q_sink[0], q_sink[1]), R_comm, fill=False, linewidth=1, ls='--'))

    # trajetórias
    for name in mob_names:
        traj = np.array([r_mobile(name, t) for t in range(1, T + 1)] + [r_mobile(name, 1)])
        ax.plot(traj[:, 0], traj[:, 1], linestyle='-', alpha=0.6)
        ax.scatter(traj[:, 0], traj[:, 1], marker='o', s=10, alpha=0.6)

    # fluxos ativos no snapshot
    flows_t = {(i, j): x_val[(i, j, t_plot)] for (i, j) in E_t[t_plot] if x_val[(i, j, t_plot)] > 1e-6}
    for (i, j), val in flows_t.items():
        pi, pj = pos_node(i, t_plot), pos_node(j, t_plot)
        ax.plot([pi[0], pj[0]], [pi[1], pj[1]], linewidth=2)
        cx, cy = (pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2
        ax.text(cx, cy, f"{val:.2f}", fontsize=8)

    ax.set_title(f"Solução: instalados e fluxos em t={t_plot} (com raios)")
    ax.axis('equal')
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    if region and len(region) == 4:
        ax.set_xlim(region[0], region[2])
        ax.set_ylim(region[1], region[3])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
