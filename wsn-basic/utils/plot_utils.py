import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ======================
# Helper: trajetórias com quebras
# ======================

def _traj_with_breaks(r_mobile, name, T, close=False, jump_factor=5.0):
    """
    Gera uma trajetória shape (N,2) com linhas 'quebradas' (NaNs) em saltos grandes.
    - close=False por padrão: não faz o wrap (evita risco entre último e primeiro).
    - jump_factor controla a sensibilidade: quebra quando passo > jump_factor * mediana.
    """
    pts = np.array([r_mobile(name, t) for t in range(1, T + 1)], dtype=float)

    # Opcionalmente fechar (em geral, NÃO faça para caminhos abertos)
    if close:
        pts = np.vstack([pts, pts[0]])

    # Distâncias entre pontos consecutivos
    if len(pts) >= 2:
        diffs = np.diff(pts, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        med = np.median(dists) if np.any(dists > 0) else 0.0
        # limiar robusto: maior entre 1e-9 e mediana*jump_factor
        thr = max(1e-9, med * jump_factor)

        # Monta com NaNs onde houver salto
        rows = [pts[0]]
        for k in range(1, len(pts)):
            if dists[k - 1] > thr:
                rows.append([np.nan, np.nan])  # quebra a linha aqui
            rows.append(pts[k])
        pts = np.array(rows, dtype=float)
    return pts

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

    # trajetórias (NÃO fecha; quebra saltos com NaN)
    for name in mob_names:
        traj = _traj_with_breaks(r_mobile, name, T, close=False, jump_factor=5.0)
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

def plot_solution(F, installed, q_fixed, q_sink, R_comm, 
                  mob_names, T, r_mobile, region, out_path="./pic2.png"):
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

    # trajetórias (contexto): não fechar, com quebras
    for name in mob_names:
        traj = _traj_with_breaks(r_mobile, name, T, close=False, jump_factor=5.0)
        ax.plot(traj[:, 0], traj[:, 1], linestyle='-', alpha=0.6)
        ax.scatter(traj[:, 0], traj[:, 1], marker='o', s=10, alpha=0.6)

    ax.set_title(f"Solução (raios de comunicação instalados)")
    ax.axis('equal')
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    if region and len(region) == 4:
        ax.set_xlim(region[0], region[2])
        ax.set_ylim(region[1], region[3])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()