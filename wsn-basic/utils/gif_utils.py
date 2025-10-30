import os
import io, shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(
        "Este script requer 'Pillow' para gerar o GIF. Instale com: pip install pillow"
    ) from e 
    
def save_routes_gif(installed, r_mobile, mob_names, q_sink, q_fixed, R_comm, region, x_val, E_t, T, F):
    # Pasta de frames
    frames_dir = "./frames_gif"
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    # Limiares e aparência
    FLOW_EPS = 1e-6                # fluxo mínimo para considerar uma aresta ativa
    NODE_S_SMALL = 40
    NODE_S_BIG = 120

    # Pré-computar trajetórias para contexto (uma vez só)
    traj_cache = {name: np.array([r_mobile(name, t) for t in range(1, T+1)]) for name in mob_names}

    def _pos_node(n, t):
        if n[0] == "sink":
            return q_sink
        if n[0] == "f":
            return q_fixed[n]
        if n[0] == "m":
            return r_mobile(n[1], t)
        raise ValueError("nó desconhecido")

    def _draw_frame(t):
        """Desenha e retorna um objeto PIL.Image do frame do tempo t."""
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()

        # Todos candidatos (marcadores pequenos)
        for j in F:
            q = q_fixed[j]
            ax.scatter([q[0]], [q[1]], marker='s', s=NODE_S_SMALL, alpha=0.6, label=None)

        # Instalados (marcadores maiores + raio)
        for j in installed:
            q = q_fixed[j]
            ax.scatter([q[0]], [q[1]], marker='s', s=NODE_S_BIG, label=None)
            ax.add_patch(Circle((q[0], q[1]), R_comm, fill=False, linewidth=1, ls='--'))

        # Sink + raio
        ax.scatter([q_sink[0]], [q_sink[1]], marker='*', s=180, label=None)
        ax.add_patch(Circle((q_sink[0], q_sink[1]), R_comm, fill=False, linewidth=1, ls='--'))

        # Trajetórias (linhas claras para contexto)
        for name in mob_names:
            traj = traj_cache[name]
            ax.plot(traj[:, 0], traj[:, 1], linestyle='-', alpha=0.25, label=None)

        # Posição atual de cada móvel (ponto mais destacado)
        for name in mob_names:
            pm = r_mobile(name, t)
            ax.scatter([pm[0]], [pm[1]], marker='o', s=60, label=None)

        # Arestas ativas (x_ij(t) > FLOW_EPS)
        flows = {(i, j): x_val[(i, j, t)] for (i, j) in E_t[t] if x_val[(i, j, t)] > FLOW_EPS}
        for (i, j), val in flows.items():
            pi, pj = _pos_node(i, t), _pos_node(j, t)
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], linewidth=2)
            cx, cy = (pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2
            ax.text(cx, cy, f"{val:.2f}", fontsize=8)

        ax.set_title(f"Rotas de comunicação (t = {t})")
        ax.axis('equal')
        ax.grid(True)
        if region and len(region) == 4:
            ax.set_xlim(region[0], region[2])
            ax.set_ylim(region[1], region[3])

        plt.tight_layout()

        # Renderizar figura para um buffer e criar PIL.Image
        buf = io.BytesIO()
        fig.canvas.print_png(buf)
        plt.close(fig)  # fecha a figura para liberar memória
        buf.seek(0)
        img = Image.open(buf).convert("P")  # paleta para GIF
        return img

    # Gerar e salvar frames individuais + montar GIF
    frames = []
    for t in range(1, T + 1):
        img = _draw_frame(t)
        frame_path = os.path.join(frames_dir, f"frame_{t:03d}.png")
        img.convert("RGB").save(frame_path, format="PNG")
        frames.append(img)

    # Salvar GIF (autoplay/loop quando embutido no navegador/apresentações)
    gif_path = "./routes.gif"
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=500,   # ms por frame (2 fps); ajuste aqui
            loop=0          # 0 = loop infinito
        )

def save_routes2_gif(installed, r_mobile, mob_names, q_sink, q_fixed, R_comm, region, x_val, E_t, T, F):
    # Pasta de frames
    frames_dir = "./frames_gif"
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    # Limiares e aparência
    FLOW_EPS = 1e-6
    NODE_S_SMALL = 40
    NODE_S_BIG = 120

    # Pré-computar trajetórias para contexto (uma vez só)
    traj_cache = {name: np.array([r_mobile(name, t) for t in range(1, T+1)]) for name in mob_names}

    def _pos_node(n, t):
        if n[0] == "sink":
            return q_sink
        if n[0] == "f":
            return q_fixed[n]
        if n[0] == "m":
            return r_mobile(n[1], t)
        raise ValueError("nó desconhecido")

    def _draw_frame(t):
        """Desenha e retorna um objeto PIL.Image do frame do tempo t."""
        fig = plt.figure(figsize=(10, 8))
        ax = plt.gca()

        # Todos candidatos (sem círculos)
        for j in F:
            q = q_fixed[j]
            ax.scatter([q[0]], [q[1]], marker='s', s=NODE_S_SMALL, alpha=0.6, label=None)

        # Instalados (sem círculos)
        for j in installed:
            q = q_fixed[j]
            ax.scatter([q[0]], [q[1]], marker='s', s=NODE_S_BIG, label=None)

        # Sink (sem círculo)
        ax.scatter([q_sink[0]], [q_sink[1]], marker='*', s=180, label=None)

        # Trajetórias dos móveis – destacadas com tracejado
        for name in mob_names:
            traj = traj_cache[name]
            ax.plot(traj[:, 0], traj[:, 1], linestyle='--', linewidth=2, alpha=0.7, label=None)

        # Posição atual de cada móvel (ponto + círculo de alcance)
        for name in mob_names:
            pm = r_mobile(name, t)
            ax.scatter([pm[0]], [pm[1]], marker='o', s=60, label=None)
            ax.add_patch(Circle((pm[0], pm[1]), R_comm, fill=False, linewidth=1, ls='--'))

        # Arestas ativas; evidenciar rotas que envolvem móveis com tracejado
        flows = {(i, j): x_val[(i, j, t)] for (i, j) in E_t[t] if x_val[(i, j, t)] > FLOW_EPS}
        for (i, j), val in flows.items():
            pi, pj = _pos_node(i, t), _pos_node(j, t)
            is_mobile_route = (i[0] == "m") or (j[0] == "m")
            ls = '--' if is_mobile_route else '-'
            lw = 2.5 if is_mobile_route else 1.5
            alpha = 0.95 if is_mobile_route else 0.7
            ax.plot([pi[0], pj[0]], [pi[1], pj[1]], linewidth=lw, linestyle=ls, alpha=alpha)
            cx, cy = (pi[0] + pj[0]) / 2, (pi[1] + pj[1]) / 2
            ax.text(cx, cy, f"{val:.2f}", fontsize=8)

        ax.set_title(f"Rotas de comunicação (t = {t})")
        ax.axis('equal')
        ax.grid(True)
        if region and len(region) == 4:
            ax.set_xlim(region[0], region[2])
            ax.set_ylim(region[1], region[3])

        plt.tight_layout()

        # Renderizar figura para buffer → PIL.Image
        buf = io.BytesIO()
        fig.canvas.print_png(buf)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("P")  # paleta para GIF
        return img

    # Gerar e salvar frames individuais + montar GIF
    frames = []
    for t in range(1, T + 1):
        img = _draw_frame(t)
        frame_path = os.path.join(frames_dir, f"frame_{t:03d}.png")
        img.convert("RGB").save(frame_path, format="PNG")
        frames.append(img)

    # Salvar GIF (loop infinito)
    gif_path = "./routes2.gif"
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=500,   # ms por frame
            loop=0
        )
       
        
        
def save_mobile_gif(mobile_name, r_mobile, q_sink, q_fixed, R_comm, region, T, out_path="mobile.gif", fps=10):
    """
    Cria um GIF do móvel se deslocando, mostrando o sink e candidatos.
    Requer 'imageio' instalado.
    """
    import imageio.v2 as imageio

    frames = []
    for t in range(1, T + 1):
        fig, ax = plt.subplots(figsize=(6, 6))
        # candidatos
        for j, q in q_fixed.items():
            ax.scatter([q[0]], [q[1]], marker='s', s=40, alpha=0.6)
        # sink
        ax.scatter([q_sink[0]], [q_sink[1]], marker='*', s=180)
        # móvel
        p = r_mobile(mobile_name, t)
        ax.scatter([p[0]], [p[1]], marker='o', s=80)
        ax.add_patch(Circle((p[0], p[1]), R_comm, fill=False, linewidth=1, ls='--'))

        ax.set_title(f"{mobile_name} - t={t}")
        ax.axis('equal')
        ax.grid(True)
        if region and len(region) == 4:
            ax.set_xlim(region[0], region[2])
            ax.set_ylim(region[1], region[3])
        plt.tight_layout()

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps)