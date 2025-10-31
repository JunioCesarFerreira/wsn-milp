import numpy as np
import math

# =====================================================================================
# Gerador de nós fixos em GRADE centrada no sink (substitui os aleatórios)
# =====================================================================================

def add_grid_fixed_motes(sim_model: dict, rows: int = 5, cols: int = 5, d: float = 45.0) -> dict:
    """
    Remove todos os motes fixos, preserva apenas o 'root' (sink) e adiciona nós fixos
    em uma grade rows x cols com espaçamento d, centrada no sink.

    - O ponto central da grade coincide com a posição do sink.
    - Os novos nós recebem nomes 'node1', 'node2', ...
    """
    se = sim_model["simulationElements"]
    fixed = se["fixedMotes"]

    # Identificar sink ('root'); caso não exista, usar o primeiro como sink.
    sink_idx = None
    for idx, fm in enumerate(fixed):
        if fm.get("name", "").lower() == "root":
            sink_idx = idx
            break
    if sink_idx is None:
        if not fixed:
            raise ValueError("Não há motes fixos para definir o sink.")
        sink_idx = 0

    sink = fixed[sink_idx]
    sink_pos = np.array(sink["position"], dtype=float)

    # Resetar lista de fixos para manter apenas o sink
    se["fixedMotes"] = [sink]
    fixed = se["fixedMotes"]

    # Criar grade centrada no sink
    cx = (cols - 1) / 2.0
    ry = (rows - 1) / 2.0

    # Nomear novos nós
    next_id = 1

    def next_name():
        nonlocal next_id
        nm = f"node{next_id}"
        next_id += 1
        return nm

    # Inserir nós da grade (pular o centro exato onde já existe o sink)
    for i in range(rows):
        for j in range(cols):
            offset = np.array([(j - cx) * d, (i - ry) * d], dtype=float)
            pos = sink_pos + offset
            if np.allclose(pos, sink_pos, atol=1e-9):
                continue  # centro é o sink
            fixed.append({
                "position": [float(pos[0]), float(pos[1])],
                "name": next_name(),
                "sourceCode": "node.c"
            })

    return sim_model


# =====================================================================================
# Grade TRIANGULAR (hexagonal) centrada no sink
# =====================================================================================

def add_triangular_grid_fixed_motes(sim_model: dict, rows: int = 6, cols: int = 6, d: float = 45.0) -> dict:
    """
    Remove todos os motes fixos, preserva apenas o 'root' (sink) e adiciona nós fixos
    em uma grade triangular (hexagonal) rows×cols com espaçamento horizontal d e
    espaçamento vertical dy = d*sqrt(3)/2, centrada no sink.

    - Linhas ímpares são deslocadas em d/2 (padrão de colmeia).
    - O centro geométrico da grade coincide com a posição do sink.
    - Os novos nós recebem nomes 'node1', 'node2', ...
    """
    se = sim_model["simulationElements"]
    fixed = se["fixedMotes"]

    # Identificar sink ('root'); caso não exista, usar o primeiro como sink.
    sink_idx = None
    for idx, fm in enumerate(fixed):
        if fm.get("name", "").lower() == "root":
            sink_idx = idx
            break
    if sink_idx is None:
        if not fixed:
            raise ValueError("Não há motes fixos para definir o sink.")
        sink_idx = 0

    sink = fixed[sink_idx]
    sink_pos = np.array(sink["position"], dtype=float)

    # Resetar lista de fixos para manter apenas o sink
    se["fixedMotes"] = [sink]
    fixed = se["fixedMotes"]

    dy = d * math.sqrt(3) / 2.0

    # 1) Gera todos os pontos relativos da grade
    rel_points = []
    for i in range(rows):
        for j in range(cols):
            x = j * d + (i % 2) * (d / 2.0)   # deslocamento em colunas alternadas
            y = i * dy
            rel_points.append(np.array([x, y], dtype=float))

    # 2) Centraliza a grade: subtrai o centroide dos pontos
    rel_arr = np.array(rel_points, dtype=float)
    center = np.mean(rel_arr, axis=0)
    rel_arr_centered = rel_arr - center  # agora o centro é (0,0)

    # 3) Tradução para o sink
    abs_points = rel_arr_centered + sink_pos

    # 4) Inserir nós (pular o ponto que coincidir exatamente com o sink)
    next_id = 1
    def next_name():
        nonlocal next_id
        nm = f"node{next_id}"
        next_id += 1
        return nm

    for p in abs_points:
        if np.allclose(p, sink_pos, atol=1e-9):
            continue
        fixed.append({
            "position": [float(p[0]), float(p[1])],
            "name": next_name(),
            "sourceCode": "node.c"
        })

    return sim_model