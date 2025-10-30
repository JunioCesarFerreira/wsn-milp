# sim_utils.py
import json
import math
import numpy as np

def load_simulation_json(path_or_dict):
    """Aceita um caminho de arquivo .json OU um dict já carregado e retorna o bloco 'simulationModel'."""
    if isinstance(path_or_dict, str):
        with open(path_or_dict, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(path_or_dict, dict):
        data = path_or_dict
    else:
        raise TypeError("path_or_dict deve ser str (caminho) ou dict (JSON carregado).")
    return data["simulationModel"]

def _safe_eval_expr(expr: str, t: float) -> float:
    """
    Avalia expressão de string do JSON com variável escalar 't' em [0,1].
    Permite np.* e números; sem builtins perigosos.
    """
    try:
        return float(expr)
    except (ValueError, TypeError):
        pass
    allowed_globals = {"__builtins__": {}, "np": np, "math": math}
    allowed_locals = {"t": float(t)}
    return float(eval(expr, allowed_globals, allowed_locals))

def _segment_length(function_pair, nsamples=200):
    """
    Aproxima o comprimento do segmento definido por (x_expr, y_expr), t in [0,1],
    por uma polilinha com 'nsamples' amostras.
    """
    x_expr, y_expr = function_pair
    ts = np.linspace(0.0, 1.0, nsamples + 1)
    pts = np.stack([
        [_safe_eval_expr(str(x_expr), tt), _safe_eval_expr(str(y_expr), tt)]
        for tt in ts
    ], axis=0)
    diffs = np.diff(pts, axis=0)
    seg_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    return seg_len

def _distribute_integer_proportions(total_steps, weights):
    """
    Dado total_steps (int) e uma lista/array de pesos não-negativos,
    retorna uma lista de inteiros que somam total_steps, proporcional aos pesos.
    Estratégia: floor + atribuição dos restos de maiores frações.
    """
    w = np.asarray(weights, dtype=float)
    if len(w) == 0:
        return []
    if np.all(w <= 0):
        base = total_steps // len(w)
        rem  = total_steps % len(w)
        steps = [base] * len(w)
        for i in range(rem):
            steps[i] += 1
        return steps
    w = np.maximum(w, 0.0)
    W = float(np.sum(w))
    raw = (total_steps * w / W) if W > 0 else np.zeros_like(w)
    flo = np.floor(raw).astype(int)
    rem = int(total_steps - np.sum(flo))
    frac = raw - flo
    order = np.argsort(-frac)
    steps = flo.tolist()
    for k in range(rem):
        steps[order[k]] += 1
    return steps

def make_mobile_trajectory_fn(function_path, is_closed: bool, is_roundtrip: bool, T: int, speed: float):
    """
    Cria uma função r(tau) que retorna posição 2D para tau=1..T (inteiros).
    A distribuição de passos por segmento considera o tempo de cada trecho:
        time_k = length_k / speed
    e aloca passos proporcionalmente a time_k.

    - function_path: lista de [x_expr, y_expr] com 't'∈[0,1].
    - is_closed: percurso cíclico.
    - is_roundtrip: se não for fechado, ida-e-volta.
    """
    K = len(function_path)
    if K == 0:
        raise ValueError("functionPath vazio.")

    # Comprimento por segmento original
    lens_by_k = [_segment_length(function_path[k], nsamples=200) for k in range(K)]

    # Sequência efetiva
    if is_closed:
        seq = list(range(K))
    elif is_roundtrip and K > 1:
        seq = list(range(K)) + list(range(K-2, 0, -1))
    else:
        seq = list(range(K))

    # Tempo efetivo por segmento (comprimento/velocidade)
    spd = float(speed) if speed is not None else 1.0
    spd = 1.0 if spd <= 0 else spd
    times_eff = [lens_by_k[k] / spd for k in seq]

    # Alocação de passos proporcional ao "tempo" de cada segmento
    steps_per_seg = _distribute_integer_proportions(T, times_eff)
    if T >= len(seq):
        steps_per_seg = [max(1, s) for s in steps_per_seg]
        surplus = int(sum(steps_per_seg) - T)
        if surplus > 0:
            # remove 1 passo dos com menor tempo efetivo
            order = np.argsort(times_eff)
            for idx in order:
                if surplus == 0:
                    break
                if steps_per_seg[idx] > 1:
                    steps_per_seg[idx] -= 1
                    surplus -= 1
        elif surplus < 0:
            deficit = -surplus
            order = np.argsort(-np.asarray(times_eff))
            for k in range(deficit):
                steps_per_seg[order[k % len(seq)]] += 1

    cut = np.cumsum([0] + steps_per_seg)
    S = len(seq)

    def r_of_tau(tau: int) -> np.ndarray:
        u = tau - 1
        seg_eff = int(np.searchsorted(cut, u, side="right") - 1)
        seg_eff = min(max(seg_eff, 0), S-1)
        local_len = steps_per_seg[seg_eff]
        if local_len <= 1:
            tloc = 1.0
        else:
            tloc = (u - cut[seg_eff]) / (local_len - 1)  # [0,1]
        k = seq[seg_eff]
        x_expr, y_expr = function_path[k]
        x = _safe_eval_expr(str(x_expr), tloc)
        y = _safe_eval_expr(str(y_expr), tloc)
        return np.array([x, y], dtype=float)

    return r_of_tau

# ==============================
# Fixos aleatórios
# ==============================

def add_random_fixed_motes(sim_model: dict, n_new: int = 20, seed: int = 42) -> dict:
    """
    Adiciona n_new motes fixos aleatórios dentro de simulationModel['region'].
    Mantém 'root' e continua a numeração 'nodeX' a partir do maior X existente.
    Opera diretamente sobre o dict 'simulationModel' (retorna o mesmo objeto mutado).
    """
    xmin, ymin, xmax, ymax = map(float, sim_model["region"])
    fixed = sim_model["simulationElements"]["fixedMotes"]

    # nomes existentes e maior índice nodeNN
    import re
    names = {fm["name"] for fm in fixed}
    pat = re.compile(r"^node(\d+)$", re.IGNORECASE)
    max_idx = 0
    for nm in names:
        m = pat.match(nm)
        if m:
            max_idx = max(max_idx, int(m.group(1)))

    rng = np.random.default_rng(seed)
    for _ in range(n_new):
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        max_idx += 1
        new_name = f"node{max_idx}"
        while new_name in names:
            max_idx += 1
            new_name = f"node{max_idx}"
        new_fixed = {
            "position": [float(x), float(y)],
            "name": new_name,
            "sourceCode": "node.c"
        }
        fixed.append(new_fixed)
        names.add(new_name)
    return sim_model