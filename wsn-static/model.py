import numpy as np
import json
from pathlib import Path

# bibliotecas locais
from utils.sim_utils import load_simulation_json
from utils.sim_utils import add_random_fixed_motes
from utils.plot_utils import plot_installed_graph
from utils.plot_utils import plot_candidates_and_paths
from utils.plot_utils import plot_solution

# ==============================
# Gurobi
# ==============================
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    raise RuntimeError("Este script requer 'gurobipy'. Instale e garanta uma licença ativa do Gurobi.") from e

# ==============================
# Entrada
# ==============================
SIM_JSON_PATH = "./input.json"   # ajuste conforme necessário
RESULTS_PATH = Path("./output")
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# 1) Carrega JSON base
sim = load_simulation_json(SIM_JSON_PATH)

# ==============================
# Construção dos conjuntos/posições a partir do JSON
# ==============================

# tempo (não usado diretamente neste modelo, mas mantido)
duration = int(sim.get("duration", 60))

R_comm   = float(sim.get("radiusOfReach", 100.0))
R_interf = float(sim.get("radiusOfInter", 200.0))
region   = sim.get("region", [-200, -200, 200, 200])

fixed_list  = sim["simulationElements"]["fixedMotes"]

# parâmetros do modelo
cap0 = 10.0
kdecay = 0.1

w = float(sim.get("w_install", 1e6))  # peso da instalação no objetivo
k_cov = int(sim.get("k_coverage", 2))         # nível de cobertura k

# ------------------------------
# sink e nós fixos candidatos
# ------------------------------
sink_name = None
for fm in fixed_list:
    if fm.get("name", "").lower() == "root":
        sink_name = fm["name"]
        break
if sink_name is None and fixed_list:
    sink_name = fixed_list[0]["name"]

sink = ("sink", sink_name)
q_sink = None
J = []           # fixos candidatos (exclui sink)
q_fixed = {}     # ("f", name) -> np.array([x,y])

for fm in fixed_list:
    name = str(fm["name"])
    pos = np.array(fm["position"], dtype=float)
    if name == sink_name:
        q_sink = pos
    else:
        node_id = ("f", name)
        J.append(node_id)
        q_fixed[node_id] = pos

if q_sink is None:
    raise ValueError("Não foi possível determinar a posição do sink.")

def pos_node(n):
    """Retorna a posição (x,y) do nó n."""
    if n[0] == "sink":
        return q_sink
    if n[0] == "f":
        return q_fixed[n]
    raise ValueError(f"nó desconhecido: {n}")

# ==============================
# Conjunto de alvos (T) para cobertura
# ==============================
targets_raw = sim.get("targets", [])  # opcional; adapte ao seu JSON
T = []           # índices dos alvos
q_target = {}    # i -> np.array([x,y])

for idx, tgt in enumerate(targets_raw):
    pos = np.array(tgt["position"], dtype=float)
    T.append(idx)
    q_target[idx] = pos

print(q_target)

# ==============================
# Funções auxiliares
# ==============================

def link_possible(pi, pj):
    return np.linalg.norm(pi - pj) <= R_comm

def capacity(pi, pj):
    d = np.linalg.norm(pi - pj)
    return max(0.0, cap0 * (1 - kdecay * d) ** 2)

def link_cost(pi, pj):
    """w_uv no modelo."""
    N = 2
    delta = 0.5
    d = np.linalg.norm(pi - pj)
    noise = np.random.uniform(-delta, delta)
    if 0 < d <= R_comm:
        return d**N + noise
    elif R_comm < d <= R_interf:
        return (R_interf - d)**N + noise
    else:
        return 0.0

# ==============================
# Pré-processamento de arestas
# ==============================
A = {}
C = {}
w = {}
nodes_t = [sink] + J
E = []

for i in nodes_t:
    for j in nodes_t:
        if i == j:
            continue
        pi, pj = pos_node(i), pos_node(j)
        feas = link_possible(pi, pj)
        A[(i, j)] = 1 if feas else 0
        if feas:
            cap = capacity(pi, pj)
            if cap > 0:
                E.append((i, j))
                C[(i, j)] = cap   # big-M específico da aresta
                w[(i, j)] = link_cost(pi, pj)

# ==============================
# Modelo Gurobi
# ==============================
mdl = gp.Model("WSN_Placement_Routing")

mdl.Params.OutputFlag = 1  # 0 para silenciar logs

# --------------------------------------------
# Variáveis
#  - y_j: instalação de fixos (j ∈ J)
#  - z_uv: ativação da aresta (u,v)
#  - x_uv: fluxo na aresta (u,v)
# --------------------------------------------
y = {
    j: mdl.addVar(vtype=GRB.BINARY, name=f"y_{j[0]}_{j[1]}")
    for j in J
}

z = {}
xvar = {}

for (u, v) in E:
    z[(u, v)] = mdl.addVar(
        vtype=GRB.BINARY,
        name=f"z_{u[0]}_{u[1]}__{v[0]}_{v[1]}"
    )
    xvar[(u, v)] = mdl.addVar(
        lb=0.0,
        name=f"x_{u[0]}_{u[1]}__{v[0]}_{v[1]}"
    )

mdl.update()

# --------------------------------------------
# Objetivo
#   α * sum_j y_j  +  sum_(u,v) w_uv z_uv
# --------------------------------------------
obj_install = gp.quicksum(y[j] for j in J)
obj_edges = gp.quicksum(w[(u, v)] * z[(u, v)] for (u, v) in E)

mdl.setObjective(w * obj_install + obj_edges, GRB.MINIMIZE)

# --------------------------------------------
# Restrições
# --------------------------------------------

# (Cobertura)  sum_j a_ij y_j >= k,  ∀ i∈T
# Aqui: a_ij = 1 se dist(target_i, fixed_j) <= R_comm
if T:
    for i in T:
        pos_i = q_target[i]
        covering_sensors = [
            j for j in J
            if np.linalg.norm(pos_i - q_fixed[j]) <= R_comm
        ]
        if covering_sensors:
            mdl.addConstr(
                gp.quicksum(y[j] for j in covering_sensors) >= k_cov,
                name=f"cov_target_{i}"
            )
        else:
            # Nenhum candidato cobre este alvo;
            # se quiser forçar inviabilidade, poderia exigir >= k_cov mesmo assim.
            pass

# (Link só pode ativar se extremos instalados)
# z_uv <= y_u, z_uv <= y_v para (u,v) com u,v ∈ J
# z_sj <= y_j, z_js <= y_j para arestas que envolvem o sink
for (u, v) in E:
    # ponta u
    if u != sink and u[0] == "f":
        mdl.addConstr(
            z[(u, v)] <= y[u],
            name=f"z_le_y_u_{u[1]}__{v[1]}"
        )
    # ponta v
    if v != sink and v[0] == "f":
        mdl.addConstr(
            z[(u, v)] <= y[v],
            name=f"z_le_y_v_{u[1]}__{v[1]}"
        )

# (Capacidade) 0 <= x_uv <= M_uv * z_uv
for (u, v) in E:
    M_uv = C[(u, v)]
    mdl.addConstr(
        xvar[(u, v)] <= M_uv * z[(u, v)],
        name=f"cap_{u[1]}__{v[1]}"
    )

# (Conservação de fluxo nos fixos):
#   sum_{v:(j,v)∈E} x_jv - sum_{u:(u,j)∈E} x_uj = y_j , ∀ j∈J
for j in J:
    outflow = gp.quicksum(
        xvar[(j, v)] for (u, v) in E if u == j
    )
    inflow = gp.quicksum(
        xvar[(u, j)] for (u, v) in E if v == j
    )
    mdl.addConstr(
        outflow - inflow == y[j],
        name=f"flow_cons_{j[1]}"
    )

# (Nó sink):
#   sum_{u:(u,s)∈E} x_us = sum_{j∈J} y_j
inflow_sink = gp.quicksum(
    xvar[(u, sink)] for (u, v) in E if v == sink
)
total_y = gp.quicksum(y[j] for j in J)

mdl.addConstr(
    inflow_sink == total_y,
    name="flow_sink_in"
)

# opcional: sink não envia fluxo
outflow_sink = gp.quicksum(
    xvar[(sink, v)] for (u, v) in E if u == sink
)
mdl.addConstr(
    outflow_sink == 0.0,
    name="flow_sink_out"
)

# ==============================
# Plot candidatos (antes de resolver)
# ==============================
plot_candidates_and_paths(
    F=J, q_fixed=q_fixed, q_sink=q_sink, R_comm=R_comm,
    mob_names=[], r_mobile=[], T=[0],
    region=region, out_path=RESULTS_PATH / "pic_candidates.jpg"
)

# Resolver
mdl.optimize()
status = mdl.Status
if status == GRB.INFEASIBLE:
    try:
        mdl.computeIIS()
        mdl.write("model.ilp")
    except Exception:
        pass
    raise RuntimeError("Modelo inviável. IIS salvo (se permitido) em 'model.ilp'.")

if status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
    raise RuntimeError(f"Modelo não resolveu para ótimo/subótimo. Status Gurobi: {status}")

# ==============================
# Pós-processamento e plots
# ==============================

# valores
y_val = {j: y[j].X for j in J}
installed = [j for j, v in y_val.items() if v > 0.5]

x_val = {(u, v): xvar[(u, v)].X for (u, v) in E}
z_val = {(u, v): z[(u, v)].X for (u, v) in E}

# Monta fixedMotes de saída
fixed_motes_out = []
fixed_motes_out.append({
    "position": [float(q_sink[0]), float(q_sink[1])],
    "name": "root",
    "sourceCode": "node.c"
})

count = 1
for j in installed:
    pos = q_fixed[j]
    fixed_motes_out.append({
        "position": [float(pos[0]), float(pos[1])],
        "name": f"node{count}",
        "sourceCode": "node.c"
    })
    count += 1

sim["simulationElements"]["fixedMotes"] = fixed_motes_out

with open(RESULTS_PATH / "output.json", "w", encoding="utf-8") as f:
    json.dump(sim, f, ensure_ascii=False, indent=4)

plot_solution(
    F=J, installed=installed, q_fixed=q_fixed, q_sink=q_sink, R_comm=R_comm, R_inter=R_interf,
    mob_names=[], T=[0], r_mobile=[],
    region=region, out_path=RESULTS_PATH / "pic_installed.png"
)

plot_installed_graph(
    installed=installed, q_fixed=q_fixed, q_sink=q_sink, R_comm=R_comm,
    region=region, out_path=RESULTS_PATH / "pic_installed_graph.png"
)

print("Done.")
