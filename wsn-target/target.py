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

# 1. Carrega JSON base
sim = load_simulation_json(SIM_JSON_PATH)

# ==============================
# Construção dos conjuntos/posições a partir do JSON
# ==============================

# tempo (não usado diretamente neste modelo, mas mantido)
duration = int(sim.get("duration", 60))

R_comm   = float(sim.get("radiusOfReach", 100.0))
R_interf = float(sim.get("radiusOfInter", 200.0))
region   = sim.get("region", [-200, -200, 200, 200])

# raio de cobertura (R_cov). Se não existir no JSON, usa R_comm como fallback.
R_cov = float(sim.get("radiusOfCov", R_comm))

fixed_list  = sim["simulationElements"]["fixedMotes"]

# parâmetros do modelo (modelo target)
w_install = float(sim.get("w_install", 1e6))    # peso w da instalação no objetivo
k_cov     = int(sim.get("k_coverage", 2))       # nível de cobertura k
g_conn    = int(sim.get("g_connectivity", 1))   # nível de conectividade g
M_max     = float(sim.get("M_max", 1e4))        # big-M global para as capacidades de fluxo
lambda_thr = float(sim.get("lambda_thr", 1.0))  # λ do termo de throughput no objetivo
G_max      = float(sim.get("G_max", 3000.0))    # G_max: limite superior de g_j quando y_j=1
alpha = float(sim.get("alpha_thr", 0.9))

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
    """Retorna a posição (x,y) do nó n, conforme o modelo target."""
    if n[0] == "sink":
        return q_sink
    if n[0] == "f":
        return q_fixed[n]
    raise ValueError(f"nó desconhecido: {n}")

# ==============================
# Conjunto de alvos (H) para cobertura
# ==============================
targets_raw = sim.get("targets", [])  # opcional; adapte ao seu JSON
H = []           # índices dos alvos
q_target = {}    # h -> np.array([x,y])

for idx, tgt in enumerate(targets_raw):
    pos = np.array(tgt["position"], dtype=float)
    H.append(idx)
    q_target[idx] = pos

# ==============================
# Funções auxiliares
# ==============================

def energy_cost(pi: np.ndarray, pj: np.ndarray) -> float:
    """
    e_{ij} como no modelo:
        d^2,                    se 0 < d <= R_comm
        (R_interf - d)^2,       se R_comm < d <= R_interf
        0,                      caso contrário.

    Como só criamos arestas com d <= R_comm, entra sempre o primeiro caso,
    mas mantemos a forma geral para coerência com o texto.
    """
    d = np.linalg.norm(pi - pj)
    if 0.0 < d <= R_comm:
        return d ** 2
    elif R_comm < d <= R_interf:
        return (R_interf - d) ** 2
    else:
        return 0.0

# ==============================
# Pré-processamento de arestas
# ==============================
A = {}      # matriz de adjacência geométrica A_{ij}
e_cost = {} # e_{ij}
E = []      # conjunto de arestas viáveis (i,j)

nodes = [sink] + J

for i in nodes:
    for j in nodes:
        if i == j:
            continue
        pi, pj = pos_node(i), pos_node(j)
        d = np.linalg.norm(pi - pj)
        if 0.0 < d <= R_comm:
            # aresta viável
            A[(i, j)] = 1
            E.append((i, j))
            e_cost[(i, j)] = energy_cost(pi, pj)
        else:
            A[(i, j)] = 0

# ==============================
# Modelo Gurobi (modelo target)
# ==============================
mdl = gp.Model("WSN_Target_Coverage_Problem")
mdl.Params.OutputFlag = 1  # 0 para silenciar logs

# --------------------------------------------
# Variáveis
#  - y_j: instalação de fixos (j ∈ J)
#  - z_ij: ativação da aresta (i,j)
#  - x_ij: fluxo na aresta (i,j)
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
    
# g_j (throughput gerado pelo nó j)
g = {
    j: mdl.addVar(lb=0.0, ub=G_max, name=f"g_{j[0]}_{j[1]}")
    for j in J
}

mdl.update()

# --------------------------------------------
# Objetivo (modelo target)
#   min  w * sum_j y_j  +  sum_(i,j) e_ij z_ij  - lambda * sum_j g_j
# --------------------------------------------
obj_install = gp.quicksum(y[j] for j in J)
obj_edges   = gp.quicksum(e_cost[(u, v)] * z[(u, v)] for (u, v) in E)
obj_thr     = gp.quicksum(g[j] for j in J)

mdl.setObjective(w_install * obj_install + obj_edges - lambda_thr * obj_thr, GRB.MINIMIZE)

# --------------------------------------------
# Restrições
# --------------------------------------------

# (k-cobertura)  sum_j a_hj y_j >= k,  ∀ h∈H
# com a_hj = 1 se dist(target_h, fixed_j) <= R_cov
if H:
    for h in H:
        pos_h = q_target[h]
        covering_sensors = [
            j for j in J
            if np.linalg.norm(pos_h - q_fixed[j]) <= R_cov
        ]
        if covering_sensors:
            mdl.addConstr(
                gp.quicksum(y[j] for j in covering_sensors) >= k_cov,
                name=f"cov_target_{h}"
            )
        else:
            # Nenhum candidato cobre este alvo; se quiser, pode forçar inviabilidade.
            pass

# (g-conectividade local):
#   ∑_{i:(i,j)∈E} A_ij y_i ≥ g * y_j,  ∀ j∈J
for j in J:
    incident = gp.quicksum(
        y[i] for i in J if i != j and A.get((i, j), 0) == 1
    )
    mdl.addConstr(
        incident >= g_conn * y[j],
        name=f"g_conn_{j[1]}"
    )
    
# (g bound) alpha * y_i <= g_j <= G_max * y_j
for j in J:
    mdl.addConstr(
        g[j] <=  G_max * y[j],
        name=f"g_le_Gmax_y_{j[1]}"
    )
    mdl.addConstr(
        g[j] >= alpha * G_max * y[j],
        name=f"g_ge_alpha_y_{j[1]}"
    )

# (Link só pode ativar se extremos instalados)
# z_ij <= y_i, z_ij <= y_j para (i,j) com i,j ∈ J
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

# (Capacidade de fluxo - big-M global)
# 0 <= x_ij <= M_max * z_ij,  ∀ (i,j)∈E
for (u, v) in E:
    mdl.addConstr(
        xvar[(u, v)] <= M_max * z[(u, v)],
        name=f"cap_{u[1]}__{v[1]}"
    )

# (Conservação de fluxo nos fixos):
#   ∑_{v:(j,v)∈E} x_jv - ∑_{u:(u,j)∈E} x_uj = y_j , ∀ j∈J
for j in J:
    outflow = gp.quicksum(
        xvar[(j, v)] for (u, v) in E if u == j
    )
    inflow = gp.quicksum(
        xvar[(u, j)] for (u, v) in E if v == j
    )
    mdl.addConstr(
        outflow - inflow == g[j],
        name=f"flow_cons_{j[1]}"
    )

# (Nó sink):
#   ∑_{u:(u,s)∈E} x_us = ∑_{j∈J} y_j
inflow_sink = gp.quicksum(
    xvar[(u, sink)] for (u, v) in E if v == sink
)

total_g = gp.quicksum(g[j] for j in J)
mdl.addConstr(inflow_sink == total_g, name="flow_sink_in")


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
    mob_names=[], r_mobile=None, T=0,
    region=region, out_path=RESULTS_PATH / "pic_candidates.jpg",
    targets=H, q_target=q_target, R_cov=R_cov,
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
    F=J, installed=installed, q_fixed=q_fixed, q_sink=q_sink,
    R_comm=R_comm, R_inter=R_interf,
    mob_names=[], T=0, r_mobile=None,
    region=region, out_path=RESULTS_PATH / "pic_installed.png",
    targets=H, q_target=q_target, R_cov=R_cov,
)

plot_installed_graph(
    installed=installed, q_fixed=q_fixed, q_sink=q_sink, R_comm=R_comm,
    region=region, out_path=RESULTS_PATH / "pic_installed_graph.png",
    targets=H, q_target=q_target,
)

print("Done.")
