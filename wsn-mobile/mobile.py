import numpy as np
import json
from pathlib import Path

# Bibliotecas locais
from utils.sim_utils import load_simulation_json
from utils.sim_utils import add_random_fixed_motes
from utils.sim_utils import make_mobile_trajectory_fn
from utils.plot_utils import plot_installed_graph
from utils.plot_utils import plot_candidates_and_paths
from utils.plot_utils import plot_solution
from utils.gif_utils import save_routes_gif, save_routes2_gif

# Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as e:
    raise RuntimeError("Este script requer 'gurobipy'. Instale e garanta uma licença ativa do Gurobi.") from e

# Parâmetros do programa
SIM_JSON_PATH = "./input.json"   # ajuste conforme necessário
RESULTS_PATH = Path("./output")
ADD_RANDOM = True                # True para injetar candidatos aleatórios

# Inicializações de parâmetros adicionais
# 1) Carrega JSON base
sim = load_simulation_json(SIM_JSON_PATH)

# 2) Adiciona fixos aleatórios (opcional)
if ADD_RANDOM:
    sim = add_random_fixed_motes(sim, n_new=50, seed=71)

# Dados de entrada advindos do arquivo JSON
duration = int(sim.get("duration", 60))
mobile_ts = [int(m.get("timeStep", 1)) for m in sim["simulationElements"]["mobileMotes"]]
dt = max(1, min(mobile_ts) if mobile_ts else 1)
T = max(1, duration // dt)

R_comm   = float(sim.get("radiusOfReach", 50.0))
R_interf = float(sim.get("radiusOfInter", 60.0))
region   = sim.get("region", [-200, -200, 200, 200])

fixed_list  = sim["simulationElements"]["fixedMotes"]
mobile_list = sim["simulationElements"]["mobileMotes"]

# Parâmetros do modelo
cap0 = 10
kdecay = 0.1
alpha_yx = 1000.0**2

# Dicionário de motes
sink_name = None # sink
for fm in fixed_list:
    if fm.get("name", "").lower() == "root":
        sink_name = fm["name"]
        break
if sink_name is None and fixed_list:
    sink_name = fixed_list[0]["name"]

sink = ("sink", sink_name)
p_sink = None
J = []           # fixos candidatos (exclui sink)
p_cand = {}      # ("j", name) -> np.array([x,y])

for fm in fixed_list:
    name = str(fm["name"])
    pos = np.array(fm["position"], dtype=float)
    if name == sink_name:
        p_sink = pos
    else:
        J.append(("j", name))
        p_cand[("j", name)] = pos

if p_sink is None:
    raise ValueError("Não foi possível determinar a posição do sink.")

# móveis e trajetórias discretas
mob_names = [m["name"] for m in mobile_list]
r_mobile_by_name = {}
for m in mobile_list:
    name      = m["name"]
    fpath     = m["functionPath"]
    is_closed = bool(m.get("isClosed", False))
    is_round  = bool(m.get("isRoundTrip", False))
    speed     = float(m.get("speed", 1.0))
    r_fn      = make_mobile_trajectory_fn(fpath, is_closed, is_round, T, speed)
    r_mobile_by_name[name] = r_fn

def r_mobile(name: str, tau: int):
    return r_mobile_by_name[name](tau)

def pos_node(n, t):
    if n[0] == "sink":
        return p_sink
    if n[0] == "j":
        return p_cand[n]
    if n[0] == "m":
        return r_mobile(n[1], t)
    raise ValueError("nó desconhecido")

# demanda: cada móvel gera 1.0 unidade por tempo
b = {(name, t): 1.0 for name in mob_names for t in range(1, T + 1)}

# ==============================
# Modelo Gurobi
# ==============================

mdl = gp.Model("WSN_Mobile_Coverage_Problem")
mdl.Params.OutputFlag = 1  # 0 para silenciar logs

def link_possible(pi, pj):
    return np.linalg.norm(pi - pj) <= R_comm

def capacity(pi, pj):
    d = np.linalg.norm(pi - pj)
    return max(0.0, cap0 * (1 - kdecay * d) ** 2)

def link_cost(pi, pj):
    N = 2
    delta = 0.5
    d = np.linalg.norm(pi - pj)    
    noise = np.random.uniform(-delta, delta)
    if 0 < d <= R_comm:
        return d**N + noise
    elif R_comm < d <= R_interf:
        return (R_interf - d)**N + noise
    else:
        return 0

E_t = {}
A = {}
C = {}
cost = {}

for t in range(1, T+1):
    nodes_t = [sink] + J + [("m", name) for name in mob_names]
    E_t[t] = []
    for i in nodes_t:
        for j in nodes_t:
            if i == j:
                continue
            pi, pj = pos_node(i, t), pos_node(j, t)
            feas = link_possible(pi, pj)
            A[(i, j, t)] = 1 if feas else 0
            if feas:
                cap = capacity(pi, pj)
                if cap > 0:
                    E_t[t].append((i, j))
                    C[(i, j, t)] = cap
                    cost[(i, j, t)] = link_cost(pi, pj)

# --------------------------------------------
# Variáveis
#  - y_j: instalação de fixos (j ∈ J = F)
#  - z_ij(t): ativação da aresta (i,j) no slot t
#  - x_ij(t): fluxo na aresta (i,j) no slot t
# --------------------------------------------
y = {j: mdl.addVar(vtype=GRB.BINARY, name=f"y_{j[1]}") for j in J}  # j é ("j", name)

z = {}
xvar = {}
for t in range(1, T + 1):
    for (i, j) in E_t[t]:
        z[(i, j, t)] = mdl.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}_t{t}")
        xvar[(i, j, t)] = mdl.addVar(lb=0.0, name=f"x_{i}_{j}_t{t}")

mdl.update()

# --------------------------------------------
# Objetivo (Modelo 2)
#   min  alpha_yx * sum_j (alpha_{inter} * k_j + 1) * y_j  +  sum_t sum_(i,j) dtilde_ij(t) * x_ij(t)
# --------------------------------------------
obj_install = gp.quicksum((y[j] + alpha_yx * y[j]) for j in J)
obj_flow = gp.quicksum(
    cost[(i, j, t)] * xvar[(i, j, t)]
    for t in range(1, T + 1)
    for (i, j) in E_t[t]
)
mdl.setObjective(obj_install + obj_flow, GRB.MINIMIZE)

# --------------------------------------------
# Restrições
# --------------------------------------------

# (1) Existência / viabilidade do link: z_ij(t) ≤ A_ij(t)
for t in range(1, T + 1):
    for (i, j) in E_t[t]:
        mdl.addConstr(z[(i, j, t)] <= A[(i, j, t)], name=f"exist_{i}_{j}_t{t}")

# (2) Instalação em fixos nas extremidades: z_ij(t) ≤ y_i e z_ij(t) ≤ y_j quando i ou j ∈ J
#     (apenas quando a ponta é fixa; não há y para sink ou móveis)
for t in range(1, T + 1):
    for (i, j) in E_t[t]:
        if i[0] == "j":  # i é um candidato fixo
            mdl.addConstr(z[(i, j, t)] <= y[i], name=f"inst_i_{i}_{j}_t{t}")
        if j[0] == "j":  # j é um candidato fixo
            mdl.addConstr(z[(i, j, t)] <= y[j], name=f"inst_j_{i}_{j}_t{t}")

# (3) Capacidade: 0 ≤ x_ij(t) ≤ C_ij(t) * z_ij(t)
for t in range(1, T + 1):
    for (i, j) in E_t[t]:
        mdl.addConstr(
            xvar[(i, j, t)] <= C[(i, j, t)] * z[(i, j, t)],
            name=f"cap_{i}_{j}_t{t}"
        )

# (4) Conservação de fluxo nos móveis: sum_out - sum_in = b_{m,t}
for t in range(1, T + 1):
    for name in mob_names:
        m_node = ("m", name)
        outflow = gp.quicksum(xvar[(m_node, j, t)] for (ii, j) in E_t[t] if ii == m_node)
        inflow  = gp.quicksum(xvar[(i, m_node, t)] for (i, jj) in E_t[t] if jj == m_node)
        mdl.addConstr(outflow - inflow == b[(name, t)], name=f"flow_mobile_{name}_t{t}")

# (5) Conservação de fluxo nos fixos: sum_out - sum_in = 0
for t in range(1, T + 1):
    for i in J:
        outflow = gp.quicksum(xvar[(i, j, t)] for (ii, j) in E_t[t] if ii == i)
        inflow  = gp.quicksum(xvar[(j, i, t)] for (j, jj) in E_t[t] if jj == i)
        mdl.addConstr(outflow - inflow == 0.0, name=f"flow_fixed_{i}_t{t}")

# (6) Balanço no sink s: sum_in = sum_m b_{m,t}
for t in range(1, T + 1):
    inflow_s  = gp.quicksum(xvar[(i, sink, t)] for (i, j) in E_t[t] if j == sink)
    total_bt  = gp.quicksum(b[(name, t)] for name in mob_names)
    mdl.addConstr(inflow_s == total_bt, name=f"flow_sink_t{t}")

# plot candidatos
plot_candidates_and_paths(
    F=J, q_fixed=p_cand, q_sink=p_sink, R_comm=R_comm,
    mob_names=mob_names, r_mobile=r_mobile, T=T, region=region, out_path=RESULTS_PATH / "pic_candidates.jpg"
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
x_val = {(i, j, t): xvar[(i, j, t)].X for t in range(1, T + 1) for (i, j) in E_t[t]}
z_val = {(i, j, t): z[(i, j, t)].X for t in range(1, T + 1) for (i, j) in E_t[t]}

fixed_motes_out = []                 
fixed_motes_out.append({
    "position": [0.0, 0.0],
    "name": "root",
    "sourceCode": "node.c"
})
count = 1
for j in installed:
    pos = p_cand[j]                 
    name = j[1]                     
    fixed_motes_out.append({
        "position": [float(pos[0]), float(pos[1])],
        "name": f"node{count}",
        "sourceCode": "node.c"
    })
    count+=1

sim["simulationElements"]["fixedMotes"] = fixed_motes_out

with open(RESULTS_PATH / "output.json", "w", encoding="utf-8") as f:
    json.dump(sim, f, ensure_ascii=False, indent=4)
    
plot_solution(
    F=J, installed=installed, q_fixed=p_cand, q_sink=p_sink, R_comm=R_comm, R_inter=R_interf,
    mob_names=mob_names, T=T, r_mobile=r_mobile, 
    region=region, out_path=RESULTS_PATH / "pic_installed.png"
)

plot_installed_graph(
    installed=installed, q_fixed=p_cand, q_sink=p_sink, R_comm=R_comm,
    region=region, out_path=RESULTS_PATH /"pic_installed_graph.png"
)

save_routes_gif(installed, r_mobile, mob_names, p_sink, p_cand, R_comm, 
                region, x_val, E_t, T, J, out_dir_path=RESULTS_PATH)
save_routes2_gif(installed, r_mobile, mob_names, p_sink, p_cand, R_comm, 
                 region, x_val, E_t, T, J, out_dir_path=RESULTS_PATH)

print("Done.")
