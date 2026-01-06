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

# SCIP
try:
    from pyscipopt import Model, quicksum
except Exception as e:
    raise RuntimeError(
        "Este script requer 'pyscipopt'. Instale SCIP + PySCIPOpt."
    ) from e

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

# ------------------------------
# Parâmetros do modelo (modelo mobile)
# ------------------------------
C0        = 10.0        # capacidade máxima nominal (C_0)
kdecay    = 0.1         # fator de atenuação (k_decay)
w_install = 1000.0**2   # peso w da função objetivo para instalação de motes

# Dicionário de motes
sink_name = None  # nome do sink
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

# Móveis e trajetórias discretas
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

def pos_node(n, t: int):
    """Posição espacial p_i(t) do nó i no instante t, conforme o modelo mobile."""
    if n[0] == "sink":
        return p_sink
    if n[0] == "j":   # candidato fixo
        return p_cand[n]
    if n[0] == "m":   # móvel
        return r_mobile(n[1], t)
    raise ValueError(f"Nó desconhecido: {n}")

# Demanda: cada móvel gera 1.0 unidade por tempo (b_{m,t} = 1)
b = {(name, t): 1.0 for name in mob_names for t in range(1, T + 1)}

# ==============================
# Modelo SCIP
# ==============================
mdl = Model("WSN_Mobile_Coverage_Problem")

# Verbosidade (similar ao OutputFlag=1)
# 0 = silencioso; 4/5 = razoável; mais alto = muito detalhado
mdl.setParam("display/verblevel", 4)

# --------------------------------------------
# Funções auxiliares: capacidade C_ij(t) e energia e_ij(t)
# --------------------------------------------
def capacity(pi: np.ndarray, pj: np.ndarray) -> float:
    """C_{ij}(t) = max{0, C0 * (1 - kdecay * d)^2} conforme o modelo mobile."""
    d = np.linalg.norm(pi - pj)
    return max(0.0, C0 * (1.0 - kdecay * d) ** 2)

def energy_cost(pi: np.ndarray, pj: np.ndarray) -> float:
    """
    e_{ij}(t) conforme o modelo:
        d^2,                       se 0 < d <= R_comm
        (R_interf - d)^2,          se R_comm < d <= R_interf
        0,                         caso contrário.
    No modelo mobile, (i,j) ∈ E_t implica d <= R_comm, então na prática
    entra sempre o primeiro ramo, mas mantemos a forma geral.
    """
    d = np.linalg.norm(pi - pj)
    if 0.0 < d <= R_comm:
        return d ** 2
    elif R_comm < d <= R_interf:
        return (R_interf - d) ** 2
    else:
        return 0.0

# --------------------------------------------
# Construção de E_t, C_{ij}(t), e_{ij}(t)
# --------------------------------------------
E_t = {}          # t -> lista de arestas (i,j)
C = {}            # (i,j,t) -> capacidade
e_cost = {}       # (i,j,t) -> e_{ij}(t)

for t in range(1, T + 1):
    nodes_t = [sink] + J + [("m", name) for name in mob_names]
    E_t[t] = []
    for i in nodes_t:
        for j in nodes_t:
            if i == j:
                continue
            pi, pj = pos_node(i, t), pos_node(j, t)
            d = np.linalg.norm(pi - pj)
            # Definição de E_t pelo raio de comunicação (0 < d <= R_comm)
            if 0.0 < d <= R_comm:
                cap = capacity(pi, pj)
                if cap > 0.0:
                    E_t[t].append((i, j))
                    C[(i, j, t)] = cap
                    e_cost[(i, j, t)] = energy_cost(pi, pj)

# --------------------------------------------
# Variáveis
#  - y_j: instalação de fixos (j ∈ J = F)
#  - z_ij(t): ativação da aresta (i,j) no slot t
#  - x_ij(t): fluxo na aresta (i,j) no slot t
# --------------------------------------------
y = {j: mdl.addVar(vtype="B", name=f"y_{j[1]}") for j in J}

z = {}
xvar = {}
for t in range(1, T + 1):
    for (i, j) in E_t[t]:
        z[(i, j, t)] = mdl.addVar(vtype="B", name=f"z_{i}_{j}_t{t}")
        xvar[(i, j, t)] = mdl.addVar(lb=0.0, vtype="C", name=f"x_{i}_{j}_t{t}")

# --------------------------------------------
# Objetivo (modelo mobile)
#   min  w * sum_j y_j  +  sum_t sum_(i,j) e_ij(t) * x_ij(t)
# --------------------------------------------
obj_install = w_install * quicksum(y[j] for j in J)
obj_flow = quicksum(
    e_cost[(i, j, t)] * xvar[(i, j, t)]
    for t in range(1, T + 1)
    for (i, j) in E_t[t]
)
mdl.setObjective(obj_install + obj_flow, "minimize")

# --------------------------------------------
# Restrições (modelo mobile)
# --------------------------------------------

# (1) Capacidade: 0 ≤ x_ij(t) ≤ C_ij(t) * z_ij(t)
for t in range(1, T + 1):
    for (i, j) in E_t[t]:
        mdl.addCons(
            xvar[(i, j, t)] <= C[(i, j, t)] * z[(i, j, t)],
            name=f"cap_{i}_{j}_t{t}"
        )

# (2) Instalação em fixos nas extremidades:
#     z_ij(t) ≤ y_i e z_ij(t) ≤ y_j quando i ou j ∈ J
for t in range(1, T + 1):
    for (i, j) in E_t[t]:
        if i[0] == "j":
            mdl.addCons(z[(i, j, t)] <= y[i], name=f"inst_i_{i}_{j}_t{t}")
        if j[0] == "j":
            mdl.addCons(z[(i, j, t)] <= y[j], name=f"inst_j_{i}_{j}_t{t}")

# (3) Conservação de fluxo nos móveis: sum_out - sum_in = b_{m,t}
for t in range(1, T + 1):
    for name in mob_names:
        m_node = ("m", name)
        outflow = quicksum(
            xvar[(m_node, j, t)] for (ii, j) in E_t[t] if ii == m_node
        )
        inflow = quicksum(
            xvar[(i, m_node, t)] for (i, jj) in E_t[t] if jj == m_node
        )
        mdl.addCons(outflow - inflow == b[(name, t)],
                    name=f"flow_mobile_{name}_t{t}")

# (4) Conservação de fluxo nos fixos: sum_out - sum_in = 0
for t in range(1, T + 1):
    for j_node in J:
        outflow = quicksum(
            xvar[(j_node, v, t)] for (u, v) in E_t[t] if u == j_node
        )
        inflow = quicksum(
            xvar[(u, j_node, t)] for (u, v) in E_t[t] if v == j_node
        )
        mdl.addCons(outflow - inflow == 0.0,
                    name=f"flow_fixed_{j_node}_t{t}")

# (5) Balanço no sink s: sum_in = sum_m b_{m,t}
for t in range(1, T + 1):
    inflow_s = quicksum(
        xvar[(i, sink, t)] for (i, j) in E_t[t] if j == sink
    )
    total_bt = quicksum(b[(name, t)] for name in mob_names)
    mdl.addCons(inflow_s == total_bt, name=f"flow_sink_t{t}")

# --------------------------------------------
# Plots de candidatos (mantidos)
# --------------------------------------------
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

plot_candidates_and_paths(
    F=J, q_fixed=p_cand, q_sink=p_sink, R_comm=R_comm,
    mob_names=mob_names, r_mobile=r_mobile, T=T, region=region,
    out_path=RESULTS_PATH / "pic_candidates.jpg"
)

# Resolver
mdl.optimize()
status = mdl.getStatus()

# Tratamento de status (equivalente ao Gurobi)
if status in ("infeasible", "infeasible_or_unbounded"):
    # SCIP pode gerar prova/diagnóstico dependendo da config; exportamos LP como fallback
    try:
        mdl.writeProblem(str(RESULTS_PATH / "model_infeasible.lp"))
    except Exception:
        pass
    raise RuntimeError(
        "Modelo inviável. (Se possível) LP salvo em 'output/model_infeasible.lp'."
    )

if status not in ("optimal", "gaplimit", "timelimit"):
    raise RuntimeError(f"Modelo não resolveu (status SCIP): {status}")

# ==============================
# Pós-processamento e plots (mantidos)
# ==============================

# valores
y_val = {j: mdl.getVal(y[j]) for j in J}
installed = [j for j, v in y_val.items() if v > 0.5]

x_val = {(i, j, t): mdl.getVal(xvar[(i, j, t)])
         for t in range(1, T + 1) for (i, j) in E_t[t]}
z_val = {(i, j, t): mdl.getVal(z[(i, j, t)])
         for t in range(1, T + 1) for (i, j) in E_t[t]}

fixed_motes_out = []
fixed_motes_out.append({
    "position": [float(p_sink[0]), float(p_sink[1])],
    "name": "root",
    "sourceCode": "node.c"
})
count = 1
for j_node in installed:
    pos = p_cand[j_node]
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
    F=J, installed=installed, q_fixed=p_cand, q_sink=p_sink,
    R_comm=R_comm, R_inter=R_interf,
    mob_names=mob_names, T=T, r_mobile=r_mobile,
    region=region, out_path=RESULTS_PATH / "pic_installed.png"
)

plot_installed_graph(
    installed=installed, q_fixed=p_cand, q_sink=p_sink, R_comm=R_comm,
    region=region, out_path=RESULTS_PATH / "pic_installed_graph.png"
)

save_routes_gif(
    installed, r_mobile, mob_names, p_sink, p_cand, R_comm,
    region, x_val, E_t, T, J, out_dir_path=RESULTS_PATH
)
save_routes2_gif(
    installed, r_mobile, mob_names, p_sink, p_cand, R_comm,
    region, x_val, E_t, T, J, out_dir_path=RESULTS_PATH
)

print("Done.")
