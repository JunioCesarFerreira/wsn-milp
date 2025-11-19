# Modelos Iniciais

## Problema com Mobilidade

### Pré-processamento

$$
d_{ij}(t)=\|p_i(t)-p_j(t)\|_2
$$

$$
A_{ij}(t)=\begin{cases}
1,\quad &d_{ij}(t)\le R_{com},\\
0, & \text{caso contrário}.
\end{cases}
$$

$$
C_{ij}(t)=\max\Big\{0, C_0\big(1-k_{decay}d_{ij}(t)\big)^2\Big\}
$$


### Modelo 1
$$
\begin{aligned}
\min_{y,z,x}\quad
& \alpha_{yx}\sum_{j\in\mathcal J} y_j
+ \sum_{t\in\mathcal T}
  \sum_{(i,j)\in \mathcal E_t}
  d_{ij}^2(t) x_{ij}(t)
\\
\text{s.a.}\quad
& z_{ij}(t)\le A_{ij}(t),
&& \forall (i,j)\in\mathcal E_t,\ \forall t,
\\
& z_{ij}(t)\le y_i,\quad z_{ij}(t)\le y_j,
&& \forall (i,j)\in\mathcal E_t,\ i,j\in\mathcal J,\ \forall t,
\\
& 0\le x_{ij}(t)\le C_{ij}(t)\,z_{ij}(t),
&& \forall (i,j)\in\mathcal E_t,\ \forall t,
\\
& \sum_{j:(m,j)\in \mathcal E_t} x_{mj}(t)
  - \sum_{i:(i,m)\in \mathcal E_t} x_{im}(t)
  = b_{m,t},
&& \forall m\in\mathcal M,\ \forall t,
\\
& \sum_{j:(i,j)\in \mathcal E_t} x_{ij}(t)
  - \sum_{k:(k,i)\in \mathcal E_t} x_{ki}(t)
  = 0,
&& \forall i\in\mathcal J,\ \forall t,
\\
& \sum_{i:(i,s)\in \mathcal E_t} x_{i s}(t)
  = \sum_{m\in\mathcal M} b_{m,t},
&& \forall t,
\\
& y_j\in\{0,1\},\quad z_{ij}(t)\in\{0,1\},\quad x_{ij}(t)\ge 0.
\end{aligned}
$$

---

## Incluindo Interferência Custo Local

### Pré-processamento

$$
d_{ij}(t)=\|p_i(t)-p_j(t)\|_2
$$

$$
A_{ij}(t)=\begin{cases}
1,\quad &d_{ij}(t)\le R_{com},\\
0, & \text{caso contrário}.
\end{cases}
$$

$$
C_{ij}(t)=\max\Big\{0, C_0\big(1-k_{decay}d_{ij}(t)\big)^2\Big\}
$$

### Penalizando Interferência

$$
\Eta_j=\big\{q_i\in Q\;\big|\;R_{\text{com}}\le\|q_i-q_j\|_2\le R_{\text{inter}}\big\},\quad\forall j\in\mathcal J
$$

$$
\eta_j=|\Eta_j|+1
$$

### Modelo 2
$$
\begin{aligned}
\min_{y,z,x}\quad
& \alpha_{yx}\sum_{j\in\mathcal J} (\alpha_{\text{inter}} \eta_j+1)y_j
+ \sum_{t\in\mathcal T}
  \sum_{(i,j)\in \mathcal E_t}
  d_{ij}^2(t) x_{ij}(t)
\\[0.4em]
\text{s.a.}\quad
& z_{ij}(t)\le A_{ij}(t),
&& \forall (i,j)\in\mathcal E_t,\ \forall t,
\\[0.25em]
& z_{ij}(t)\le y_i,\quad z_{ij}(t)\le y_j,
&& \forall (i,j)\in\mathcal E_t,\ i,j\in\mathcal J,\ \forall t,
\\[0.25em]
& 0\le x_{ij}(t)\le C_{ij}(t)\,z_{ij}(t),
&& \forall (i,j)\in\mathcal E_t,\ \forall t,
\\[0.25em]
& \sum_{j:(m,j)\in \mathcal E_t} x_{mj}(t)
  - \sum_{i:(i,m)\in \mathcal E_t} x_{im}(t)
  = b_{m,t},
&& \forall m\in\mathcal M,\ \forall t,
\\[0.25em]
& \sum_{j:(i,j)\in \mathcal E_t} x_{ij}(t)
  - \sum_{k:(k,i)\in \mathcal E_t} x_{ki}(t)
  = 0,
&& \forall i\in\mathcal J,\ \forall t,
\\[0.25em]
& \sum_{i:(i,s)\in \mathcal E_t} x_{i s}(t)
  = \sum_{m\in\mathcal M} b_{m,t},
&& \forall t,
\\[0.25em]
& y_j\in\{0,1\},\quad z_{ij}(t)\in\{0,1\},\quad x_{ij}(t)\ge 0.
\end{aligned}
$$

---

## Incluindo Interferência como Peso

### Pré-processamento

$$
d_{ij}(t)=\|p_i(t)-p_j(t)\|_2
$$

$$
A_{ij}(t)=\begin{cases}
1,\quad &d_{ij}(t)\le R_{com},\\
0, & \text{caso contrário}.
\end{cases}
$$

$$
C_{ij}(t)=\max\Big\{0, C_0\big(1-k_{decay}d_{ij}(t)\big)^2\Big\}
$$

### Penalizando Interferência

$$
w_{ij}(t):=\begin{cases}
d_{ij}(t)^2,\quad&\text{se }0<d_{ij}(t)\le R_C,\\
(R_I-d_{ij}(t))^2, &\text{se }R_C<d_{ij}(t)\le R_I,\\
0, &\text{caso contrário.}
\end{cases}
$$


### Modelo 3
$$
\begin{aligned}
\min_{y,z,x}\quad
& \alpha_{yx}\sum_{j\in\mathcal J} y_j
+ \sum_{t\in\mathcal T}
  \sum_{(i,j)\in \mathcal E_t}
  w_{ij}(t) x_{ij}(t)
\\[0.4em]
\text{s.a.}\quad
& z_{ij}(t)\le A_{ij}(t),
&& \forall (i,j)\in\mathcal E_t,\ \forall t,
\\[0.25em]
& z_{ij}(t)\le y_i,\quad z_{ij}(t)\le y_j,
&& \forall (i,j)\in\mathcal E_t,\ i,j\in\mathcal J,\ \forall t,
\\[0.25em]
& 0\le x_{ij}(t)\le C_{ij}(t)\,z_{ij}(t),
&& \forall (i,j)\in\mathcal E_t,\ \forall t,
\\[0.25em]
& \sum_{j:(m,j)\in \mathcal E_t} x_{mj}(t)
  - \sum_{i:(i,m)\in \mathcal E_t} x_{im}(t)
  = b_{m,t},
&& \forall m\in\mathcal M,\ \forall t,
\\[0.25em]
& \sum_{j:(i,j)\in \mathcal E_t} x_{ij}(t)
  - \sum_{k:(k,i)\in \mathcal E_t} x_{ki}(t)
  = 0,
&& \forall i\in\mathcal J,\ \forall t,
\\[0.25em]
& \sum_{i:(i,s)\in \mathcal E_t} x_{i s}(t)
  = \sum_{m\in\mathcal M} b_{m,t},
&& \forall t,
\\[0.25em]
& y_j\in\{0,1\},\quad z_{ij}(t)\in\{0,1\},\quad x_{ij}(t)\ge 0.
\end{aligned}
$$

---

# Problema Estático

### Modelo 1

$$
\begin{aligned}
  \min_{y,z,x}\quad
  & \alpha \sum_{j\in\mathcal{J}} y_j
    + \sum_{(u,v)\in\mathcal{E}} w_{uv}\,z_{uv}\\
  \text{s.a.}\quad
  & \sum_{j\in\mathcal{J}} a_{ij}\,y_j \;\ge\; k,
    && \forall i\in\mathcal{T},\\
  & z_{uv} \le y_u,\quad
    z_{uv} \le y_v,
    && \forall (u,v)\in\mathcal{E}\cap(\mathcal{J}\times\mathcal{J}),\\
  & z_{sj} \le y_j,\quad z_{js} \le y_j,
    && \forall j\in\mathcal{J} \text{, } (s,j),(j,s)\in\mathcal{E},\\
  & 0 \le x_{uv} \le M\,z_{uv},
    && \forall (u,v)\in\mathcal{E},\\
  & \sum_{v:(j,v)\in\mathcal{E}} x_{jv}
    - \sum_{u:(u,j)\in\mathcal{E}} x_{uj}
    = y_j,
    && \forall j\in\mathcal{J},\\
  & \sum_{u:(u,s)\in\mathcal{E}} x_{us}
    = \sum_{j\in\mathcal{J}} y_j,
    && \text{(nó sink)},\\
  & y_j \in \{0,1\},
    && \forall j\in\mathcal{J},\\
  & z_{uv} \in \{0,1\},
    && \forall (u,v)\in\mathcal{E},\\
  & f_{uv} \ge 0,
    && \forall (u,v)\in\mathcal{E}.
\end{aligned}
$$

---

## Tarefa PDDL

- Incluir variações de caminhos disparados por eventos.
- Usar PDDL e ROS2 para realizar planejametos em tempo de execução.



---

Trechos que ainda vou usar...

$$
w_{ij}(t):=\begin{cases}
10 n \log_{10}(d_{ij}(t)),\quad&\text{se }0<d_{ij}(t)\le R_C,\\
10 n \log_{10}(R_I-d_{ij}(t)), &\text{se }R_C<d_{ij}(t)\le R_I,\\
0, &\text{caso contrário.}
\end{cases}
$$

$$
I_{(i,j),(k,l)}(t) =
\begin{cases}
1, & \text{se } (i,j)\neq(k,l)\ \text{e}\ 
\min\{\|p_i(t)-p_k(t)\|_2,\|p_j(t)-p_l(t)\|_2\}<R_\text{inter},\\[0.25em]
0, & \text{caso contrário.}
\end{cases}
$$
