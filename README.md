# Mobile Wireless Sensor Network Coverage Problem

## Preprocessing

$$
d_{ij}(t)=\Vert p_i(t)-p_j(t) \Vert_2
$$


$$
A_{ij}(t)=\begin{cases}
1,\quad &d_{ij}(t)\le R_{\text{com}},\\
0, & \text{other cases}.
\end{cases}
$$


$$
C_{ij}(t)=\max\Big\lbrace 0, C_0\big(1-k_{decay}d_{ij}(t)\big)^2\Big\rbrace
$$


$$
e_{ij}(t) =
\begin{cases}
d_{ij}^2(t), & \text{if } 0 < d_{ij}(t) \le R_{\text{com}},\\
(R_{\text{inter}} - d_{ij}(t))^2, & \text{if } R_{\text{com}} < d_{ij}(t)\le R_{\text{inter}},\\
0, & \text{other cases}.
\end{cases}
$$


$$
\mathcal{E}_t=\big\lbrace(i,j)\in \mathcal{V}\times \mathcal{V}\ |\ 0<d_{ij}(t)\le R_{\text{com}},\ i\neq j\big\rbrace
$$


## MILP

$$
\begin{aligned}
  \min_{y,z,x}\quad 
    & w\sum_{j\in\mathcal J} y_j 
    + \sum_{t\in\mathcal T}\sum_{(i,j)\in \mathcal E_t}e_{ij}(t) x_{ij}(t)
    - \lambda\sum_{t\in\mathcal{T}}\sum_{m\in\mathcal{M}}g_m(t) \\
\text{s.a.}\quad
  & z_{ij}(t)\le y_i,\quad z_{ij}(t)\le y_j,
  && \forall (i,j)\in\mathcal E_t\cap(\mathcal{J}\times\mathcal{J}),\ \forall t\in\mathcal T,
  \\
  & z_{sj}(t) \le y_j,\quad z_{js}(t) \le y_j, 
  && \forall j\in\mathcal{J}\ | \ (s,j),(j,s)\in\mathcal{E}_t,\ \forall t\in\mathcal T,
  \\
  & 0\le x_{ij}(t)\le C_{ij}(t)\,z_{ij}(t), 
  && \forall (i,j)\in\mathcal E_t,\ \forall t\in\mathcal T,
  \\
  & \sum_{i:(m,i)\in \mathcal E_t} x_{mi}(t) - \sum_{i:(i,m)\in \mathcal E_t} x_{im}(t) = g_m(t),
  && \forall m\in\mathcal M,\ \forall t\in\mathcal T,
  \\
  & \sum_{i:(j,i)\in \mathcal E_t} x_{ji}(t) - \sum_{i:(i,j)\in \mathcal E_t} x_{ij}(t) = 0,
  && \forall j\in\mathcal J,\ \forall t\in\mathcal T,
  \\
  & \sum_{i:(i,s)\in \mathcal E_t} x_{i s}(t) = \sum_{m\in\mathcal M} g_m(t),
  && \forall t\in\mathcal T,
  \\
  & y_j\in\lbrace0,1\rbrace, 
  && \forall j\in\mathcal J,
  \\
  & z_{ij}(t)\in\lbrace0,1\rbrace,\quad x_{ij}(t)\ge 0, 
  && \forall (i,j)\in\mathcal{E}_t,\ \forall t\in\mathcal T
  \\
  & \alpha b_{m} \le g_m(t) \le b_{m}, && \forall m\in\mathcal{M}, \ \forall t\in\mathcal{T}.
\end{aligned}
$$

---

# Target Coverage Problem in Wireless Sensor Networks

## Preprocessing

$$
a_{hj} =
\begin{cases}
1, & \text{if } \Vert \xi_h - q_j\Vert_2 \le R_{\text{cov}},\\
0, & \text{other cases,}
\end{cases}
$$

## MILP

$$
\begin{aligned}
  \min_{y,z,x}\quad
  & w \sum_{j\in\mathcal{J}} y_j
    + \sum_{(i,j)\in\mathcal{E}} e_{ij}\,z_{ij}
    - \lambda\sum_{j\in\mathcal{J}}g_j \\
  \text{s.a.}\quad
  & z_{ij} \le y_i,\quad z_{ij} \le y_j,
  && \forall (i,j)\in\mathcal{E}\cap(\mathcal{J}\times\mathcal{J}), 
  \\
  & z_{sj} \le y_j,\quad z_{js} \le y_j, 
  && \forall j\in\mathcal{J}\,|\, (s,j),(j,s)\in\mathcal{E},
  \\
  & 0 \le x_{ij} \le M_{\max}\ z_{ij},
  && \forall (i,j)\in\mathcal{E}, 
  \\
  & \alpha G_{\max}y_j \le g_j \le G_{\max}y_j,
  && \forall j\in\mathcal{J},
  \\
  & \sum_{i:(j,i)\in\mathcal{E}} x_{ji}
      - \sum_{i:(i,j)\in\mathcal{E}} x_{ij}
      = g_j,
  && \forall j\in\mathcal{J}, 
  \\
  & \sum_{i:(i,s)\in\mathcal{E}} x_{is}
      = \sum_{j\in\mathcal{J}} g_j,
  && \text{(sink node)}, 
  \\
  & \sum_{j\in\mathcal{J}} a_{hj}\,y_j \;\ge\; \mathbf{k},
  && \forall h\in\mathcal{H}, 
  \\
  & \sum_{i:(i,j)\in\mathcal{E}} A_{ij}\,y_i
      \;\ge\; \mathbf{g}\,y_j,
  && \forall j\in\mathcal{J}, 
  \\
  & y_j \in \lbrace0,1\rbrace,\quad g_j \ge 0,
  && \forall j\in\mathcal{J},
  \\
  & z_{ij} \in \lbrace0,1\rbrace,\quad x_{ij} \ge 0,
  && \forall (i,j)\in\mathcal{E}.
\end{aligned}
$$

---


