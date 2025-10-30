# Modelo Inicial

## Definições
$K_j=\{q_i\in Q:R_{com}\le\|q_i-q_j\|_2\le R_{inter}\}$

$k_j=|K_j|$

## Modelo
$$
\begin{aligned}
\min_{y,z,x}\quad
& \sum_{j\in\mathcal J} k_j y_j
+ \sum_{t\in\mathcal T}
  \sum_{(i,j)\in \mathcal E_t}
  d_{ij}(t) x_{ij}(t)
\\[0.4em]
\text{s.a.}\quad
& z_{ij}(t)\le A_{ij}(t),
&& \forall (i,j)\in\mathcal E_t,\ \forall t,
\\[0.25em]
& z_{ij}(t)\le y_i,\quad z_{ij}(t)\le y_j,
&& \forall (i,j)\in\mathcal E_t,\ i,j\in\mathcal J,\ \forall t,
\\[0.25em]
& 0\le x_{ij}(t)\le C\,z_{ij}(t),
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
