## Relatório Comparativo de Solvers MILP

**Gurobi × HiGHS × SCIP**
*Aplicado a modelos de Wireless Sensor Networks com mobilidade*

### 1. Contexto Experimental

Os solvers foram avaliados na resolução de modelos MILP formulados para **problemas de cobertura e conectividade em Wireless Sensor Networks (WSN)**, incluindo um **modelo com mobilidade temporal**, caracterizado por:

* Estrutura de **fluxo em rede** com conservação por tempo;
* Variáveis binárias de **seleção de nós** e **ativação de arestas**;
* Variáveis contínuas de **fluxo** acopladas a binárias via restrições do tipo *big-M*;
* Forte dependência de **geometria euclidiana** e decomposição temporal.

O modelo apresenta, portanto, uma combinação de *network design*, *facility location* e *multi-period flow*, reconhecidamente desafiadora para solvers MILP genéricos.

---

### 2. Resultados Observados

Em uma instância representativa do **problema com mobilidade**:

* **Gurobi** resolveu o modelo até otimalidade em aproximadamente **172 segundos**.
* **SCIP** e **HiGHS** **não conseguiram concluir a resolução** (nem provar otimalidade nem retornar solução final) após **mais de 5 minutos de execução**, momento em que os testes foram interrompidos manualmente.

Este comportamento foi consistente com execuções adicionais, variando apenas marginalmente conforme parâmetros internos dos solvers.

---

### 3. Análise Comparativa

#### 3.1 Gurobi

* **Desempenho**: Excelente; único solver capaz de resolver a instância dentro de tempo aceitável.
* **Principais diferenciais**:

  * Presolve extremamente agressivo;
  * Geração avançada de cortes específicos para problemas de rede e fluxo;
  * Heurísticas primais eficazes, encontrando boas soluções rapidamente;
  * Paralelismo interno altamente otimizado.
* **Limitação**: Solver proprietário, dependente de licença.

**Conclusão**: Altamente adequado como *baseline* e para experimentos finais de alta complexidade.

---

#### 3.2 SCIP

* **Desempenho**: Insatisfatório para o modelo considerado.
* **Comportamento observado**:

  * Crescimento lento da árvore de branch-and-bound;
  * Dificuldade em obter boas soluções incumbentes iniciais;
  * Alto custo computacional na prova de otimalidade.
* **Observação técnica**:

  * Embora o SCIP seja um dos solvers open-source mais avançados, sua abordagem genérica sofre em problemas com:

    * grande número de binárias fracas;
    * relaxações lineares pouco informativas;
    * acoplamento forte entre decisões discretas e fluxo contínuo.

**Conclusão**: Adequado para MILPs genéricos ou menores, mas pouco competitivo neste tipo específico de problema WSN com mobilidade.

---

#### 3.3 HiGHS

* **Desempenho**: Similar ao SCIP neste experimento, não resolvendo a instância no tempo observado.
* **Comportamento observado**:

  * Boa performance em relaxações lineares;
  * Dificuldade significativa na fase MIP;
  * Branch-and-bound pouco eficaz frente à estrutura do problema.
* **Observação técnica**:

  * O foco principal do HiGHS ainda é LP/QP;
  * O suporte a MIP, embora em evolução, não explora profundamente estruturas de rede complexas e temporais.

**Conclusão**: Promissor como solver open-source moderno, mas ainda imaturo para MILPs grandes e altamente estruturados como os considerados neste trabalho.

---

### 4. Síntese Comparativa

| Solver | Licença      | Resolveu a instância? | Tempo observado | Avaliação geral |
| ------ | ------------ | --------------------- | --------------- | --------------- |
| Gurobi | Proprietária | Sim                   | ~172 s          | Excelente       |
| SCIP   | Open-source  | Não                   | > 300 s         | Limitado        |
| HiGHS  | Open-source  | Não                   | > 300 s         | Limitado        |

---

### 5. Conclusão Geral

Os resultados indicam que, **para modelos MILP de WSN com mobilidade, grande escala temporal e acoplamento fluxo–ativação**, o uso de solvers MILP genéricos open-source (SCIP e HiGHS) apresenta **limitações práticas significativas**, enquanto o Gurobi demonstra clara superioridade em termos de desempenho.

Essas observações reforçam que:

* O gargalo não está na formulação, mas na **estrutura intrinsecamente difícil do problema**;
* Solvers genéricos não exploram plenamente a semântica de rede e mobilidade;
* Abordagens alternativas, como **decomposição**, **reformulação de rede** ou **solvers específicos/híbridos**, são caminhos naturais para escalabilidade.

---
