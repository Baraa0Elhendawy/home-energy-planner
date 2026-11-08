# Home Energy Planner: Final Report

### 1. Problem Definition
The Home Energy Planner is an integrated discrete optimization system designed to manage a household's daily electricity consumption. The objective is to strategically schedule flexible appliances and modulate a home battery to minimize financial cost under a dynamic Time-of-Use tariff. This must be achieved without the combined total load exceeding a strict physical grid limit.

### 2. Design Choices
The architecture decouples the problem into three independent inference engines:
* **Combinatorial Scheduling (CSP):** Handles topological constraints. Variables are appliances, domains are start hours (0-23). Arc Consistency (AC-3) proactively prunes domain values that violate structural constraints (precedence, shared circuits), drastically reducing the search space before Backtracking.
* **Pathfinding (Search):** Evaluates the economic state space `(hour, battery_level)`. A* Search is utilized with a strictly admissible heuristic, ensuring the algorithm never overestimates the true cost and guarantees mathematical optimality for the battery routing.

### 3. Neural Network
To bound the optimization problem, a Multi-Layer Perceptron (`MLPRegressor`) forecasts the inelastic base load for the next 24 hours based on cyclic features and autoregressive lags. During evaluation, the architecture achieved a Mean Squared Error of 0.4307, outperforming naive baselines and preventing the Search engine from pre-charging the battery for false peaks.

### 4. Integration
The engines cooperate through a dynamic feedback loop. If the battery's behavior alters the final grid draw and breaks the physical power limit, the specific appliance schedule is hashed and injected as a negative constraint into the CSP's `forbidden_combos`, forcing the CSP to backtrack and propose the next-best structural layout.

### 5. Limits
The primary limitation is the discretization of the continuous energy domain. To prevent the A* branching factor from exploding, battery actions were restricted to discrete maximum/minimum steps rather than continuous fractional kW modulation.

### 6. Conclusions
Fusing probabilistic neural forecasting with deterministic graph traversals yields a highly resilient architecture. By enforcing structural topology via AC-3 and leaving purely economic decisions to the A* heuristic, the system efficiently navigates competing constraints.
