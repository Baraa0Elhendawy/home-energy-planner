# Home Energy Planner (Discrete Optimization Pipeline)

A production-grade AI pipeline designed to minimize household electricity costs under a Time-of-Use tariff while strictly enforcing physical grid limits. This project demonstrates a multi-paradigm approach to resource allocation, integrating machine learning forecasting with deterministic graph traversals and constraint programming.

## System Architecture

The pipeline decouples the scheduling problem into three independent inference engines operating in a dynamic feedback loop:

1. **Neural Network Forecaster (`nn.py`):** 
   - A Multi-Layer Perceptron (`MLPRegressor`) predicting the inelastic 24-hour base load.
   - Utilizes temporal features and autoregressive lags to prevent false-peak battery pre-charging.

2. **Constraint Satisfaction Problem (`csp.py`):** 
   - Handles strict topological appliance scheduling. 
   - Implements **Arc Consistency (AC-3)** and **Forward Checking** to proactively prune invalid start times (e.g., precedence rules, shared-circuit limits) before initiating backtracking.

3. **A* Search Engine (`search.py`):** 
   - Evaluates the economic state space `(hour, battery_level)`.
   - Uses a mathematically strictly admissible heuristic to guarantee optimal financial battery routing without expanding unnecessary nodes.

## Installation & Setup (Linux / Fedora)

```bash
# Clone the repository
git clone [https://github.com/Baraa0Elhendawy/home-energy-planner.git](https://github.com/Baraa0Elhendawy/home-energy-planner.git)
cd home-energy-planner

# Set up the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy scikit-learn joblib matplotlib
```

## Execution

1. **Generate Data:** `python make_data.py`
2. **Train Forecaster:** `python nn.py`
3. **Run Integrated Demo:** `python run_demo.py`
4. **Execute Statistical Evaluation:** `python eval.py`

## Experimental Results
Rigorous multi-seed evaluations over 30 simulated days proved:
- **A* Efficiency:** Expanded ~30% fewer state nodes than Uniform Cost Search.
- **CSP Pruning:** AC-3 reduced structural backtracks from thousands to exactly zero under tight power constraints.
- **Financial Impact:** Neural forecasting significantly reduced grid limit violations and total financial cost compared to naive baseline routing.
