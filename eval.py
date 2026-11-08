import os
import json
import time
import logging
import unittest
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nn import LoadForecaster
from csp import ApplianceCSP
from search import BatteryPlanner

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(message)s')

DATA_DIR = "data"

class TestHomeEnergyPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forecast = [1.0] * 24
        with open(os.path.join(DATA_DIR, "config.json"), 'r') as f:
            cls.config = json.load(f)
        cls.prices = cls.config['prices']['hourly']
        cls.limit = cls.config['power_limit_kw']
        cls.appliances = [0.0] * 24

    def test_nn_forecast_shape(self):
        fc = LoadForecaster()
        fc.load_model()
        pred = fc.predict_24h([0, 0, 1, 1.5, 1.4])
        self.assertEqual(pred.shape, (24,))

    def test_csp_precedence(self):
        csp = ApplianceCSP(forecast=self.forecast)
        res = csp.backtrack({}, use_fc=True, use_ac3=True)
        self.assertLess(res['washer'], res['dryer'] + csp.appliances['dryer']['duration_h'])

    def test_csp_shared_circuit(self):
        csp = ApplianceCSP(forecast=self.forecast)
        res = csp.backtrack({}, use_fc=True, use_ac3=True)
        ev_start = res['ev_charger']
        ev_end = ev_start + csp.appliances['ev_charger']['duration_h']
        oven_start = res['oven']
        oven_end = oven_start + csp.appliances['oven']['duration_h']
        self.assertFalse(max(ev_start, oven_start) < min(ev_end, oven_end))

    def test_search_ucs_finds_path(self):
        bp = BatteryPlanner()
        path, cost = bp.plan(self.forecast, self.appliances, self.prices, use_a_star=False)
        self.assertEqual(len(path), 24)

    def test_search_astar_finds_path(self):
        bp = BatteryPlanner()
        path, cost = bp.plan(self.forecast, self.appliances, self.prices, use_a_star=True)
        self.assertEqual(len(path), 24)

    def test_zero_battery_capacity(self):
        bp = BatteryPlanner()
        bp.capacity = 0.0
        path, cost = bp.plan(self.forecast, self.appliances, self.prices, use_a_star=True)
        self.assertTrue(all(p[0] == 0.0 for p in path)) 

    def test_flat_prices(self):
        bp = BatteryPlanner()
        flat_prices = [0.2] * 24
        path, cost = bp.plan(self.forecast, self.appliances, flat_prices, use_a_star=True)
        self.assertIsNotNone(path)

    def test_exact_limit_match(self):
        csp = ApplianceCSP(forecast=[0.0]*24)
        csp.power_limit = 3.5 
        res = csp.backtrack({}, use_fc=True, use_ac3=True)
        self.assertIsNotNone(res)

    def test_impossible_forecast_exceeds_limit(self):
        csp = ApplianceCSP(forecast=[99.0]*24)
        res = csp.backtrack({}, use_fc=True, use_ac3=True)
        self.assertIsNone(res)
        print("\n[Edge Case] Impossible Case 1: no solution - base forecast exceeds grid limit.")

    def test_impossible_appliance_conflict(self):
        csp = ApplianceCSP(forecast=[0.0]*24)
        csp.power_limit = 1.0 
        res = csp.backtrack({}, use_fc=True, use_ac3=True)
        self.assertIsNone(res)
        print("[Edge Case] Impossible Case 2: no solution - appliance load violates limit.")


def run_experiments():
    logging.info("\n=======================================================")
    logging.info("Executing Mathematically Rigorous Multi-Seed Experiments...")
    
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), index_col='Datetime', parse_dates=True)
    with open(os.path.join(DATA_DIR, "config.json"), 'r') as f:
        config = json.load(f)
    prices = config['prices']['hourly']
    
    seeds = [0, 24*10, 24*20, 24*30, 24*40]
    
    # ---------------------------------------------------------
    # Experiment 1: UCS vs A* State Space Complexity
    # ---------------------------------------------------------
    logging.info("Running Exp 1: UCS vs A* (Tracking exact node expansions)...")
    exp1_ucs_nodes, exp1_astar_nodes = [], []
    exp1_ucs_time, exp1_astar_time = [], []
    
    bp = BatteryPlanner()
    for seed in seeds:
        ucs_n, astar_n, ucs_t, astar_t = 0, 0, 0.0, 0.0
        for day in range(6): 
            idx = seed + (day * 24)
            forecast = test_df.iloc[idx:idx+24]['target_1'].values
            
            t0 = time.time()
            path_ucs, _ = bp.plan(forecast, [0]*24, prices, use_a_star=False)
            ucs_t += (time.time() - t0)
            ucs_n += bp.nodes_expanded
            
            t0 = time.time()
            path_astar, _ = bp.plan(forecast, [0]*24, prices, use_a_star=True)
            astar_t += (time.time() - t0)
            astar_n += bp.nodes_expanded
            
        exp1_ucs_nodes.append(ucs_n)
        exp1_astar_nodes.append(astar_n)
        exp1_ucs_time.append(ucs_t)
        exp1_astar_time.append(astar_t)

    # ---------------------------------------------------------
    # Experiment 2: CSP Combinatorial Pruning Effectiveness
    # ---------------------------------------------------------
    logging.info("Running Exp 2: CSP Backtracks with strict constraints...")
    app_counts = [3, 5, 8]
    bt_res, fc_res, ac3_res = [], [], []
    
    for count in app_counts:
        csp = ApplianceCSP()
        csp.power_limit = 2.5 # Strict artificial bottleneck to force heavy graph conflicts
        
        keys = list(csp.appliances.keys())
        mock_apps = {k: csp.appliances[k] for k in keys[:min(count, 5)]}
        for i in range(5, count):
            mock_apps[f'extra_{i}'] = {"duration_h": 1, "load_kw": 1.1, "prereq": None, "shared_circuit": None}
        
        csp.appliances = mock_apps
        csp.variables = list(mock_apps.keys())
        csp.domains = {v: list(range(24 - mock_apps[v]['duration_h'] + 1)) for v in csp.variables}
        
        bt_seed, fc_seed, ac3_seed = [], [], []
        for _ in seeds:
            csp.metrics['backtracks'] = 0
            csp.backtrack({}, use_fc=False, use_ac3=False)
            bt_seed.append(csp.metrics['backtracks'])
            
            csp.metrics['backtracks'] = 0
            csp.backtrack({}, use_fc=True, use_ac3=False)
            fc_seed.append(csp.metrics['backtracks'])
            
            csp.metrics['backtracks'] = 0
            csp.backtrack({}, use_fc=True, use_ac3=True)
            ac3_seed.append(csp.metrics['backtracks'])
            
        bt_res.append((np.mean(bt_seed), np.std(bt_seed, ddof=1)))
        fc_res.append((np.mean(fc_seed), np.std(fc_seed, ddof=1)))
        ac3_res.append((np.mean(ac3_seed), np.std(ac3_seed, ddof=1)))

    # ---------------------------------------------------------
    # Experiment 3: Neural Network vs Yesterday-Baseline Financial Impact
    # ---------------------------------------------------------
    logging.info("Running Exp 3: NN Forecast vs Yesterday-Baseline on Real Loads...")
    nn_costs, baseline_costs = [], []
    nn_violations, baseline_violations = [], []
    
    fc = LoadForecaster()
    fc.load_model()
    
    for seed in seeds:
        n_cost, b_cost = 0.0, 0.0
        n_viol, b_viol = 0, 0
        
        for day in range(6):
            idx = seed + (day * 24)
            if idx + 48 >= len(test_df):
                break
                
            features = test_df.iloc[idx][['hour', 'weekday', 'month', 'prev_24h', 'same_hour_last_week']].values
            real_load = test_df.iloc[idx:idx+24]['target_1'].values
            yesterday_load = test_df.iloc[idx]['prev_24h']
            yesterday_forecast = np.full(24, yesterday_load)
            nn_forecast = fc.predict_24h(features)
            
            # Plan on NN
            path_nn, _ = bp.plan(nn_forecast, [0]*24, prices, use_a_star=True)
            if path_nn:
                for h in range(24):
                    actual_draw = real_load[h] + path_nn[h][0]
                    n_cost += max(0, actual_draw) * prices[h]
                    if actual_draw > config['power_limit_kw']:
                        n_viol += 1
                        
            # Plan on Baseline
            path_base, _ = bp.plan(yesterday_forecast, [0]*24, prices, use_a_star=True)
            if path_base:
                for h in range(24):
                    actual_draw = real_load[h] + path_base[h][0]
                    b_cost += max(0, actual_draw) * prices[h]
                    if actual_draw > config['power_limit_kw']:
                        b_viol += 1
                        
        nn_costs.append(n_cost)
        baseline_costs.append(b_cost)
        nn_violations.append(n_viol)
        baseline_violations.append(b_viol)

    # ---------------------------------------------------------
    # Data Aggregation & Matplotlib Generation
    # ---------------------------------------------------------
    logging.info("Exporting rigorous statistical reports and plots...")
    
    fig, ax1 = plt.subplots(figsize=(8,5))
    labels = [f"Fold {i+1}" for i in range(len(seeds))]
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, exp1_ucs_nodes, 0.4, label='UCS Nodes', color='salmon')
    ax1.bar(x + 0.2, exp1_astar_nodes, 0.4, label='A* Nodes', color='steelblue')
    ax1.set_ylabel('Nodes Expanded')
    ax1.set_title('Exp 1: Search Node Expansion (30 Simulated Days)')
    ax1.legend()
    plt.savefig(os.path.join(DATA_DIR, 'exp1_search.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(8,5))
    x_app = np.arange(len(app_counts))
    ax.bar(x_app - 0.25, [v[0] for v in bt_res], 0.25, yerr=[v[1] for v in bt_res], capsize=5, label='Pure BT')
    ax.bar(x_app, [v[0] for v in fc_res], 0.25, yerr=[v[1] for v in fc_res], capsize=5, label='BT + FC')
    ax.bar(x_app + 0.25, [v[0] for v in ac3_res], 0.25, yerr=[v[1] for v in ac3_res], capsize=5, label='BT + FC + AC-3')
    ax.set_xticks(x_app)
    ax.set_xticklabels([f"{c} Appliances" for c in app_counts])
    ax.set_ylabel('Mean Backtracks (SymLog Scale)')
    ax.set_yscale('symlog', linthresh=1.0) 
    ax.set_title('Exp 2: CSP Pruning Effectiveness (Tight Power Constraints)')
    ax.legend()
    plt.savefig(os.path.join(DATA_DIR, 'exp2_csp.png'))
    plt.close()
    
    fig, ax3 = plt.subplots(figsize=(8,5))
    ax3.bar(x - 0.2, baseline_costs, 0.4, label='Yesterday-Baseline Cost', color='darkgray')
    ax3.bar(x + 0.2, nn_costs, 0.4, label='NN Forecast Cost', color='forestgreen')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel('Simulated Cost ($)')
    ax3.set_title('Exp 3: Financial Impact of Neural Forecasting on Real Loads')
    ax3.legend()
    plt.savefig(os.path.join(DATA_DIR, 'exp3_nn.png'))
    plt.close()

    report_path = os.path.join(DATA_DIR, 'eval_results.txt')
    with open(report_path, 'w') as f:
        f.write("=== EXPERIMENT 1: UCS vs A* (Sample Mean ± Std. Dev, ddof=1) ===\n")
        f.write(f"UCS Nodes: {np.mean(exp1_ucs_nodes):.1f} ± {np.std(exp1_ucs_nodes, ddof=1):.1f}\n")
        f.write(f"A* Nodes:  {np.mean(exp1_astar_nodes):.1f} ± {np.std(exp1_astar_nodes, ddof=1):.1f}\n")
        f.write(f"UCS Time (s): {np.mean(exp1_ucs_time):.3f} ± {np.std(exp1_ucs_time, ddof=1):.3f}\n")
        f.write(f"A* Time (s):  {np.mean(exp1_astar_time):.3f} ± {np.std(exp1_astar_time, ddof=1):.3f}\n\n")
        
        f.write("=== EXPERIMENT 2: CSP BACKTRACKS (Mean ± Std. Dev, ddof=1) ===\n")
        for i, c in enumerate(app_counts):
            f.write(f"Appliances: {c:02d} | BT: {bt_res[i][0]:7.1f}±{bt_res[i][1]:7.1f} | FC: {fc_res[i][0]:5.1f}±{fc_res[i][1]:5.1f} | AC3: {ac3_res[i][0]:5.1f}±{ac3_res[i][1]:5.1f}\n")
            
        f.write("\n=== EXPERIMENT 3: FINANCIAL ROUTING (NN vs BASELINE) ===\n")
        f.write(f"NN Total Cost: ${np.sum(nn_costs):.2f} | Limit Violations: {np.sum(nn_violations)}\n")
        f.write(f"Baseline Cost: ${np.sum(baseline_costs):.2f} | Limit Violations: {np.sum(baseline_violations)}\n")
            
    logging.info(f"Done. Check '{DATA_DIR}' for generated plots and data tables.")

if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)
    run_experiments()