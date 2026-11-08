import os
import json
import logging
import pandas as pd
import numpy as np
from nn import LoadForecaster
from csp import ApplianceCSP
from search import BatteryPlanner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_integration_demo():
    logging.info("--- Starting Integrated Home Energy Planner Demo ---")
    
    # 1. Load Data and Model
    data_dir = "data"
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError("test.csv not found. Run make_data.py first.")
        
    test_df = pd.read_csv(test_path, index_col='Datetime', parse_dates=True)
    
    forecaster = LoadForecaster()
    try:
        forecaster.load_model()
    except FileNotFoundError:
        logging.info("Trained model not found. Training now...")
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), index_col='Datetime', parse_dates=True)
        forecaster.train(train_df)
        forecaster.save_model()
        
    with open(os.path.join(data_dir, "config.json"), 'r') as f:
        config = json.load(f)
        
    prices = config['prices']['hourly']
    power_limit = config['power_limit_kw']
    
    # 2. Pick a sample day from the test set (e.g., index 0)
    sample_row = test_df.iloc[0]
    features = [sample_row['hour'], sample_row['weekday'], sample_row['month'], sample_row['prev_24h'], sample_row['same_hour_last_week']]
    
    logging.info(f"Selected Timestamp: {test_df.index[0]}")
    
    # 3. Neural Network Predicts Next 24 Hours Load
    nn_forecast = forecaster.predict_24h(features)
    logging.info(f"NN Forecast Peak Load: {max(nn_forecast):.2f} kW")
    
    # Initialize Search and CSP
    search_eng = BatteryPlanner()
    max_loops = 5
    loop_count = 0
    final_plan = None
    
    # 4. The Feedback Loop: Search failure -> New CSP Constraint
    while loop_count < max_loops:
        loop_count += 1
        logging.info(f"--- CSP-Search Feedback Loop Iteration {loop_count} ---")
        
        csp = ApplianceCSP(forecast=nn_forecast)
        schedule = csp.backtrack({}, use_fc=True, use_ac3=True)
        
        if not schedule:
            logging.warning("CSP failed to find a valid appliance schedule under current constraints.")
            break
            
        logging.info(f"CSP Proposed Appliance Schedule: {schedule}")
        
        # Build hourly appliance load vector
        appliance_load = np.zeros(24)
        for app, start_hour in schedule.items():
            duration = csp.appliances[app]['duration_h']
            load = csp.appliances[app]['load_kw']
            for h in range(start_hour, start_hour + duration):
                appliance_load[h] += load
                
        # Run A* Battery Search
        path, cost = search_eng.plan(nn_forecast, appliance_load, prices, use_a_star=True)
        
        if not path:
            logging.warning("A* Battery Search failed to find a valid plan.")
            break
            
        # Verify physical grid limits across all 24 hours
        violation_found = False
        for h in range(24):
            grid_action = path[h][0] # grid impact of battery action
            total_draw = nn_forecast[h] + appliance_load[h] + grid_action
            if total_draw > power_limit:
                logging.warning(f"Constraint Violation at Hour {h}: Total draw {total_draw:.2f} kW exceeds limit {power_limit} kW.")
                # Inject negative constraint into CSP
                csp.forbidden_combos.append(schedule)
                violation_found = True
                break
                
        if not violation_found:
            final_plan = {
                "schedule": schedule,
                "battery_path": path,
                "total_cost": cost,
                "nodes_expanded": search_eng.nodes_expanded
            }
            logging.info("Successfully established a conflict-free integrated plan!")
            break
            
    if final_plan:
        print("\n================ FINAL OPTIMIZED PLAN ================")
        print(f"Total Simulated Cost: ${final_plan['total_cost']:.2f}")
        print(f"Search Nodes Expanded: {final_plan['nodes_expanded']}")
        print("Appliance Timetable:")
        for app, hr in final_plan['schedule'].items():
            print(f"  - {app}: starts at {hr:02d}:00")
        print("=======================================================")
    else:
        print("Integration failed to resolve a valid plan within max loops.")

if __name__ == "__main__":
    run_integration_demo()