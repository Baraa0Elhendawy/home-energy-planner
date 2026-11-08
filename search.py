import json
import heapq
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BatteryPlanner:
    def __init__(self, config_path="data/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.capacity = self.config['battery']['capacity_kwh']
        self.max_rate = self.config['battery']['max_rate_kw']
        self.efficiency = self.config['battery']['efficiency']
        self.power_limit = self.config['power_limit_kw']
        
        self.nodes_expanded = 0
        self.nodes_generated = 0

    def plan(self, forecast, appliance_load, prices, use_a_star=True):
        self.nodes_expanded = 0
        self.nodes_generated = 0
        
        # State: (hour, rounded_battery_level)
        # We round the battery level to 2 decimal places to prevent state-space explosion from float drift
        start_state = (0, 0.0)
        
        # Priority Queue: (estimated_total_cost, current_cost, path, state)
        # path is a list of tuples: (action_kw_grid_impact, new_batt_level)
        pq = [(0.0, 0.0, [], start_state)]
        visited = {}
        
        cheapest_price = min(prices)
        
        while pq:
            est_total, current_cost, path, state = heapq.heappop(pq)
            hour, batt = state
            self.nodes_expanded += 1
            
            if hour == 24:
                return path, current_cost
                
            if state in visited and visited[state] <= current_cost:
                continue
            visited[state] = current_cost
            
            total_house_load = forecast[hour] + appliance_load[hour]
            actions = set([0.0]) # Always evaluate Idle
            
            # Action: Discharge to cover house load
            if total_house_load > 0:
                drain_needed = total_house_load / self.efficiency
                max_drain_possible = min(self.max_rate, batt)
                if drain_needed <= max_drain_possible:
                    actions.add(-total_house_load)
                else:
                    actions.add(-(max_drain_possible * self.efficiency))
                    
            # Action: Charge at max rate
            max_charge_allowed = min(self.max_rate, (self.capacity - batt) / self.efficiency)
            if max_charge_allowed > 0:
                actions.add(max_charge_allowed)
                
            # Action: Charge exactly up to the grid limit
            headroom = self.power_limit - total_house_load
            if 0 < headroom < max_charge_allowed:
                actions.add(headroom)
            
            for action in actions:
                self.nodes_generated += 1
                
                # Calculate new battery level
                if action > 0: # Charging
                    new_batt = batt + (action * self.efficiency)
                else: # Discharging (action is negative)
                    new_batt = batt - (abs(action) / self.efficiency)
                    
                new_batt = round(new_batt, 2)
                
                # Calculate grid draw
                grid_draw = total_house_load + action
                if grid_draw < 0:
                    grid_draw = 0.0 # Cannot export to grid in this model
                    
                # Strict check against physical grid limit
                if grid_draw > self.power_limit:
                    continue 
                    
                step_cost = grid_draw * prices[hour]
                new_cost = current_cost + step_cost
                
                # Admissible Heuristic
                heuristic = 0.0
                if use_a_star and hour < 23:
                    future_load = sum(forecast[hour+1:]) + sum(appliance_load[hour+1:])
                    remaining_energy_needed = max(0, future_load - (new_batt * self.efficiency))
                    heuristic = remaining_energy_needed * cheapest_price
                    
                next_state = (hour + 1, new_batt)
                heapq.heappush(pq, (new_cost + heuristic, new_cost, path + [(action, new_batt)], next_state))
                
        return None, float('inf')

if __name__ == "__main__":
    logging.info("Testing Search Module on a known small example...")
    # Known example: Forecast is 1.0 kW flat. Prices are cheap at hour 0 ($0.1), expensive at hour 1 ($1.0).
    # Optimal logic: Pre-charge at hour 0 to avoid paying at hour 1.
    bp = BatteryPlanner()
    mock_forecast = [1.0, 1.0] + [0.0]*22
    mock_appliances = [0.0]*24
    mock_prices = [0.1, 1.0] + [0.1]*22
    
    logging.info("Running A* Search...")
    path, cost = bp.plan(mock_forecast, mock_appliances, mock_prices, use_a_star=True)
    
    if path:
        logging.info(f"Path found. Total Cost: ${cost:.3f}")
        logging.info(f"Nodes Expanded: {bp.nodes_expanded}, Nodes Generated: {bp.nodes_generated}")
        logging.info(f"Hour 0 Action: {path[0][0]:.2f} kW grid impact (Battery: {path[0][1]:.2f} kWh)")
        logging.info(f"Hour 1 Action: {path[1][0]:.2f} kW grid impact (Battery: {path[1][1]:.2f} kWh)")
    else:
        logging.error("Failed to find a path!")