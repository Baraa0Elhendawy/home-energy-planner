import json
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ApplianceCSP:
    def __init__(self, config_path="data/config.json", forecast=None):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.appliances = self.config['appliances']
        self.variables = list(self.appliances.keys())
        self.power_limit = self.config['power_limit_kw']
        self.forecast = forecast if forecast is not None else [0.0] * 24
        
        # Domains: lists of valid start hours (0-23). 
        # Truncated so appliances finish before midnight.
        self.domains = {}
        for var, details in self.appliances.items():
            max_start = 24 - details['duration_h']
            self.domains[var] = list(range(max_start + 1))
            
        self.forbidden_combos = []
        self.metrics = {'assignments': 0, 'backtracks': 0}

    def get_load_at_hour(self, hour, assignment):
        """Calculates total load (base forecast + assigned appliances) at a specific hour."""
        load = self.forecast[hour]
        for var, start_hour in assignment.items():
            duration = self.appliances[var]['duration_h']
            if start_hour <= hour < start_hour + duration:
                load += self.appliances[var]['load_kw']
        return load

    def is_consistent(self, var, val, assignment):
        """Checks if assigning var=val violates any constraints against current assignment."""
        details = self.appliances[var]
        
        # 1. Total Load Limit Check
        for h in range(val, val + details['duration_h']):
            if self.get_load_at_hour(h, assignment) + details['load_kw'] > self.power_limit:
                return False

        # 2. Prerequisite Check (e.g., dryer must start AFTER washer finishes)
        prereq = details['prereq']
        if prereq and prereq in assignment:
            if val < assignment[prereq] + self.appliances[prereq]['duration_h']:
                return False
        # Also check reverse: if assigning the prereq, it must finish before the dependent starts
        for assigned_var, assigned_val in assignment.items():
            if self.appliances[assigned_var]['prereq'] == var:
                if assigned_val < val + details['duration_h']:
                    return False

        # 3. Shared Circuit Check (cannot overlap in time)
        shared = details['shared_circuit']
        if shared and shared in assignment:
            shared_start = assignment[shared]
            shared_end = shared_start + self.appliances[shared]['duration_h']
            val_end = val + details['duration_h']
            if max(val, shared_start) < min(val_end, shared_end): # Time overlap condition
                return False

        # 4. Dynamic Negative Constraints (from Search failures)
        temp_assignment = assignment.copy()
        temp_assignment[var] = val
        for combo in self.forbidden_combos:
            if all(temp_assignment.get(k) == v for k, v in combo.items()):
                return False

        return True

    def forward_check(self, unassigned, assignment, current_domains):
        """Prunes domains of unassigned variables based on current assignment."""
        next_domains = {}
        for v in unassigned:
            valid_vals = [val for val in current_domains[v] if self.is_consistent(v, val, assignment)]
            if not valid_vals:
                return None # Domain wipeout
            next_domains[v] = valid_vals
        return next_domains

    def revise(self, xi, xj, current_domains):
        """Removes values from xi's domain that have no valid partner in xj's domain."""
        revised = False
        to_remove = []
        for x in current_domains[xi]:
            satisfies_constraint = False
            for y in current_domains[xj]:
                # Temporarily assign xj=y to check consistency of xi=x
                if self.is_consistent(xi, x, {xj: y}):
                    satisfies_constraint = True
                    break
            if not satisfies_constraint:
                to_remove.append(x)
                revised = True
                
        for x in to_remove:
            current_domains[xi].remove(x)
        return revised

    def ac3(self, current_domains):
        """Arc Consistency 3: Enforces pairwise consistency across all variables."""
        # Initialize queue with all directed arcs (pairs of variables)
        queue = deque()
        for xi in self.variables:
            for xj in self.variables:
                if xi != xj:
                    queue.append((xi, xj))
                    
        while queue:
            xi, xj = queue.popleft()
            if self.revise(xi, xj, current_domains):
                if not current_domains[xi]:
                    return False # Domain wipeout
                for xk in self.variables:
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))
        return True

    def backtrack(self, assignment, use_fc=True, use_ac3=True, current_domains=None):
        """Recursive backtracking solver."""
        if current_domains is None:
            current_domains = {k: list(v) for k, v in self.domains.items()}
            if use_ac3:
                if not self.ac3(current_domains):
                    return None

        if len(assignment) == len(self.variables):
            return assignment

        # Select unassigned variable (Most Restricted Variable heuristic could go here)
        unassigned = [v for v in self.variables if v not in assignment]
        var = unassigned[0]

        for val in current_domains[var]:
            if self.is_consistent(var, val, assignment):
                self.metrics['assignments'] += 1
                assignment[var] = val

                next_domains = {k: list(v) for k, v in current_domains.items()}
                
                if use_fc:
                    fc_domains = self.forward_check([v for v in unassigned if v != var], assignment, next_domains)
                    if fc_domains is None:
                        del assignment[var]
                        self.metrics['backtracks'] += 1
                        continue
                    next_domains.update(fc_domains)

                result = self.backtrack(assignment, use_fc, use_ac3, next_domains)
                if result:
                    return result

                del assignment[var]
                self.metrics['backtracks'] += 1

        return None

if __name__ == "__main__":
    logging.info("Testing CSP Module...")
    # Mock a flat forecast to test basic scheduling
    mock_forecast = [1.0] * 24 
    csp = ApplianceCSP(forecast=mock_forecast)
    
    logging.info("Running BT + FC + AC-3...")
    schedule = csp.backtrack({}, use_fc=True, use_ac3=True)
    
    if schedule:
        logging.info(f"Valid Schedule Found: {schedule}")
        logging.info(f"Metrics: {csp.metrics}")
    else:
        logging.error("No valid schedule found!")