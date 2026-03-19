import numpy as np
from typing import Dict, Any

class CMDPSolver:
    """
    Constrained Markov Decision Process solver for FleetSafe-VLA
    Validates cost constraints over proposed actions.
    """
    def compute_cost(self, state: Dict[str, Any], action: np.ndarray) -> float:
        """
        Computes the expected CMDP cost of taking the action.
        """
        # Given this is a high level validation solver for the OS,
        # we implement logic resolving cost based on closeness to obstacles.
        cost = 0.0
        
        predicted_pos = state.get('robot_position', np.zeros(3)) + (action[:3] if len(action) >= 3 else 0)
        
        if 'obstacles' in state:
            for obs in state['obstacles']:
                dist = np.linalg.norm(predicted_pos - obs['position'])
                if dist < 1.0:
                    cost += 1.0 / (dist + 0.01) # increase cost drastically as distance closes
                    
        return cost
