"""
SafetyKernel: Formal CBF-QP constraint enforcement for VLA actions.
Implements Semantic Barrier Functions (SBF) guided by Natural Language,
and formal guarantees under bounded state estimation error (Theorem 3).
"""
import time
import numpy as np
import cvxpy as cp
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging

@dataclass
class SafetyConstraint:
    """Represents a Semantic Barrier Function (SBF) mapped from Language"""
    name: str
    constraint_type: str  # "cbf", "stl_formula", "cmdp_cost"
    # For CBF: h(x) >= 0 defines the safe set.
    h_func: Optional[Callable[[np.ndarray], float]] = None
    grad_h_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    alpha: float = 1.0    # Class-K function parameter 
    gamma: float = 1.0    # Lipschitz bound on grad_h
    priority: int = 1
    formula: str = ""
    cost_threshold: float = 0.0

class LLMSemanticParser:
    """Mock parser grounding natural language into continuous SBFs"""
    @staticmethod
    def parse_instruction(instruction: str) -> List[SafetyConstraint]:
        constraints = []
        if "avoid humans" in instruction.lower():
            # Example: Maintain 1.0m distance from humans
            # h(x) = distance(x, human) - 1.0 >= 0
            # For this mock, assume humans are at origin (0,0)
            def h_human(pos: np.ndarray) -> float:
                return np.linalg.norm(pos[:2]) - 1.0
            
            def grad_h_human(pos: np.ndarray) -> np.ndarray:
                n = np.linalg.norm(pos[:2])
                if n < 1e-4: return np.array([1.0, 0.0, 0.0]) # fallback
                return np.array([pos[0]/n, pos[1]/n, 0.0])
            
            constraints.append(SafetyConstraint(
                name="avoid_humans",
                constraint_type="cbf",
                h_func=h_human,
                grad_h_func=grad_h_human,
                alpha=2.0,
                gamma=1.0,
                priority=1
            ))
        return constraints

class SafetyKernel:
    """
    Core safety enforcement module wrapping VLA Foundation Models.
    Projects unsafe actions onto the constraints using an SBF-QP.
    """
    
    def __init__(self, epsilon_error_bound: float = 0.05):
        self.constraints: Dict[str, SafetyConstraint] = {}
        self.violation_history: List[Dict] = []
        self.logger = logging.getLogger(__name__)
        self.epsilon = epsilon_error_bound # Maximum state estimation error (Theorem 3)
        self.parser = LLMSemanticParser()
        
        # Spatial Integration: Drift-Aware geometric firewall
        try:
            from fleetsafe_vla.modules.spatial_registration import DriftAwareRegistration
            self.spatial_guard = DriftAwareRegistration(shift_limit_ratio=0.05)
            # Mock anchor setting for initialization
            self.spatial_guard.set_anchor(np.array([[-5, -5, -5], [5, 5, 5]]))
        except ImportError:
            self.spatial_guard = None
        
    def load_language_constraints(self, instruction: str):
        self.logger.info(f"Parsing language instruction: {instruction}")
        parsed_constraints = self.parser.parse_instruction(instruction)
        for c in parsed_constraints:
            self.register_constraint(c)

    def register_constraint(self, constraint: SafetyConstraint):
        self.constraints[constraint.name] = constraint
        self.logger.info(f"Registered constraint: {constraint.name}")
    
    def validate_action(self, 
                       state: Dict, 
                       action: np.ndarray,
                       robot_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        
        # Geometric Firewall: Validate spatial registration constraints first
        if self.spatial_guard and not self.spatial_guard.process_incoming_frame(state):
            self.logger.error(f"Geometric verification failed for {robot_id} - Auto-Calibration Triggered.")
            return False, "Catastrophic Spatial Drift"

        # Priority 1 (critical) semantic constraints
        for constraint in sorted(self.constraints.values(), key=lambda c: c.priority):
            if not self._check_constraint(state, action, constraint, robot_id):
                self._log_violation(constraint, state, action, robot_id)
                return False, constraint.name
        
        return True, None
    
    def _check_constraint(self, state: Dict, action: np.ndarray, constraint: SafetyConstraint, robot_id: Optional[str]) -> bool:
        if constraint.constraint_type == "cbf":
            if constraint.h_func is None or constraint.grad_h_func is None:
                return True
            pos = state.get('robot_position', np.zeros(3))
            h = constraint.h_func(pos)
            grad_h = constraint.grad_h_func(pos)
            
            # CBF Condition: grad_h^T * u >= -alpha * h + gamma * epsilon
            # The safety margin (gamma * epsilon) ensures robust invariance (Theorem 3)
            robustness_margin = constraint.gamma * self.epsilon
            # We assume action is a velocity control u = (vx, vy, v_theta)
            vel = action[:3]
            if np.dot(grad_h, vel) < -constraint.alpha * h + robustness_margin:
                return False
            return True
        elif constraint.constraint_type == "stl_formula":
            from fleetsafe_vla.kernel.stl_verifier import STLVerifier
            return STLVerifier().evaluate(constraint.formula, state, action)
        elif constraint.constraint_type == "cmdp_cost":
            from fleetsafe_vla.kernel.cmdp_solver import CMDPSolver
            return CMDPSolver().compute_cost(state, action) <= constraint.cost_threshold
        return True
    
    def _log_violation(self, constraint: SafetyConstraint, state: Dict, action: np.ndarray, robot_id: Optional[str]):
        """Log constraint violation for debugging/learning."""
        violation = {
            'timestamp': time.time(),
            'constraint': constraint.name,
            'state': state,
            'action': action.tolist() if isinstance(action, np.ndarray) else action,
            'robot_id': robot_id
        }
        self.violation_history.append(violation)
        self.logger.warning(f"Safety violation: {constraint.name} (robot={robot_id})")

    def project_to_safe_action(self, state: Dict, uvla: np.ndarray) -> np.ndarray:
        """
        Projects an unsafe VLA action onto the nearest safe action space
        using a Control Barrier Function Quadratic Program (CBF-QP).
        
        u* = argmin 1/2 ||u - uvla||^2
             s.t. grad_h^T u >= -alpha * h + gamma * epsilon
        """
        nu = len(uvla)
        u = cp.Variable(nu)
        
        objective = cp.Minimize(0.5 * cp.sum_squares(u - uvla))
        constraints = []
        
        pos = state.get('robot_position', np.zeros(nu))
        
        for c_name, c in self.constraints.items():
            if c.constraint_type == "cbf" and c.h_func and c.grad_h_func:
                h_val = c.h_func(pos)
                grad_h_val = c.grad_h_func(pos)
                robust_margin = c.gamma * self.epsilon
                
                # grad_h^T * u >= -alpha * h + gamma * epsilon
                constraints.append(
                    grad_h_val @ u[:3] >= -c.alpha * h_val + robust_margin
                )
                
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.OSQP, warm_start=True)
            if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
                return u.value
            else:
                self.logger.error("CBF-QP Infeasible! Halting.")
                return np.zeros_like(uvla)
        except Exception as e:
            self.logger.error(f"CBF-QP Solver Failed: {e}")
            return np.zeros_like(uvla)
