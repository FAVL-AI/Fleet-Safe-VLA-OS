#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
 FLEET SAFE VLA - HFB-S | Notebook 07: 7D Cognitive Safety Modeling
═══════════════════════════════════════════════════════════════════════════════
 Extends the MDP with a 7-fleet-safe-vla cognitive safety state space
 using Control Barrier Functions (CBF) for formal safety guarantees.

 State Space: s = (x, y, z, t, F, v, I) ∈ S⁷
   1. Spatial (x,y,z) : COM position relative to support polygon
   2. Temporal (t)     : Deadline-aware scheduling, time-to-preempt (TTP)
   3. Force (F)        : Contact force envelope (0-800N)
   4. Velocity (v)     : Zone-constrained velocity field
   5. Intent (I)       : Goal-conditioned policy with intent prediction

 Safety Guarantee:
   S_safe = {s : h(s) ≥ 0}  (Control Barrier Function)
   CBF: ḣ(s,a) + α·h(s) ≥ 0  ∀a ∈ A

 Usage:
   python notebooks/07_cognitive_7d_modeling.py [--dry-run] [--alpha 1.0]
═══════════════════════════════════════════════════════════════════════════════
"""
import os
import sys
import json
import math
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("NB07_Cognitive7D")

# ═══════════════════════════════════════════════════════════════════
#  7D Cognitive State
# ═══════════════════════════════════════════════════════════════════
@dataclass
class CognitiveState:
    """7-fleet-safe-vla cognitive safety state.
    
    s = (x, y, z, t, F, v, I)
    
    Each dimension has physical meaning and safety bounds.
    """
    # Spatial: COM position (meters)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.5   # Base height
    
    # Temporal: time-to-preempt (seconds)
    t: float = 1.0   # TTP: time budget before preemption
    
    # Force: peak contact force (Newtons)
    F: float = 0.0   # Current max contact force
    
    # Velocity: current speed (m/s)
    v: float = 0.0   # Actual velocity magnitude
    
    # Intent: goal confidence (0-1)
    I: float = 1.0   # Intent certainty / goal progress
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.t, self.F, self.v, self.I])
    
    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'CognitiveState':
        return cls(x=vec[0], y=vec[1], z=vec[2], t=vec[3],
                   F=vec[4], v=vec[5], I=vec[6])


# ═══════════════════════════════════════════════════════════════════
#  Safety Bounds
# ═══════════════════════════════════════════════════════════════════
@dataclass
class SafetyBounds:
    """Physical bounds for each dimension of the 7D state."""
    # Spatial bounds
    x_range: Tuple[float, float] = (-1.0, 16.0)   # Hospital floor
    y_range: Tuple[float, float] = (-1.0, 8.0)
    z_min: float = 0.35                             # Min base height (fall)
    z_max: float = 0.65                             # Max base height
    
    # Temporal
    ttp_min: float = 0.005     # 5ms minimum (E-Stop threshold)
    ttp_safe: float = 0.050    # 50ms preferred
    
    # Force
    f_max: float = 800.0       # Max contact force (N)
    f_critical: float = 500.0  # Warning threshold
    
    # Velocity (zone-dependent)
    v_max_green: float = 0.8   # m/s in green zone
    v_max_amber: float = 0.4   # m/s in amber zone
    v_max_red: float = 0.2     # m/s in red zone
    
    # Intent
    intent_min: float = 0.3    # Below this: intent unclear, reduce speed


# ═══════════════════════════════════════════════════════════════════
#  Control Barrier Functions (CBF)
# ═══════════════════════════════════════════════════════════════════
class ControlBarrierFunction:
    """Control Barrier Function for formal safety verification.
    
    Definition:
      h: S → ℝ  such that  h(s) ≥ 0  ⟹  s ∈ S_safe
      
    CBF Condition (continuous):
      ḣ(s,a) + α·h(s) ≥ 0  ∀a ∈ A
      
    This guarantees forward invariance of the safe set:
      If s(0) ∈ S_safe  ⟹  s(t) ∈ S_safe  ∀t ≥ 0
      
    For the 7D state, we use a composite CBF:
      h(s) = min(h₁(s), h₂(s), ..., h₅(s))
      
    Where each hᵢ corresponds to a safety dimension.
    """
    
    def __init__(self, bounds: SafetyBounds = None, alpha: float = 1.0):
        self.bounds = bounds or SafetyBounds()
        self.alpha = alpha  # CBF class-K function parameter
    
    def h_height(self, s: CognitiveState) -> float:
        """Height barrier: h₁(s) = z - z_min."""
        return s.z - self.bounds.z_min
    
    def h_force(self, s: CognitiveState) -> float:
        """Force barrier: h₂(s) = F_max - F."""
        return self.bounds.f_max - s.F
    
    def h_ttp(self, s: CognitiveState) -> float:
        """Time-to-preempt barrier: h₃(s) = t - t_min."""
        return s.t - self.bounds.ttp_min
    
    def h_velocity(self, s: CognitiveState, zone: str = "green") -> float:
        """Velocity barrier: h₄(s) = v_max(zone) - v."""
        v_max = {
            "green": self.bounds.v_max_green,
            "amber": self.bounds.v_max_amber,
            "red": self.bounds.v_max_red,
        }.get(zone, self.bounds.v_max_green)
        return v_max - s.v
    
    def h_intent(self, s: CognitiveState) -> float:
        """Intent barrier: h₅(s) = I - I_min."""
        return s.I - self.bounds.intent_min
    
    def h_composite(self, s: CognitiveState,
                    zone: str = "green") -> Tuple[float, Dict[str, float]]:
        """Composite CBF: h(s) = min(h₁, ..., h₅).
        
        Returns (min_barrier, all_barriers_dict).
        """
        barriers = {
            "height": self.h_height(s),
            "force": self.h_force(s),
            "ttp": self.h_ttp(s),
            "velocity": self.h_velocity(s, zone),
            "intent": self.h_intent(s),
        }
        return min(barriers.values()), barriers
    
    def is_safe(self, s: CognitiveState, zone: str = "green") -> bool:
        """Check if state is in the safe set."""
        h_min, _ = self.h_composite(s, zone)
        return h_min >= 0
    
    def verify_cbf_condition(self, s: CognitiveState, s_next: CognitiveState,
                             dt: float, zone: str = "green") -> Dict:
        """Verify the CBF condition: ḣ(s,a) + α·h(s) ≥ 0.
        
        Discrete approximation: (h(s') - h(s))/dt + α·h(s) ≥ 0
        """
        h_now, barriers_now = self.h_composite(s, zone)
        h_next, barriers_next = self.h_composite(s_next, zone)
        
        h_dot = (h_next - h_now) / dt
        cbf_value = h_dot + self.alpha * h_now
        
        return {
            "h_now": h_now,
            "h_next": h_next,
            "h_dot": h_dot,
            "cbf_value": cbf_value,
            "cbf_satisfied": cbf_value >= 0,
            "safe_now": h_now >= 0,
            "safe_next": h_next >= 0,
            "barriers": barriers_now,
        }


# ═══════════════════════════════════════════════════════════════════
#  CBF-QP Safety Filter
# ═══════════════════════════════════════════════════════════════════
class CBFSafetyFilter:
    """CBF-based Quadratic Program safety filter.
    
    Finds the closest safe action to the nominal (RL) action:
    
      a* = argmin_a ‖a - a_nom‖²
           s.t. ḣ(s,a) + α·h(s) ≥ 0  (CBF constraint)
                a ∈ [a_min, a_max]     (actuator limits)
    
    Simplified as projection: if CBF violated, project back.
    """
    
    def __init__(self, cbf: ControlBarrierFunction, action_dim: int = 12):
        self.cbf = cbf
        self.action_dim = action_dim
        self.n_filtered = 0
        self.n_total = 0
    
    def filter(self, action: np.ndarray, state: CognitiveState,
               zone: str = "green", dt: float = 0.02) -> Tuple[np.ndarray, Dict]:
        """Apply CBF safety filter to nominal action.
        
        Returns (safe_action, info_dict).
        """
        self.n_total += 1
        
        # Predict next state under nominal action
        s_next = self._predict_next_state(state, action, dt)
        
        # Check CBF condition
        cbf_result = self.cbf.verify_cbf_condition(state, s_next, dt, zone)
        
        if cbf_result["cbf_satisfied"]:
            return action, {"filtered": False, **cbf_result}
        
        # CBF violated — project action into safe set
        self.n_filtered += 1
        safe_action = self._project_to_safe(action, state, zone, dt)
        
        # Verify projected action
        s_safe = self._predict_next_state(state, safe_action, dt)
        cbf_verify = self.cbf.verify_cbf_condition(state, s_safe, dt, zone)
        
        return safe_action, {"filtered": True, **cbf_verify}
    
    def _predict_next_state(self, state: CognitiveState,
                            action: np.ndarray,
                            dt: float) -> CognitiveState:
        """Simple forward model to predict next state."""
        s = state.to_vector().copy()
        
        # Spatial update (simplified kinematics)
        if len(action) >= 6:
            s[0] += action[0] * 0.01 * dt  # x
            s[1] += action[1] * 0.01 * dt  # y
            s[2] += action[2] * 0.001 * dt  # z (height change)
        
        # TTP decreases with action magnitude
        action_magnitude = float(np.linalg.norm(action))
        s[3] = max(0.001, s[3] - action_magnitude * 0.0001)
        
        # Force increases with large actions  
        s[4] = min(800, max(0, s[4] + action_magnitude * 2 - 5))
        
        # Velocity from action
        s[5] = min(1.0, action_magnitude * 0.05)
        
        # Intent stays or decays slightly
        s[6] = max(0, s[6] - 0.001)
        
        return CognitiveState.from_vector(s)
    
    def _project_to_safe(self, action: np.ndarray,
                         state: CognitiveState, zone: str,
                         dt: float) -> np.ndarray:
        """Project action to satisfy CBF constraint.
        
        Binary search on action scaling factor.
        """
        # Scale down action until CBF is satisfied
        for scale in np.linspace(1.0, 0.0, 20):
            scaled_action = action * scale
            s_next = self._predict_next_state(state, scaled_action, dt)
            h_min, _ = self.cbf.h_composite(s_next, zone)
            
            h_dot = (h_min - self.cbf.h_composite(state, zone)[0]) / dt
            cbf_val = h_dot + self.cbf.alpha * self.cbf.h_composite(state, zone)[0]
            
            if cbf_val >= 0:
                return scaled_action
        
        return np.zeros_like(action)  # Emergency: zero action
    
    @property
    def stats(self) -> Dict:
        total = max(self.n_total, 1)
        return {
            "total_actions": self.n_total,
            "filtered": self.n_filtered,
            "filter_rate": self.n_filtered / total,
        }


# ═══════════════════════════════════════════════════════════════════
#  Signal Temporal Logic (STL) Monitor
# ═══════════════════════════════════════════════════════════════════
class STLMonitor:
    """Signal Temporal Logic robustness monitor.
    
    Monitors safety specifications expressed in STL:
    
    φ₁: □[0,T] (z ≥ z_min)           — Always stay above min height
    φ₂: □[0,T] (F ≤ F_max)           — Always keep force below limit
    φ₃: □[0,T] (v ≤ v_zone)          — Always respect zone speed
    φ₄: ◇[0,δ] (t_preempt < t_max)   — Eventually preempt within deadline
    φ₅: □[0,T] (h(s) ≥ 0)            — Always satisfy CBF
    
    Robustness: ρ(φ, s, 0) = min_t∈[0,T] h(s_t)
    Positive ρ ⟹ specification satisfied with margin.
    """
    
    def __init__(self):
        self.specs = {
            "always_upright": [],    # φ₁
            "force_bounded": [],     # φ₂
            "speed_compliant": [],   # φ₃
            "preempt_timely": [],    # φ₄
            "cbf_invariant": [],     # φ₅
        }
        self.timestamps = []
    
    def update(self, state: CognitiveState, cbf_h: float,
               zone_speed_limit: float, t: float):
        """Update all STL monitors with new state."""
        self.timestamps.append(t)
        
        # φ₁: z ≥ 0.35
        self.specs["always_upright"].append(state.z - 0.35)
        
        # φ₂: F ≤ 800
        self.specs["force_bounded"].append(800.0 - state.F)
        
        # φ₃: v ≤ v_zone
        self.specs["speed_compliant"].append(zone_speed_limit - state.v)
        
        # φ₄: TTP > 5ms
        self.specs["preempt_timely"].append(state.t - 0.005)
        
        # φ₅: h(s) ≥ 0
        self.specs["cbf_invariant"].append(cbf_h)
    
    def robustness(self) -> Dict[str, float]:
        """Compute STL robustness for each specification.
        
        ρ(□φ) = min_t ρ(φ, t)  — robustness of "always"
        """
        return {
            name: float(min(vals)) if vals else float("inf")
            for name, vals in self.specs.items()
        }
    
    def all_satisfied(self) -> bool:
        """Check if all specifications have positive robustness."""
        rob = self.robustness()
        return all(v >= 0 for v in rob.values())


# ═══════════════════════════════════════════════════════════════════
#  7D Cognitive Training
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Cognitive7DConfig:
    """Training configuration for 7D cognitive safety model."""
    num_episodes: int = 1000
    max_steps_per_episode: int = 200
    dt: float = 0.02              # 50 Hz
    alpha: float = 1.0            # CBF class-K parameter
    
    # Domain randomization
    com_noise_std: float = 0.005
    force_noise_std: float = 10.0
    latency_noise_std: float = 0.002
    
    auto_shutdown: bool = True


class Cognitive7DTrainer:
    """Trains and validates the 7D cognitive safety model."""
    
    def __init__(self, config: Cognitive7DConfig = None):
        self.cfg = config or Cognitive7DConfig()
        self.bounds = SafetyBounds()
        self.cbf = ControlBarrierFunction(self.bounds, alpha=self.cfg.alpha)
        self.safety_filter = CBFSafetyFilter(self.cbf, action_dim=12)
        self.stl = STLMonitor()
        
        self.metrics = {
            "episode": [], "total_reward": [],
            "cbf_violations": [], "stl_robustness": [],
            "filter_rate": [], "safe_episodes": [],
        }
    
    def _run_episode(self, episode_id: int) -> Dict:
        """Run a single episode with CBF safety filter."""
        np.random.seed(episode_id)
        
        # Initial state
        state = CognitiveState(
            x=np.random.uniform(1, 15),
            y=np.random.uniform(1, 7),
            z=np.random.uniform(0.45, 0.55),
            t=np.random.uniform(0.05, 0.2),
            F=np.random.uniform(0, 100),
            v=0.0,
            I=np.random.uniform(0.7, 1.0),
        )
        
        stl = STLMonitor()
        total_reward = 0
        cbf_violations = 0
        
        for step in range(self.cfg.max_steps_per_episode):
            t = step * self.cfg.dt
            
            # Nominal action (simulated RL policy)
            nominal_action = np.random.randn(12) * 0.3
            
            # Determine zone
            zone = "green"
            if state.x > 12 and state.y > 3.5:
                zone = "red"
            elif state.x > 9:
                zone = "amber"
            
            # CBF safety filter
            safe_action, info = self.safety_filter.filter(
                nominal_action, state, zone, self.cfg.dt)
            
            if not info.get("cbf_satisfied", True):
                cbf_violations += 1
            
            # Apply action and get next state
            state = self.safety_filter._predict_next_state(
                state, safe_action, self.cfg.dt)
            
            # Add noise (domain randomization)
            state.z += np.random.normal(0, self.cfg.com_noise_std)
            state.F = max(0, state.F + np.random.normal(0, self.cfg.force_noise_std))
            state.t = max(0.001, state.t + np.random.normal(0, self.cfg.latency_noise_std))
            
            # STL monitoring
            h_min, _ = self.cbf.h_composite(state, zone)
            zone_limit = {"green": 0.8, "amber": 0.4, "red": 0.2}[zone]
            stl.update(state, h_min, zone_limit, t)
            
            # Reward
            reward = h_min * 2.0 + (1.0 if self.cbf.is_safe(state, zone) else -5.0)
            total_reward += reward
            
            # Early termination on severe violation
            if state.z < 0.3 or state.F > 1000:
                break
        
        robustness = stl.robustness()
        
        return {
            "total_reward": total_reward,
            "cbf_violations": cbf_violations,
            "stl_robustness": robustness,
            "all_stl_satisfied": stl.all_satisfied(),
            "steps": step + 1,
            "final_state": asdict(state),
        }
    
    def train(self, dry_run: bool = False):
        """Run cognitive 7D training/validation."""
        n_episodes = 20 if dry_run else self.cfg.num_episodes
        
        print("=" * 72)
        print("  FLEET SAFE VLA - HFB-S | 7D Cognitive Safety Modeling")
        print("=" * 72)
        print(f"  State: s = (x, y, z, t, F, v, I) ∈ S⁷")
        print(f"  CBF α : {self.cfg.alpha}")
        print(f"  Episodes: {n_episodes}")
        print()
        
        safe_count = 0
        start = time.time()
        
        for ep in range(1, n_episodes + 1):
            result = self._run_episode(ep)
            
            self.metrics["episode"].append(ep)
            self.metrics["total_reward"].append(result["total_reward"])
            self.metrics["cbf_violations"].append(result["cbf_violations"])
            
            # STL robustness (min across all specs)
            rob_vals = list(result["stl_robustness"].values())
            min_rob = min(rob_vals) if rob_vals else 0
            self.metrics["stl_robustness"].append(min_rob)
            self.metrics["safe_episodes"].append(int(result["all_stl_satisfied"]))
            
            if result["all_stl_satisfied"]:
                safe_count += 1
            
            if ep % max(n_episodes // 10, 1) == 0:
                safety_rate = safe_count / ep
                print(f"  Episode {ep:5d} | R={result['total_reward']:7.1f} | "
                      f"CBF_viol={result['cbf_violations']:2d} | "
                      f"STL_ρ={min_rob:.3f} | "
                      f"Safe={safety_rate:.1%}")
        
        filter_stats = self.safety_filter.stats
        elapsed = time.time() - start
        
        results_dir = PROJECT_ROOT / "training_logs" / "07_cognitive_7d"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "metrics.json").write_text(json.dumps(self.metrics, indent=2))
        
        print(f"\n  ✅ 7D Cognitive Modeling complete!")
        print(f"  Safe episodes  : {safe_count}/{n_episodes} ({safe_count/n_episodes:.1%})")
        print(f"  CBF filter rate: {filter_stats['filter_rate']:.1%}")
        print(f"  Mean STL ρ     : {np.mean(self.metrics['stl_robustness']):.3f}")
        print(f"  Time           : {elapsed:.1f}s")
        print("=" * 72)
        
        if self.cfg.auto_shutdown and not dry_run:
            self._auto_shutdown()
    
    def _auto_shutdown(self):
        print("\n  🔄 Auto-shutdown: stopping GCP instance...")
        try:
            import subprocess
            subprocess.run(
                ["gcloud", "compute", "instances", "stop", "isaac-l4-dev",
                 "--zone=us-central1-a", "--quiet"], timeout=60, check=False)
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="7D Cognitive Safety Modeling")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Cognitive7DConfig(alpha=args.alpha)
    if args.episodes:
        cfg.num_episodes = args.episodes
    
    trainer = Cognitive7DTrainer(config=cfg)
    trainer.train(dry_run=args.dry_run)
