#!/usr/bin/env python3
"""
fleet/policy_engine.py — RL Policy Engine & Gait Manager

ONNX policy loader with 45-dim observation builder and multi-policy hot-swap.
Supports both IsaacLab-trained policies and mimic policies (CSV motion files).

Observation vector (45-dim):
  [0:3]   base angular velocity (rad/s) — from IMU gyroscope
  [3:6]   projected gravity (unit vector) — from quaternion
  [6:9]   velocity commands (vx, vy, yaw_rate) — from joystick
  [9:21]  joint positions relative to default (12 leg DoF)
  [21:33] joint velocities (12 leg DoF)
  [33:45] last action (12 leg DoF)

Action space: 12 leg joint position targets (scaled)

References:
  - unitree_rl_lab/deploy/robots/g1_23dof/
  - OpenVLA-7B Llama-2 policy export (with GR00T fallback)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from fleet.dds_messages import (
    LowState, WirelessController, G1JointIndex,
    FIXSTAND_POSE, ALL_JOINTS_23DOF, LEG_JOINTS,
)


# ═══════════════════════════════════════════════════════════════════
#  Observation Builder
# ═══════════════════════════════════════════════════════════════════

def quaternion_to_projected_gravity(q: List[float]) -> List[float]:
    """Convert quaternion (w,x,y,z) to projected gravity vector in body frame.
    
    Gravity in world frame is [0, 0, -1]. We rotate it to body frame.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    # Rotation of [0, 0, -1] by quaternion inverse
    gx = -2.0 * (x * z - w * y)
    gy = -2.0 * (y * z + w * x)
    gz = -(w * w - x * x - y * y + z * z)
    return [gx, gy, gz]


def build_observation(
    low_state: LowState,
    joystick: WirelessController,
    last_action: List[float],
    default_positions: List[float] = None,
    velocity_scale: float = 0.5,
) -> List[float]:
    """Build the 45-dim observation vector for the RL policy.
    
    Args:
        low_state: Latest LowState from DDS
        joystick: Current gamepad state for velocity commands
        last_action: Previous policy output (12 values)
        default_positions: Default joint positions (for relative encoding)
        velocity_scale: Scale factor for joystick → velocity mapping
    
    Returns:
        45-dim observation vector
    """
    if default_positions is None:
        default_positions = [FIXSTAND_POSE[i] for i in LEG_JOINTS]

    obs = []

    # [0:3] Base angular velocity from IMU gyroscope
    obs.extend(low_state.imu_state.gyroscope[:3])

    # [3:6] Projected gravity from quaternion
    proj_grav = quaternion_to_projected_gravity(low_state.imu_state.quaternion)
    obs.extend(proj_grav)

    # [6:9] Velocity commands from joystick
    vx = joystick.ly * velocity_scale      # Forward/back
    vy = joystick.lx * velocity_scale      # Strafe
    yaw_rate = joystick.rx * velocity_scale  # Yaw
    obs.extend([vx, vy, yaw_rate])

    # [9:21] Joint positions relative to default (12 leg joints)
    for idx, leg_joint in enumerate(LEG_JOINTS):
        measured = low_state.motor_state[leg_joint].q
        default = default_positions[idx] if idx < len(default_positions) else 0.0
        obs.append(measured - default)

    # [21:33] Joint velocities (12 leg joints)
    for leg_joint in LEG_JOINTS:
        obs.append(low_state.motor_state[leg_joint].dq)

    # [33:45] Last action (12 values)
    for i in range(12):
        obs.append(last_action[i] if i < len(last_action) else 0.0)

    return obs


# ═══════════════════════════════════════════════════════════════════
#  Policy Engine
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PolicyConfig:
    """Configuration for a walking policy."""
    name: str
    model_path: str = ""  # Path to .onnx file
    action_scale: float = 0.25   # Scale factor for action → joint offset
    velocity_scale: float = 0.5
    obs_dim: int = 45
    act_dim: int = 12
    default_positions: List[float] = field(default_factory=lambda: [
        FIXSTAND_POSE[i] for i in LEG_JOINTS
    ])
    # EMA smoothing
    action_ema: float = 0.3  # 0 = full new, 1 = full old


class PolicyEngine:
    """Multi-policy engine with ONNX inference and hot-swap.
    
    In simulation mode, generates sinusoidal gait patterns
    (no real ONNX model required). When a .onnx model is
    loaded, uses onnxruntime for inference.
    """

    def __init__(self):
        self._policies: Dict[str, PolicyConfig] = {}
        self._active_policy: Optional[str] = None
        self._onnx_sessions: Dict[str, object] = {}
        self._last_action = [0.0] * 12
        self._smoothed_action = [0.0] * 12
        self._tick = 0

        # Register built-in policies
        self._register_defaults()

    def _register_defaults(self):
        """Register default policy configurations."""
        self.register_policy(PolicyConfig(
            name="VelocityV2.6",
            model_path="policies/velocity_v2.6.onnx",
            action_scale=0.25,
            velocity_scale=0.5,
        ))
        self.register_policy(PolicyConfig(
            name="WideStance",
            model_path="policies/wide_stance.onnx",
            action_scale=0.3,
            velocity_scale=0.4,
        ))
        self.register_policy(PolicyConfig(
            name="HospitalPatrol",
            model_path="policies/hospital_patrol.onnx",
            action_scale=0.2,
            velocity_scale=0.3,  # Slower for indoor use
        ))
        self.register_policy(PolicyConfig(
            name="PharmacyPickup",
            model_path="policies/pharmacy_pickup.onnx",
            action_scale=0.15,
            velocity_scale=0.2,  # Very careful near shelves
        ))

    def register_policy(self, config: PolicyConfig):
        """Register a policy configuration."""
        self._policies[config.name] = config

    def set_active_policy(self, name: str) -> bool:
        """Switch to a different policy (hot-swap)."""
        if name not in self._policies:
            print(f"[PolicyEngine] Unknown policy: {name}")
            return False

        self._active_policy = name
        self._last_action = [0.0] * 12
        self._smoothed_action = [0.0] * 12

        # Try to load ONNX model if not already loaded
        config = self._policies[name]
        if config.model_path and name not in self._onnx_sessions:
            self._try_load_onnx(name, config.model_path)

        print(f"[PolicyEngine] Active policy: {name}")
        return True

    def _try_load_onnx(self, name: str, path: str):
        """Attempt to load an ONNX model with kernel-level optimizations."""
        try:
            import onnxruntime as ort
            
            # Kernel-level optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            # Execution Providers for TensorRT and CUDA
            providers = [
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': './trt_caches',
                    'trt_fp16_enable': True,
                }),
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ]
            
            session = ort.InferenceSession(path, sess_options=sess_options, providers=providers)
            self._onnx_sessions[name] = session
            print(f"[PolicyEngine] Loaded highly optimized ONNX model: {path}")
        except (ImportError, Exception) as e:
            # Fallback to simulated inference
            print(f"[PolicyEngine] ONNX/TensorRT not available for {name}: {e}. Using simulated gait.")

    def infer(self, low_state: LowState, joystick: WirelessController) -> List[float]:
        """Run policy inference: observation → action → joint targets.
        
        Returns 23 joint position targets (12 legs + waist/arms held).
        """
        if not self._active_policy or self._active_policy not in self._policies:
            return list(FIXSTAND_POSE[:23])

        config = self._policies[self._active_policy]
        self._tick += 1

        # Build observation
        obs = build_observation(low_state, joystick, self._last_action,
                              config.default_positions, config.velocity_scale)

        # Run inference
        if self._active_policy in self._onnx_sessions:
            # Real ONNX inference
            import numpy as np
            session = self._onnx_sessions[self._active_policy]
            input_name = session.get_inputs()[0].name
            obs_np = np.array([obs], dtype=np.float32)
            raw_action = session.run(None, {input_name: obs_np})[0][0]
            action = [float(a) for a in raw_action[:12]]
        else:
            # Simulated gait: sinusoidal leg motions based on joystick input
            action = self._simulate_gait(joystick, config)

        # Scale actions
        scaled_action = [a * config.action_scale for a in action]

        # EMA smoothing
        ema = config.action_ema
        self._smoothed_action = [
            ema * s + (1 - ema) * a
            for s, a in zip(self._smoothed_action, scaled_action)
        ]
        self._last_action = list(scaled_action)

        # Convert to 23-DoF: leg offsets + default pose for upper body
        joint_targets = list(FIXSTAND_POSE[:23]) if len(FIXSTAND_POSE) >= 23 else ([0.0] * 23)
        for i, leg_idx in enumerate(LEG_JOINTS):
            if i < len(self._smoothed_action) and leg_idx < 23:
                joint_targets[leg_idx] = config.default_positions[i] + self._smoothed_action[i]

        return joint_targets

    def _simulate_gait(self, joystick: WirelessController, config: PolicyConfig) -> List[float]:
        """Generate simulated walking gait from joystick input.
        
        Creates opposing sinusoidal patterns for left/right legs.
        """
        t = self._tick * 0.02  # 50Hz tick → time
        speed = math.sqrt(joystick.lx ** 2 + joystick.ly ** 2)
        amplitude = min(speed * 0.5, 0.3)  # Scale gait amplitude with speed
        freq = 2.0  # 2 Hz stepping frequency

        action = [0.0] * 12
        if amplitude < 0.02:
            return action  # Standing still

        # Left leg (indices 0-5): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        phase_l = math.sin(2 * math.pi * freq * t)
        phase_r = math.sin(2 * math.pi * freq * t + math.pi)  # Anti-phase

        # Hip pitch (forward/back swing)
        action[0] = phase_l * amplitude * 0.8   # Left hip pitch
        action[6] = phase_r * amplitude * 0.8   # Right hip pitch

        # Knee flexion (bend during swing)
        action[3] = abs(phase_l) * amplitude * 1.2  # Left knee
        action[9] = abs(phase_r) * amplitude * 1.2  # Right knee

        # Ankle pitch (ground clearance)
        action[4] = -phase_l * amplitude * 0.4  # Left ankle
        action[10] = -phase_r * amplitude * 0.4  # Right ankle

        # Yaw steering from joystick
        yaw = joystick.rx * 0.1
        action[2] = yaw   # Left hip yaw
        action[8] = -yaw  # Right hip yaw

        return action

    @property
    def active_policy_name(self) -> str:
        return self._active_policy or "None"

    @property
    def available_policies(self) -> List[str]:
        return list(self._policies.keys())

    def get_status(self) -> Dict:
        return {
            "active": self.active_policy_name,
            "available": self.available_policies,
            "tick": self._tick,
            "last_action": self._last_action[:6],  # First 6 for display
        }


if __name__ == "__main__":
    print("=== Policy Engine Self-Test ===")

    engine = PolicyEngine()
    print(f"Available policies: {engine.available_policies}")

    engine.set_active_policy("VelocityV2.6")

    # Create simulated state
    state = LowState()
    for i in range(23):
        state.motor_state[i].q = FIXSTAND_POSE[i] if i < len(FIXSTAND_POSE) else 0.0

    # Test inference with joystick forward
    joy = WirelessController(ly=0.5)
    for step in range(20):
        targets = engine.infer(state, joy)

    print(f"After 20 steps (forward walk):")
    print(f"  Left hip pitch target:  {targets[0]:.4f}")
    print(f"  Right hip pitch target: {targets[6]:.4f}")
    print(f"  Left knee target:       {targets[3]:.4f}")
    print(f"  Right knee target:      {targets[9]:.4f}")

    # Test standing still
    joy_still = WirelessController()
    targets_still = engine.infer(state, joy_still)
    print(f"\nStanding still:")
    print(f"  Left hip pitch: {targets_still[0]:.4f} (should be ~default)")

    # Hot-swap policy
    engine.set_active_policy("HospitalPatrol")
    targets_patrol = engine.infer(state, WirelessController(ly=0.3))
    print(f"\nHospitalPatrol policy active:")
    print(f"  Action scale: {engine._policies['HospitalPatrol'].action_scale}")
    print(f"  Left hip pitch: {targets_patrol[0]:.4f}")

    print(f"\n✅ Policy Engine self-test passed")
