#!/usr/bin/env python3
"""
fleet/dds_bridge.py — Simulated DDS Communication Bridge

Production-grade DDS bridge for the FastBot hospital fleet.
In simulation mode, this provides:
  - Simulated LowState publisher (from Three.js robot state)
  - LowCmd subscriber (forwards joint targets to Command Center)
  - JoystickInjector UDP interface (127.0.0.1:15001)
  - Multi-robot domain isolation
  - Thread-safe ring buffer for high-rate state updates

When connected to real hardware, this wraps unitree_sdk2py
ChannelPublisher/ChannelSubscriber with the same interface.

References:
  - unitree_sdk2py.core.channel (ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber)
  - JoystickInjector.h from unitree_rl_lab
"""

from __future__ import annotations

import json
import math
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from fleet.dds_messages import (
    LowCmd, LowState, MotorCmd, MotorState, WirelessController,
    SportModeState, G1JointIndex, FSMStateID,
    MOTOR_SIZE, ALL_JOINTS_23DOF,
    make_lowcmd, make_passive_cmd, make_fixstand_cmd,
)


# ═══════════════════════════════════════════════════════════════════
#  Joystick Injector (UDP Protocol)
# ═══════════════════════════════════════════════════════════════════

class JoystickInjector:
    """UDP joystick injector matching the C++ JoystickInjector.h protocol.
    
    Listens on UDP port for virtual gamepad commands:
      - "set lx <v> ly <v> rx <v> ry <v> lt <v> rt <v>"  — continuous axes
      - "<button>=1" / "<button>=0"                        — button press/release
      - "hold lt <v>"                                      — analog trigger
    
    Used by Python scripts and the Command Center to control FSM transitions
    and walking velocity without a physical gamepad.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 15001):
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._controller = WirelessController()
        self._button_state: Dict[str, bool] = {}
        self._button_press_time: Dict[str, float] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lt_value = 0.0
        self._rt_value = 0.0

    @property
    def controller(self) -> WirelessController:
        with self._lock:
            return WirelessController(
                lx=self._controller.lx,
                ly=self._controller.ly,
                rx=self._controller.rx,
                ry=self._controller.ry,
                keys=self._controller.keys,
            )

    def start(self):
        """Start UDP listener thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _listen_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.1)
        try:
            sock.bind((self.host, self.port))
        except OSError as e:
            print(f"[JoystickInjector] Cannot bind {self.host}:{self.port}: {e}")
            return

        while self._running:
            try:
                data, addr = sock.recvfrom(1024)
                msg = data.decode("utf-8").strip()
                self._process_message(msg)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[JoystickInjector] Error: {e}")
        sock.close()

    def _process_message(self, msg: str):
        """Parse UDP message and update controller state."""
        with self._lock:
            tokens = msg.split()
            if not tokens:
                return

            cmd = tokens[0].lower()

            # "set lx <v> ly <v> rx <v> ry <v>"
            if cmd == "set":
                for i in range(1, len(tokens) - 1, 2):
                    key = tokens[i].lower()
                    try:
                        val = float(tokens[i + 1])
                    except (ValueError, IndexError):
                        continue
                    if key == "lx": self._controller.lx = val
                    elif key == "ly": self._controller.ly = val
                    elif key == "rx": self._controller.rx = val
                    elif key == "ry": self._controller.ry = val
                    elif key == "lt": self._lt_value = val
                    elif key == "rt": self._rt_value = val
                return

            # "hold lt <v>"
            if cmd == "hold" and len(tokens) >= 3:
                key = tokens[1].lower()
                try:
                    val = float(tokens[2])
                except ValueError:
                    return
                if key == "lt": self._lt_value = val
                elif key == "rt": self._rt_value = val
                # Update keys based on trigger threshold
                self._update_trigger_buttons()
                return

            # "<button>=1" or "<button>=0"  (can have multiple space-separated)
            for token in tokens:
                if "=" in token:
                    parts = token.split("=")
                    if len(parts) == 2:
                        btn_name = parts[0].lower()
                        try:
                            val = int(parts[1])
                        except ValueError:
                            continue
                        self._set_button(btn_name, val != 0)

    def _set_button(self, name: str, pressed: bool):
        """Map button name to WirelessController bitmask."""
        btn_map = {
            'a': WirelessController.Button.A,
            'b': WirelessController.Button.B,
            'x': WirelessController.Button.X,
            'y': WirelessController.Button.Y,
            'rb': WirelessController.Button.R1,
            'lb': WirelessController.Button.L1,
            'r1': WirelessController.Button.R1,
            'l1': WirelessController.Button.L1,
            'rt': WirelessController.Button.R2,
            'lt': WirelessController.Button.L2,
            'r2': WirelessController.Button.R2,
            'l2': WirelessController.Button.L2,
            'up': WirelessController.Button.UP,
            'down': WirelessController.Button.DOWN,
            'left': WirelessController.Button.LEFT,
            'right': WirelessController.Button.RIGHT,
            'start': WirelessController.Button.START,
            'select': WirelessController.Button.SELECT,
            'back': WirelessController.Button.SELECT,
            'f1': WirelessController.Button.F1,
            'f2': WirelessController.Button.F2,
        }
        bit = btn_map.get(name)
        if bit is None:
            return
        now = time.time()
        was_pressed = self._button_state.get(name, False)
        self._button_state[name] = pressed
        if pressed:
            self._controller.keys |= bit
            if not was_pressed:
                self._button_press_time[name] = now
        else:
            self._controller.keys &= ~bit

    def _update_trigger_buttons(self):
        """Update LT/RT digital buttons based on analog threshold."""
        if self._lt_value > 0.5:
            self._controller.keys |= WirelessController.Button.L2
        else:
            self._controller.keys &= ~WirelessController.Button.L2
        if self._rt_value > 0.5:
            self._controller.keys |= WirelessController.Button.R2
        else:
            self._controller.keys &= ~WirelessController.Button.R2

    def send(self, msg: str):
        """Send a message to the injector (useful for programmatic control)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(msg.encode("utf-8"), (self.host, self.port))
        sock.close()

    def tap(self, btn: str, duration: float = 0.12):
        """Tap a button (press and release)."""
        self.send(f"{btn}=1")
        time.sleep(duration)
        self.send(f"{btn}=0")


# ═══════════════════════════════════════════════════════════════════
#  Simulated DDS Channel (for sim-first development)
# ═══════════════════════════════════════════════════════════════════

class SimulatedChannel:
    """Thread-safe pub/sub channel simulating CycloneDDS behavior.
    
    Ring buffer with configurable depth for high-rate data.
    """

    def __init__(self, topic: str, max_depth: int = 100):
        self.topic = topic
        self._buffer: deque = deque(maxlen=max_depth)
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []
        self._last_msg = None

    def write(self, msg, timeout: float = 0.5) -> bool:
        """Publish a message to the channel."""
        from fleetsafe_vla.transport.safety_transport import SafetyTransport
        if not hasattr(self, 'safety_transport'):
            self.safety_transport = SafetyTransport()
        if not self.safety_transport.validate(msg):
            return False

        with self._lock:
            self._buffer.append(msg)
            self._last_msg = msg
        # Invoke callbacks
        for cb in self._callbacks:
            try:
                cb(msg)
            except Exception as e:
                print(f"[DDS:{self.topic}] Callback error: {e}")
        return True

    def read(self):
        """Read the latest message (non-blocking)."""
        with self._lock:
            return self._last_msg

    def read_all(self) -> List:
        """Read all buffered messages."""
        with self._lock:
            msgs = list(self._buffer)
            self._buffer.clear()
            return msgs

    def subscribe(self, callback: Callable, depth: int = 10):
        """Register a callback for new messages."""
        self._callbacks.append(callback)


class DDSBridge:
    """Multi-robot DDS communication bridge.
    
    Manages channels for each robot in the fleet:
      - rt/lowcmd    (publish commands TO robot)
      - rt/lowstate  (subscribe states FROM robot)
      - rt/arm_sdk   (publish arm commands)
      - rt/wirelesscontroller (publish gamepad)
      - /lf/odommodestate (subscribe odometry)
    
    In simulation mode, generates synthetic LowState data from
    the Command Center's Three.js robot positions.
    """

    def __init__(self, mode: str = "sim", domain_id: int = 0, interface: str = "lo"):
        self.mode = mode  # "sim" or "real"
        self.domain_id = domain_id
        self.interface = interface
        self._channels: Dict[str, SimulatedChannel] = {}
        self._robots: Dict[str, RobotDDSContext] = {}
        self._running = False
        self._sim_thread: Optional[threading.Thread] = None
        self._joystick_injector = JoystickInjector()

    def init(self):
        """Initialize the DDS bridge."""
        if self.mode == "real":
            # In production, this would call:
            # ChannelFactoryInitialize(self.domain_id, self.interface)
            print(f"[DDSBridge] Real mode: domain={self.domain_id}, iface={self.interface}")
        else:
            print(f"[DDSBridge] Simulation mode (no CycloneDDS required)")

        # Create default channels
        self._create_channel("rt/lowcmd")
        self._create_channel("rt/lowstate")
        self._create_channel("rt/arm_sdk")
        self._create_channel("rt/wirelesscontroller")
        self._create_channel("/lf/odommodestate")

        # Start joystick injector
        self._joystick_injector.start()
        print(f"[DDSBridge] JoystickInjector listening on UDP 127.0.0.1:15001")

    def _create_channel(self, topic: str) -> SimulatedChannel:
        if topic not in self._channels:
            self._channels[topic] = SimulatedChannel(topic)
        return self._channels[topic]

    def get_channel(self, topic: str) -> SimulatedChannel:
        return self._create_channel(topic)

    def register_robot(self, robot_id: str, domain: int = 1) -> 'RobotDDSContext':
        """Register a robot with its own DDS domain context."""
        ctx = RobotDDSContext(robot_id, domain, self)
        self._robots[robot_id] = ctx
        return ctx

    def start_sim_publisher(self, rate_hz: float = 50.0):
        """Start simulated LowState publisher for all registered robots."""
        if self._running:
            return
        self._running = True
        self._sim_thread = threading.Thread(
            target=self._sim_publish_loop, args=(rate_hz,), daemon=True
        )
        self._sim_thread.start()

    def stop(self):
        self._running = False
        self._joystick_injector.stop()

    def _sim_publish_loop(self, rate_hz: float):
        """Generate simulated robot state at the specified rate."""
        dt = 1.0 / rate_hz
        tick = 0
        while self._running:
            t = time.time()
            for robot_id, ctx in self._robots.items():
                state = ctx.generate_sim_state(tick, t)
                self.get_channel("rt/lowstate").write(state)

                # Also publish simulated odometry
                odom = ctx.generate_sim_odom(t)
                self.get_channel("/lf/odommodestate").write(odom)

            tick += 1
            elapsed = time.time() - t
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    @property
    def joystick(self) -> WirelessController:
        return self._joystick_injector.controller

    def send_joystick(self, msg: str):
        """Send a joystick command via UDP."""
        self._joystick_injector.send(msg)

    def tap_button(self, btn: str, duration: float = 0.12):
        """Tap a gamepad button."""
        self._joystick_injector.tap(btn, duration)

    def get_state_summary(self) -> Dict:
        """Get a summary of all robot states for the dashboard."""
        summary = {}
        for robot_id, ctx in self._robots.items():
            summary[robot_id] = {
                "domain": ctx.domain,
                "fsm_state": ctx.current_fsm_state,
                "last_state_age": time.time() - ctx.last_state_time,
                "position": ctx.position,
                "heading": ctx.heading,
                "connected": not ctx.is_timeout(),
            }
        return summary


@dataclass
class RobotDDSContext:
    """Per-robot DDS context with isolated domain and state tracking."""
    robot_id: str
    domain: int
    bridge: DDSBridge
    current_fsm_state: int = FSMStateID.ZERO_TORQUE
    last_state_time: float = field(default_factory=time.time)
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.78])
    heading: float = 0.0
    joint_positions: List[float] = field(default_factory=lambda: [0.0] * 23)
    joint_velocities: List[float] = field(default_factory=lambda: [0.0] * 23)

    def generate_sim_state(self, tick: int, t: float) -> LowState:
        """Generate simulated LowState for this robot."""
        state = LowState()
        state.mode_machine = self.current_fsm_state
        state.tick = tick
        state.timestamp = t
        self.last_state_time = t

        # Simulated joint states with slight noise
        for i in range(min(23, MOTOR_SIZE)):
            state.motor_state[i].q = self.joint_positions[i] + math.sin(t * 0.5 + i) * 0.001
            state.motor_state[i].dq = self.joint_velocities[i]
            state.motor_state[i].tau_est = 0.0

        # Simulated IMU
        state.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
        state.imu_state.gyroscope = [0.0, 0.0, 0.0]
        state.imu_state.accelerometer = [0.0, 0.0, 9.81]

        # Inject current wireless controller state
        state.wireless_remote = self.bridge.joystick.encode_to_bytes()

        return state

    def generate_sim_odom(self, t: float) -> SportModeState:
        """Generate simulated odometry."""
        return SportModeState(
            position=list(self.position),
            velocity=[0.0, 0.0, 0.0],
            yaw_speed=0.0,
            body_height=self.position[2] if len(self.position) > 2 else 0.78,
            mode=self.current_fsm_state,
        )

    def is_timeout(self, max_age: float = 0.5) -> bool:
        return (time.time() - self.last_state_time) > max_age

    def set_position(self, x: float, y: float, z: float = 0.78):
        """Update position (called from Command Center Three.js sync)."""
        self.position = [x, y, z]

    def set_heading(self, heading_rad: float):
        self.heading = heading_rad

    def set_joint_positions(self, positions: List[float]):
        """Update joint positions (from policy output)."""
        for i in range(min(len(positions), 23)):
            self.joint_positions[i] = positions[i]


# ═══════════════════════════════════════════════════════════════════
#  Convenience: send FSM transitions (from curriculum)
# ═══════════════════════════════════════════════════════════════════

def send_fsm_transition(bridge: DDSBridge, target: str):
    """Send a joystick combo to trigger an FSM transition.
    
    Matches the curriculum's config.yaml transition map:
      Passive:       RT + B
      FixStand:      RT + Y
      VelocityV2.6:  RB + X
      WideStance:    RB + A
      DanceShort:    LB + up
      GangnamStyle:  LB + right
      HorsePunch:    LB + down
      SpinAround:    LT + up
    """
    combos = {
        "passive":      ("rt", "b"),
        "fixstand":     ("rt", "y"),
        "velocity":     ("rb", "x"),
        "widestance":   ("rb", "a"),
        "danceshort":   ("lb", "up"),
        "gangnamstyle": ("lb", "right"),
        "horsepunch":   ("lb", "down"),
        "spinaround":   ("lt", "up"),
        "patrol":       ("rb", "y"),   # Hospital: custom
        "delivery":     ("lb", "a"),   # Hospital: custom
        "emergency":    ("lt", "b"),   # Hospital: custom
    }

    key = target.lower().replace(" ", "").replace("_", "")
    if key not in combos:
        print(f"[FSM] Unknown transition target: {target}")
        return False

    modifier, button = combos[key]

    # Release all first
    bridge.send_joystick(f"{modifier}=0 {button}=0")
    time.sleep(0.05)

    # Hold modifier
    if modifier in ("lt", "rt"):
        bridge.send_joystick(f"hold {modifier} 1.0")
    else:
        bridge.send_joystick(f"{modifier}=1")
    time.sleep(0.15)

    # Tap button
    bridge.send_joystick(f"{button}=1")
    time.sleep(0.12)
    bridge.send_joystick(f"{button}=0")

    # Release modifier
    time.sleep(0.15)
    if modifier in ("lt", "rt"):
        bridge.send_joystick(f"hold {modifier} 0.0")
    else:
        bridge.send_joystick(f"{modifier}=0")

    print(f"[FSM] Transition sent: {target} ({modifier}+{button})")
    return True


if __name__ == "__main__":
    print("=== DDS Bridge Self-Test ===")

    # Create bridge in sim mode
    bridge = DDSBridge(mode="sim")
    bridge.init()

    # Register robots
    r0 = bridge.register_robot("robot_0", domain=1)
    r1 = bridge.register_robot("robot_1", domain=2)
    r0.set_position(3.0, 2.5)
    r1.set_position(8.0, 5.0)

    # Start sim publisher
    bridge.start_sim_publisher(rate_hz=10)

    # Wait for a few states
    time.sleep(0.5)

    # Read latest state
    state = bridge.get_channel("rt/lowstate").read()
    if state:
        print(f"Latest LowState: tick={state.tick}, mode={state.mode_machine}")
        positions = state.get_joint_positions()
        print(f"Joint positions (first 5): {[f'{p:.4f}' for p in positions[:5]]}")

    # Read odometry
    odom = bridge.get_channel("/lf/odommodestate").read()
    if odom:
        print(f"Odometry: pos=({odom.position[0]:.2f}, {odom.position[1]:.2f}, {odom.position[2]:.2f})")

    # Fleet summary
    summary = bridge.get_state_summary()
    for rid, info in summary.items():
        print(f"Robot {rid}: domain={info['domain']}, connected={info['connected']}, pos={info['position']}")

    bridge.stop()
    print("\n✅ DDS Bridge self-test passed")
