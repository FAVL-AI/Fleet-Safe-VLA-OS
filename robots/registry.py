#!/usr/bin/env python3
"""
robots/registry.py — URDF Robot Model Registry

Parses URDF XML files and exposes a queryable registry of robot
models with their kinematic chains, inertial properties, and sensors.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

ROBOTS_DIR = Path(__file__).parent


# ═══════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class URDFLink:
    name: str
    mass: float = 0.0
    visual_geometry: str = ""
    has_collision: bool = False
    has_inertial: bool = False


@dataclass
class URDFJoint:
    name: str
    joint_type: str       # revolute, continuous, prismatic, fixed
    parent: str
    child: str
    axis: List[float] = field(default_factory=lambda: [0, 0, 1])
    lower: float = 0.0
    upper: float = 0.0
    effort: float = 0.0
    velocity: float = 0.0
    origin_xyz: List[float] = field(default_factory=lambda: [0, 0, 0])
    origin_rpy: List[float] = field(default_factory=lambda: [0, 0, 0])


@dataclass
class RobotModel:
    id: str
    name: str
    display_name: str
    urdf_path: str
    embodiment: str           # humanoid, quadruped, diff_drive, manipulator
    total_dof: int = 0
    actuated_dof: int = 0
    total_mass_kg: float = 0.0
    total_links: int = 0
    total_joints: int = 0
    links: List[URDFLink] = field(default_factory=list)
    joints: List[URDFJoint] = field(default_factory=list)
    sensors: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # Summarize for API response
        d["joint_names"] = [j["name"] for j in d["joints"] if j["joint_type"] != "fixed"]
        d["sensor_names"] = d["sensors"]
        return d

    def to_summary(self) -> dict:
        """Compact summary without full link/joint lists."""
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "embodiment": self.embodiment,
            "total_dof": self.total_dof,
            "actuated_dof": self.actuated_dof,
            "total_mass_kg": round(self.total_mass_kg, 2),
            "total_links": self.total_links,
            "total_joints": self.total_joints,
            "sensors": self.sensors,
            "urdf_path": self.urdf_path,
            "description": self.description,
        }


# ═══════════════════════════════════════════════════════════════════
#  URDF Parser
# ═══════════════════════════════════════════════════════════════════

def parse_urdf(urdf_path: Path) -> RobotModel:
    """Parse a URDF XML file into a RobotModel."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    robot_name = root.attrib.get("name", urdf_path.stem)

    # Parse links
    links = []
    total_mass = 0.0
    for link_el in root.findall("link"):
        name = link_el.attrib.get("name", "")
        mass = 0.0
        inertial = link_el.find("inertial")
        if inertial is not None:
            mass_el = inertial.find("mass")
            if mass_el is not None:
                mass = float(mass_el.attrib.get("value", 0))
        total_mass += mass

        visual = link_el.find("visual")
        geom_type = ""
        if visual is not None:
            geom = visual.find("geometry")
            if geom is not None and len(geom) > 0:
                geom_type = geom[0].tag

        links.append(URDFLink(
            name=name,
            mass=mass,
            visual_geometry=geom_type,
            has_collision=link_el.find("collision") is not None,
            has_inertial=inertial is not None,
        ))

    # Parse joints
    joints = []
    actuated_dof = 0
    sensors = []

    for joint_el in root.findall("joint"):
        name = joint_el.attrib.get("name", "")
        jtype = joint_el.attrib.get("type", "fixed")

        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        parent = parent_el.attrib.get("link", "") if parent_el is not None else ""
        child = child_el.attrib.get("link", "") if child_el is not None else ""

        axis = [0, 0, 1]
        axis_el = joint_el.find("axis")
        if axis_el is not None:
            axis = [float(x) for x in axis_el.attrib.get("xyz", "0 0 1").split()]

        origin_xyz = [0, 0, 0]
        origin_rpy = [0, 0, 0]
        origin_el = joint_el.find("origin")
        if origin_el is not None:
            xyz_str = origin_el.attrib.get("xyz", "0 0 0")
            origin_xyz = [float(x) for x in xyz_str.split()]
            rpy_str = origin_el.attrib.get("rpy", "0 0 0")
            origin_rpy = [float(x) for x in rpy_str.split()]

        lower, upper, effort, velocity = 0.0, 0.0, 0.0, 0.0
        limit_el = joint_el.find("limit")
        if limit_el is not None:
            lower = float(limit_el.attrib.get("lower", 0))
            upper = float(limit_el.attrib.get("upper", 0))
            effort = float(limit_el.attrib.get("effort", 0))
            velocity = float(limit_el.attrib.get("velocity", 0))

        joints.append(URDFJoint(
            name=name, joint_type=jtype,
            parent=parent, child=child,
            axis=axis, lower=lower, upper=upper,
            effort=effort, velocity=velocity,
            origin_xyz=origin_xyz, origin_rpy=origin_rpy,
        ))

        if jtype in ("revolute", "continuous", "prismatic"):
            actuated_dof += 1

        # Detect sensors from link names
        if "imu" in child.lower() or "camera" in child.lower() or "lidar" in child.lower():
            sensor_type = "imu" if "imu" in child.lower() else (
                "camera" if "camera" in child.lower() else "lidar"
            )
            if sensor_type not in sensors:
                sensors.append(sensor_type)

    return RobotModel(
        id=robot_name.lower().replace(" ", "_"),
        name=robot_name,
        display_name=robot_name.replace("_", " ").title(),
        urdf_path=str(urdf_path.relative_to(ROBOTS_DIR.parent)),
        embodiment="unknown",
        total_dof=actuated_dof,
        actuated_dof=actuated_dof,
        total_mass_kg=total_mass,
        total_links=len(links),
        total_joints=len(joints),
        links=links,
        joints=joints,
        sensors=sensors,
    )


# ═══════════════════════════════════════════════════════════════════
#  Robot Registry
# ═══════════════════════════════════════════════════════════════════

# Pre-registered robots with metadata
REGISTERED_ROBOTS = {
    "unitree_g1": {
        "urdf": "robots/unitree_g1/g1_29dof.urdf",
        "display_name": "Unitree G1 Humanoid",
        "embodiment": "humanoid",
        "description": "Unitree G1 — 29-DOF bipedal humanoid (1.27m, 35.27kg). "
                       "23 actuated joints + head. Primary embodiment for FLEET-Safe VLA.",
    },
    "unitree_go2": {
        "urdf": "robots/unitree_go2/go2.urdf",
        "display_name": "Unitree Go2 Quadruped",
        "embodiment": "quadruped",
        "description": "Unitree Go2 — 12-DOF quadruped (0.40m body, ~15kg). "
                       "Hospital patrol and outdoor navigation.",
    },
    "turtlebot3_burger": {
        "urdf": "robots/turtlebot3/turtlebot3.urdf",
        "display_name": "TurtleBot3 Burger",
        "embodiment": "diff_drive",
        "description": "TurtleBot3 Burger — 2-DOF differential drive. "
                       "Baseline navigation platform for SaferPath comparison.",
    },
    "fetch": {
        "urdf": "robots/fetch/fetch.urdf",
        "display_name": "Fetch Mobile Manipulator",
        "embodiment": "manipulator",
        "description": "Fetch Robotics — 11-DOF mobile manipulator (base + 7-DOF arm). "
                       "Manipulation benchmarks and medication delivery.",
    },
}


class RobotRegistry:
    """Registry of all available robot models parsed from URDF files."""

    def __init__(self):
        self._robots: Dict[str, RobotModel] = {}
        self._load_registered()

    def _load_registered(self):
        """Load all pre-registered robots from URDF files."""
        project_root = ROBOTS_DIR.parent
        for robot_id, meta in REGISTERED_ROBOTS.items():
            urdf_path = project_root / meta["urdf"]
            if urdf_path.exists():
                try:
                    model = parse_urdf(urdf_path)
                    model.id = robot_id
                    model.display_name = meta["display_name"]
                    model.embodiment = meta["embodiment"]
                    model.description = meta["description"]
                    self._robots[robot_id] = model
                except Exception as e:
                    print(f"[RobotRegistry] Error parsing {urdf_path}: {e}")

    def add_robot(self, robot_id: str, urdf_path: Path,
                  display_name: str = "", embodiment: str = "unknown",
                  description: str = "") -> RobotModel:
        """Add a new robot from a URDF file."""
        model = parse_urdf(urdf_path)
        model.id = robot_id
        if display_name:
            model.display_name = display_name
        model.embodiment = embodiment
        model.description = description
        self._robots[robot_id] = model
        return model

    def list_robots(self) -> List[dict]:
        """List all robots (summary view)."""
        return [r.to_summary() for r in self._robots.values()]

    def get_robot(self, robot_id: str) -> Optional[dict]:
        """Get full robot detail including joints and links."""
        r = self._robots.get(robot_id)
        return r.to_dict() if r else None

    def get_robot_summary(self, robot_id: str) -> Optional[dict]:
        r = self._robots.get(robot_id)
        return r.to_summary() if r else None

    @property
    def robot_ids(self) -> List[str]:
        return list(self._robots.keys())

    def __len__(self) -> int:
        return len(self._robots)


# ═══════════════════════════════════════════════════════════════════
#  Singleton
# ═══════════════════════════════════════════════════════════════════

_registry: Optional[RobotRegistry] = None


def get_robot_registry() -> RobotRegistry:
    global _registry
    if _registry is None:
        _registry = RobotRegistry()
    return _registry


if __name__ == "__main__":
    reg = get_robot_registry()
    print(f"═══ FLEET-Safe VLA Robot Registry ═══")
    print(f"  Robots loaded: {len(reg)}")
    print()
    for r in reg.list_robots():
        print(f"  🤖 {r['id']:<22s} | {r['display_name']:<28s} | "
              f"{r['actuated_dof']:>2d} DoF | {r['total_mass_kg']:>7.2f} kg | "
              f"{r['embodiment']:<12s} | sensors: {', '.join(r['sensors'])}")
