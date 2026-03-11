#!/usr/bin/env python3
"""
server/model_registry.py — FLEET-Safe VLA Unified Model Registry

Consolidates all 13 trained models with metadata, training results,
W&B links, and HuggingFace references into a single queryable registry.

Sources:
  - training_logs/extended/*.json      (8 extended model reports)
  - training_logs/fastbot_diffusion_nav/result.json
  - training_logs/g1_cmdp_locomotion/result.json
  - training/fleet_extended_train.py   MODEL_CONFIGS
  - training/groot_n1_backbone.py      GR00TConfig
  - training/visual_reasoning.py       VisualReasonerConfig
  - notebooks/09_auto_train_orchestrator.py  TRAINABLE_MODELS
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent


# ═══════════════════════════════════════════════════════════════════
#  Model Entry
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModelEntry:
    """A single model in the registry."""
    id: str
    name: str
    display_name: str
    category: str              # core, extended, safety, policy, vla
    description: str

    # Architecture
    backbone: str = "GR00T-N1.6"
    obs_dim: int = 0
    act_dim: int = 0
    parameters: int = 0
    finetuning: str = "full"   # full, lora, adapter, frozen

    # Training
    training_status: str = "pending"   # trained, dry-run, pending
    epochs: int = 0
    final_loss: float = 0.0
    best_loss: float = 0.0
    final_svr: float = 0.0
    training_time_s: float = 0.0
    device: str = "cuda"
    timestamp: str = ""

    # Results (model-specific)
    training_results: Dict = field(default_factory=dict)

    # W&B
    wandb_project: str = "fleet-safe-vla"
    wandb_entity: str = "FrankAsanteVanLaarhoven"
    wandb_run_id: str = ""
    wandb_url: str = ""

    # HuggingFace
    huggingface_repo: str = "FrankAsanteVanLaarhoven/fleet-safe-vla"

    # Checkpoints
    checkpoint_path: str = ""
    onnx_path: str = ""

    # Training script
    training_script: str = ""

    # Tags
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════

class ModelRegistry:
    """Unified registry of all FLEET-Safe VLA trained models."""

    def __init__(self):
        self._models: Dict[str, ModelEntry] = {}
        self._load_all()

    def _load_all(self):
        """Load all models from static definitions + training reports."""
        self._register_core_models()
        self._register_extended_models()
        self._register_additional_models()
        self._load_training_reports()

    # ─── Core Models (1-2) ────────────────────────────────────────

    def _register_core_models(self):
        self._models["fastbot_diffusion_nav"] = ModelEntry(
            id="fastbot_diffusion_nav",
            name="fastbot_diffusion_nav",
            display_name="FastBot DiffusionPolicy",
            category="core",
            description="FastBot hospital navigation using Diffusion Policy "
                        "(TemporalUNet) with CBF-QP safety filter",
            backbone="GR00T-N1.6 DiT + TemporalUNet",
            obs_dim=48, act_dim=2,
            finetuning="full",
            wandb_run_id="groot-fastbot-0308-1408",
            wandb_url="https://wandb.ai/FrankAsanteVanLaarhoven/fleet-safe-vla/runs/groot-fastbot-0308-1408",
            checkpoint_path="checkpoints/groot_fastbot/best.pt",
            onnx_path="checkpoints/fastbot_diffusion_nav/fastbot_policy.onnx",
            training_script="training/groot_wb_train.py",
            tags=["groot", "fastbot", "diffusion", "cbf", "core"],
        )

        self._models["g1_cmdp_locomotion"] = ModelEntry(
            id="g1_cmdp_locomotion",
            name="g1_cmdp_locomotion",
            display_name="G1 CMDP Safe Locomotion",
            category="core",
            description="Unitree G1 Constrained MDP with PPO-Lagrangian "
                        "for safe bipedal locomotion + 3-stage safety filter",
            backbone="GR00T-N1.6 DiT + PPO-Lagrangian",
            obs_dim=48, act_dim=23,
            finetuning="full",
            wandb_run_id="groot-g1-cmdp-0308-1408",
            wandb_url="https://wandb.ai/FrankAsanteVanLaarhoven/fleet-safe-vla/runs/groot-g1-cmdp-0308-1408",
            checkpoint_path="checkpoints/groot_g1/best.pt",
            onnx_path="checkpoints/g1_cmdp_locomotion/g1_locomotion.onnx",
            training_script="training/groot_wb_train.py",
            tags=["groot", "g1", "cmdp", "safety", "cbf", "core"],
        )

    # ─── Extended Models (3-10) ───────────────────────────────────

    def _register_extended_models(self):
        extended = [
            ModelEntry(
                id="dseo_monitor",
                name="dseo_monitor",
                display_name="DSEO Runtime Monitor",
                category="extended",
                description="Deadline-sensitive safety envelope orchestrator — "
                            "QoS-aware DDS monitoring with hysteresis mode transitions",
                obs_dim=8, act_dim=3, epochs=150,
                training_script="training/fleet_extended_train.py",
                tags=["dseo", "real-time", "safety-envelope", "extended"],
            ),
            ModelEntry(
                id="zone_navigator",
                name="zone_navigator",
                display_name="Hospital Zone Navigator",
                category="extended",
                description="12-zone hospital navigation policy with zone-specific "
                            "speed limits and reward shaping",
                obs_dim=28, act_dim=2, epochs=250,
                training_script="training/fleet_extended_train.py",
                tags=["zone", "navigation", "12-zone", "extended"],
            ),
            ModelEntry(
                id="robopocket",
                name="robopocket",
                display_name="RoboPocket Online Finetuner",
                category="extended",
                description="Phone-based online policy iteration using RLPD-weighted "
                            "replay with expert/medium/noisy demonstrations",
                obs_dim=26, act_dim=4, epochs=200,
                training_script="training/fleet_extended_train.py",
                tags=["robopocket", "online", "RLPD", "extended"],
            ),
            ModelEntry(
                id="cognitive_safety",
                name="cognitive_safety",
                display_name="7D Cognitive Safety Model",
                category="safety",
                description="7-dimensional cognitive safety state space — "
                            "[CBF, STL, COM, zone, jitter, collision_risk, trust]",
                obs_dim=17, act_dim=3, epochs=300,
                training_script="training/fleet_extended_train.py",
                tags=["cognitive", "7D-safety", "trust", "extended"],
            ),
            ModelEntry(
                id="benchmark_suite",
                name="benchmark_suite",
                display_name="Benchmark Suite Evaluator",
                category="extended",
                description="Comprehensive benchmark aggregating all 8 safety metrics "
                            "for cross-embodiment evaluation",
                obs_dim=28, act_dim=2, epochs=200,
                training_script="training/fleet_extended_train.py",
                tags=["benchmark", "aggregate", "8-metrics", "extended"],
            ),
            ModelEntry(
                id="sim2real",
                name="sim2real",
                display_name="Sim-to-Real Transfer Agent",
                category="extended",
                description="Domain randomisation with friction/mass/camera/light "
                            "perturbations + ONNX export for edge deployment",
                obs_dim=29, act_dim=6, epochs=250,
                training_script="training/fleet_extended_train.py",
                tags=["sim2real", "domain-rand", "ONNX", "extended"],
            ),
            ModelEntry(
                id="fleet_coord",
                name="fleet_coord",
                display_name="Fleet Coordinator",
                category="extended",
                description="Multi-robot (8-robot) task allocation with priority "
                            "scoring and collision avoidance",
                obs_dim=65, act_dim=8, epochs=200,
                training_script="training/fleet_extended_train.py",
                tags=["fleet", "coordination", "multi-robot", "extended"],
            ),
            ModelEntry(
                id="semantic_collector",
                name="semantic_collector",
                display_name="Semantic Data Collector",
                category="extended",
                description="VLM auto-annotation pipeline with 10 object classes, "
                            "risk estimation, and confidence-gated labeling",
                obs_dim=45, act_dim=12, epochs=200,
                training_script="training/fleet_extended_train.py",
                tags=["semantic", "VLM", "auto-annotation", "extended"],
            ),
        ]
        for m in extended:
            m.backbone = "GR00T-N1.6 Transformer (768d, 12L, 16H)"
            self._models[m.id] = m

    # ─── Additional Models ────────────────────────────────────────

    def _register_additional_models(self):
        self._models["visual_reasoning"] = ModelEntry(
            id="visual_reasoning",
            name="visual_reasoning",
            display_name="Visual Reasoning (Galatolo)",
            category="vla",
            description="Lightweight Visual Reasoning for Socially-Aware Robots — "
                        "Gated MLP + Patch Unmerger two-pass architecture "
                        "(Galatolo et al. 2026)",
            backbone="Qwen 2.5 VL 7B + Gated MLP Reasoner",
            obs_dim=3584, act_dim=5,
            finetuning="lora",
            training_script="training/visual_reasoning.py",
            tags=["visual-reasoning", "galatolo", "social-awareness", "vla"],
        )

        self._models["groot_fastbot_backbone"] = ModelEntry(
            id="groot_fastbot_backbone",
            name="groot_fastbot_backbone",
            display_name="GR00T FastBot Backbone",
            category="core",
            description="NVIDIA GR00T N1.6 backbone policy with DiT action head "
                        "and CBF-QP safety filter — FastBot embodiment",
            backbone="GR00T-N1.6 (DiT-12L-16H + CBF-QP)",
            obs_dim=32, act_dim=2,
            wandb_run_id="groot-fastbot-0308-1408",
            wandb_url="https://wandb.ai/FrankAsanteVanLaarhoven/fleet-safe-vla/runs/groot-fastbot-0308-1408",
            checkpoint_path="checkpoints/groot_fastbot/best.pt",
            training_script="training/groot_n1_backbone.py",
            tags=["groot", "backbone", "fastbot", "DiT"],
        )

        self._models["groot_g1_backbone"] = ModelEntry(
            id="groot_g1_backbone",
            name="groot_g1_backbone",
            display_name="GR00T G1 CMDP Backbone",
            category="core",
            description="NVIDIA GR00T N1.6 backbone policy with DiT action head, "
                        "CBF-QP safety filter, and PPO-Lagrangian — G1 embodiment",
            backbone="GR00T-N1.6 (DiT-12L-16H + PPO-Lagrangian + CBF-QP)",
            obs_dim=48, act_dim=23,
            wandb_run_id="groot-g1-cmdp-0308-1408",
            wandb_url="https://wandb.ai/FrankAsanteVanLaarhoven/fleet-safe-vla/runs/groot-g1-cmdp-0308-1408",
            checkpoint_path="checkpoints/groot_g1/best.pt",
            training_script="training/groot_n1_backbone.py",
            tags=["groot", "backbone", "g1", "cmdp", "DiT"],
        )

    # ─── Load Training Reports ────────────────────────────────────

    def _load_training_reports(self):
        """Load training results from JSON reports on disk."""

        # Extended model reports
        ext_dir = PROJECT_ROOT / "training_logs" / "extended"
        report_map = {
            "dseo_monitor": "dseo_monitor_report.json",
            "zone_navigator": "zone_navigator_report.json",
            "robopocket": "robopocket_report.json",
            "cognitive_safety": "cognitive_safety_report.json",
            "benchmark_suite": "benchmark_suite_report.json",
            "sim2real": "sim2real_report.json",
            "fleet_coord": "fleet_coord_report.json",
            "semantic_collector": "semantic_collector_report.json",
        }
        for model_id, filename in report_map.items():
            path = ext_dir / filename
            if path.exists() and model_id in self._models:
                try:
                    data = json.loads(path.read_text())
                    m = self._models[model_id]
                    m.training_status = "trained"
                    m.parameters = data.get("parameters", 0)
                    m.epochs = data.get("epochs", m.epochs)
                    m.final_loss = data.get("final_loss", 0.0)
                    m.best_loss = data.get("best_loss", 0.0)
                    m.final_svr = data.get("final_svr", 0.0)
                    m.training_time_s = data.get("training_time_s", 0.0)
                    m.device = data.get("device", "cuda")
                    m.timestamp = data.get("timestamp", "")
                    m.training_results = data
                except Exception:
                    pass

        # Core model reports
        core_reports = {
            "fastbot_diffusion_nav": "training_logs/fastbot_diffusion_nav/result.json",
            "g1_cmdp_locomotion": "training_logs/g1_cmdp_locomotion/result.json",
        }
        for model_id, rel_path in core_reports.items():
            path = PROJECT_ROOT / rel_path
            if path.exists() and model_id in self._models:
                try:
                    data = json.loads(path.read_text())
                    m = self._models[model_id]
                    status = data.get("status", "unknown")
                    m.training_status = "trained" if status == "success" else status
                    m.final_loss = data.get("final_loss", 0.0)
                    m.training_time_s = data.get("elapsed_sec", 0.0)
                    metrics = data.get("metrics", {})
                    m.training_results = metrics
                    m.parameters = metrics.get("parameters", 0)
                    if "svr" in metrics:
                        m.final_svr = metrics["svr"]
                except Exception:
                    pass

        # GR00T backbone inherits from core
        for backbone_id, core_id in [
            ("groot_fastbot_backbone", "fastbot_diffusion_nav"),
            ("groot_g1_backbone", "g1_cmdp_locomotion"),
        ]:
            if core_id in self._models and backbone_id in self._models:
                core = self._models[core_id]
                bb = self._models[backbone_id]
                bb.training_status = core.training_status
                bb.parameters = core.parameters
                bb.timestamp = core.timestamp

    # ─── Public API ───────────────────────────────────────────────

    def list_models(self) -> List[dict]:
        """Return all models as dicts."""
        return [m.to_dict() for m in self._models.values()]

    def get_model(self, model_id: str) -> Optional[dict]:
        """Get a single model by ID."""
        m = self._models.get(model_id)
        return m.to_dict() if m else None

    def get_model_entry(self, model_id: str) -> Optional[ModelEntry]:
        """Get the raw ModelEntry."""
        return self._models.get(model_id)

    def list_by_category(self, category: str) -> List[dict]:
        """Filter models by category."""
        return [m.to_dict() for m in self._models.values()
                if m.category == category]

    def training_summary(self) -> dict:
        """Aggregate training summary across all models."""
        models = list(self._models.values())
        trained = [m for m in models if m.training_status == "trained"]
        total_params = sum(m.parameters for m in models if m.parameters > 0)
        total_time = sum(m.training_time_s for m in trained)

        return {
            "total_models": len(models),
            "trained": len(trained),
            "pending": len([m for m in models if m.training_status == "pending"]),
            "dry_run": len([m for m in models if m.training_status == "dry-run"]),
            "total_parameters": total_params,
            "total_training_time_s": total_time,
            "total_training_time_h": round(total_time / 3600, 2),
            "categories": list(set(m.category for m in models)),
            "wandb_project": "fleet-safe-vla",
            "wandb_entity": "FrankAsanteVanLaarhoven",
            "huggingface_repo": "FrankAsanteVanLaarhoven/fleet-safe-vla",
            "models_by_category": {
                cat: len([m for m in models if m.category == cat])
                for cat in set(m.category for m in models)
            },
            "zero_svr_models": len([m for m in trained if m.final_svr == 0.0]),
        }

    @property
    def model_ids(self) -> List[str]:
        return list(self._models.keys())

    def __len__(self) -> int:
        return len(self._models)


# ═══════════════════════════════════════════════════════════════════
#  Singleton
# ═══════════════════════════════════════════════════════════════════

_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get or create the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


if __name__ == "__main__":
    registry = get_registry()
    print(f"═══ FLEET-Safe VLA Model Registry ═══")
    print(f"  Total models: {len(registry)}")
    summary = registry.training_summary()
    print(f"  Trained:      {summary['trained']}")
    print(f"  Parameters:   {summary['total_parameters']:,}")
    print(f"  Training:     {summary['total_training_time_h']}h")
    print()
    for m in registry.list_models():
        status = "✅" if m["training_status"] == "trained" else "⏳"
        print(f"  {status} {m['id']:<30s} | {m['display_name']:<35s} | "
              f"{m['parameters']:>12,} params | {m['category']}")
