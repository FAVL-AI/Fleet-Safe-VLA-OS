# FLEET SAFE VLA - HFB-S

**State-of-the-Art Fleet Autonomy & Safety for Humanoid Robots**

Digital Twin Command Center with WebRTC streaming, VLA inference, RoboPocket phone-based policy iteration, DDS Safety Envelope Orchestrator (DSEO), and fleet control for Unitree G1 humanoid robots.

> Built on GCP G2 GPU instances with Isaac Sim 4.2.0 + ROS 2 Humble

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  FLEET SAFE VLA - HFB-S                                         │
├──────────────┬───────────────┬────────────────┬──────────────────┤
│ Command      │ Fleet         │ RoboPocket     │ Safety           │
│ Center (PWA) │ Controller    │ System         │ Layer            │
├──────────────┼───────────────┼────────────────┼──────────────────┤
│ Dashboard    │ DDS Bridge    │ Inference Srv  │ DSEO Node        │
│ 3D Viewport  │ Policy Engine │ AR Foresight   │ Safety Monitor   │
│ Widget Mgr   │ FSM Control   │ Online Finetune│ MDP Extensions   │
│ SDK Launcher │ Arm Control   │ SLAM Monitor   │ QoS Profiles     │
│ WebRTC       │ Rewards       │ BLE Gripper    │ Metrics Pub      │
│              │               │ Multi-Device   │ E-Stop           │
└──────────────┴───────────────┴────────────────┴──────────────────┘
         │                │                │
    ┌────┴────┐     ┌─────┴─────┐    ┌────┴────┐
    │ GCP VM  │     │ CycloneDDS│    │ Isaac   │
    │ G2 GPU  │     │ QoS       │    │ Sim 4.2 │
    └─────────┘     └───────────┘    └─────────┘
```

## Components

### 🤖 Command Center (`fastbot_command_center.html`)
- **Ephemeral Dashboard** — Drag-and-drop floating widgets, fully customisable layout
- **3-State Sidebar** — Rail (4px) → Icon-only (48px) → Full (280px) with premium SVG icons
- **12 Service Widgets** — Fleet, VLA, Dataset, Telemetry, Safety, DDS, Network, Cameras, Log, RoboPocket, DSEO, Safety Metrics
- **PWA** — Installable, offline-capable, with downloadable SDK launcher
- **WebRTC** — Low-latency Isaac Sim streaming via GCP

### 🧠 Fleet Controller (`fleet/`)
| Module | Purpose |
|--------|---------|
| `dds_bridge.py` | CycloneDDS ↔ Python bridge |
| `policy_engine.py` | OpenVLA Llama-2 inference (GR00T fallback) |
| `fsm_controller.py` | Finite State Machine for robot behavior |
| `arm_controller.py` | Dual arm manipulation |
| `rewards.py` | Reward functions for RL training |
| `mdp_safe_extensions.py` | Safety observables, rewards, terminations, action filter, C-walk |
| `safe_g1_env_cfg.py` | Full environment config with curriculum |
| `dseo_node.py` | DDS Safety Envelope Orchestrator — risk scoring + mode switching |
| `safety_monitor_node.py` | Hard E-stop controller with command watchdog |
| `dds_metrics_publisher.py` | Per-topic deadline, latency, liveliness metrics |

### 📱 RoboPocket (`robopocket/`)
Phone-based policy iteration — improve robot policies without a robot.

| Module | Purpose |
|--------|---------|
| `inference_server.py` | FastAPI DiffusionPolicy server (<150ms RTT) |
| `ar_visual_foresight.py` | AR coin-path trajectory projection |
| `data_serving_node.py` | RLPD 50/50 offline/online batch sampler |
| `online_finetuning.py` | Async DDPM training with model sync |
| `isomorphic_gripper.py` | ESP32 BLE + Jacobian DLS IK solver |
| `slam_quality_monitor.py` | 5-stage VIO validation |
| `multi_device_sync.py` | Cristian's clock sync + ARKit map merge |

### 🔒 Safety Layer
- **3 DSEO Modes**: Normal (20ms QoS) → Degraded (10ms) → Emergency (5ms)
- **Hysteresis** mode switching prevents chattering
- **Hard E-stop** with command watchdog and safe-stop commands
- **Safety MDP**: COM margin rewards (weight=5.0), contact force limits, progressive curriculum

### 🏗️ Pipeline (`pipeline/`)
- OpenVLA training scripts (with GR00T fallback support)
- HDF5 → LeRobot dataset conversion
- CycloneDDS XML configuration
- DDS QoS profiles (Normal/Degraded/Emergency)

---

## GCP Server Setup

### Prerequisites
- GCP account with GPU quota (G2 series recommended)
- Docker & Docker Compose
- Python 3.10+

### 1. Provision GCP VM
```bash
# Create G2 GPU instance for Isaac Sim
./setup_isaac_sim_vm.sh

# Install Isaac Sim dependencies
./install_isaac_deps.sh
```

### 2. Launch the Server
```bash
# Start the FastAPI server + WebRTC signaling
cd server && pip install -r requirements.txt
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# Or use Docker
docker-compose up -d
```

### 3. Launch Isaac Sim + ROS 2
```bash
# Launch hospital simulation
./launch_fastbot_hospital.sh

# Or launch Isaac Lab with ROS 2
./launch_isaac_lab_ros2.sh
```

### 4. Open the Dashboard
Navigate to `http://<GCP_EXTERNAL_IP>:8000` — the Command Center PWA loads automatically.

---

## Quick Start (Local Development)

```bash
# Clone
git clone https://github.com/FrankAsanteVanLaarhoven/Fleet-Safe-VLA-FastBots-G1.git
cd Fleet-Safe-VLA-FastBots-G1

# Install server deps
pip install -r server/requirements.txt

# Start server
python -m uvicorn server.api:app --host 0.0.0.0 --port 8000

# Open browser
open http://localhost:8000
```

---

## Project Structure

```
├── fastbot_command_center.html   # Main PWA dashboard
├── fleet/                        # Fleet controller + safety layer
│   ├── dds_bridge.py
│   ├── policy_engine.py
│   ├── dseo_node.py
│   ├── safety_monitor_node.py
│   ├── mdp_safe_extensions.py
│   └── ...
├── robopocket/                   # Phone-based policy iteration
│   ├── inference_server.py
│   ├── ar_visual_foresight.py
│   ├── online_finetuning.py
│   └── ...
├── notebooks/                    # SOTA training notebooks (NB01-NB10)
│   ├── 01_environment_setup.py
│   ├── 02_safe_locomotion_training.py
│   ├── 03_dseo_runtime_training.py
│   ├── 04_hospital_navigation.py
│   ├── 05_robopocket_finetuning.py
│   ├── 06_diffusion_policy_training.py
│   ├── 07_cognitive_7d_modeling.py
│   ├── 08_benchmark_metrics.py
│   ├── 09_auto_train_orchestrator.py
│   └── 10_sim_to_real_transfer.py
├── training/                     # Training infrastructure
│   ├── visual_reasoning.py       # Galatolo et al. 2026 VLM reasoner
│   └── auto_shutdown.py          # GCP cost optimization
├── pipeline/                     # Training & DDS config
│   ├── cyclonedds.xml
│   ├── g1_safety_qos.xml
│   └── train_groot.sh
├── server/                       # FastAPI + WebRTC
├── pwa/                          # PWA assets
├── .github/workflows/
│   ├── deploy.yml                # Pages deploy
│   ├── test-matrix.yml           # Full test matrix
│   └── training-ci.yml           # Training CI/CD pipeline
├── Dockerfile.training           # Multi-server GPU training image
├── deploy_training.sh            # Multi-server deploy script
├── requirements_training.txt     # ML dependencies
├── wandb.settings                # Weights & Biases config
└── docker-compose.yml
```

---

## 🧠 Training Infrastructure

### SOTA Training Notebooks

| # | Notebook | Methods |
|---|----------|---------|
| 01 | Environment Setup | Dependency check, CUDA validation, reproducibility |
| 02 | Safe Locomotion | CMDP Lagrangian, 3-stage safety filter, curriculum |
| 03 | DSEO Runtime | QoS profiles, hysteresis mode switching |
| 04 | Hospital Navigation | Zone-aware, 12 reward functions |
| 05 | RoboPocket Finetune | RLPD 50/50, DDPM, Jacobian DLS IK |
| 06 | Diffusion Policy | ResNet-18 + Temporal U-Net, EMA, cosine LR |
| 07 | 7D Cognitive | CBF-QP safety filter, STL robustness |
| 08 | Benchmark Suite | 8 metrics × 17 models, blockchain certification |
| 09 | Auto-Train | One-click orchestrator, budget, GCP auto-shutdown |
| 10 | Sim-to-Real | Domain randomization, ONNX export, multi-platform |

### Benchmarked Models (17)

| Model | Type | Params | Finetuning |
|-------|------|--------|------------|
| **SafeVLA (Ours)** | VLA+Safety | 8.1B | LoRA-r16 |
| LLaMA-3.1-8B-VLA | VLA | 8B | LoRA-r16 |
| LLaMA-3.1-70B-VLA | VLA | 70B | LoRA-r64 |
| BERT-Safety-Classifier | VLA | 110M | Full |
| OpenVLA-7B | VLA | 7B | LoRA-r32 |
| RT-2-PaLM-E | VLA | 55B | Frozen |
| Octo-Base | VLA | 93M | Full |
| RoboMamba | Safety | 2.8B | Full (SSM) |
| Sim2VLA | Safety | 7.2B | Adapter |
| DiffusionPolicy | Policy | 25.5M | Full |
| GR00T-N1 | Policy | 1.2B | Adapter |
| π₀-Flow | Policy | 3B | LoRA |

### VLM Visual Reasoning (Galatolo et al. 2026)

Integrated from [VLM-Reasoning-for-Robotics](https://github.com/alessioGalatolo/VLM-Reasoning-for-Robotics):
- Gated MLP language-to-vision feedback loop (<3% extra params)
- Two-pass training: reasoning hint → visual reinterpretation
- Hospital HRI: intention recognition (5 classes), zone understanding (6 classes)
- Compatible backbones: Qwen 2.5 VL (7B), Gemma 3 (4B), LLaVA-OV 1.5 (4B)

### Benchmark Metrics

| Metric | Name | Target |
|--------|------|--------|
| DMR | Deadline Miss Rate | < 0.1% |
| AJ | Action Jitter | < 0.05 |
| TTP | Time to Preempt | < 50ms |
| SVR | Safety Violation Rate | 0% |
| STL ρ | Temporal Logic Robustness | > 0 |
| η | Energy Efficiency | > 0.8 |

---

## 🚀 Training Quick Start

### One-Click Training
```bash
# Install dependencies
pip install -r requirements_training.txt

# Train all models (dry-run)
python notebooks/09_auto_train_orchestrator.py --train-all --dry-run

# Train specific model
python notebooks/09_auto_train_orchestrator.py --train SafeVLA --dry-run

# Run benchmarks
python notebooks/08_benchmark_metrics.py --dry-run --export-csv

# Visual reasoning training
python training/visual_reasoning.py --backbone qwen_2.5_7b --dry-run
```

### Multi-Server Deployment
```bash
# GCP g2-standard-4 (NVIDIA L4)
./deploy_training.sh gcp-l4 SafeVLA --dry-run

# Newcastle University HPC (via SSH)
./deploy_training.sh ncl-hpc SafeVLA --dry-run

# NAISS Alvis C3SE (4×A100, as per Galatolo et al. 2026)
./deploy_training.sh naiss SafeVLA --dry-run

# Docker (any GPU machine)
./deploy_training.sh docker SafeVLA --dry-run

# Local GPU
./deploy_training.sh local SafeVLA --dry-run
```

### Docker Training
```bash
docker build -t fleet-safe-vla:latest -f Dockerfile.training .
docker run --gpus all \
  -v $(pwd)/training_logs:/app/training_logs \
  --env-file .env \
  fleet-safe-vla:latest \
  python notebooks/09_auto_train_orchestrator.py --train-all --dry-run
```

---

## 🔄 CI/CD Pipeline

GitHub Actions automatically triggers on pushes to `notebooks/`, `training/`, or `fleet/`:

| Job | Trigger | Purpose |
|-----|---------|---------|
| **Lint** | Every push | `py_compile` + `ruff` on all training code |
| **Test** | Every push | Dry-run tests on Python 3.10 & 3.11 |
| **Security** | Every push | `bandit` vulnerability scan |
| **Benchmark** | After tests | Full benchmark suite with CSV export |
| **W&B Sync** | Main only | Upload metrics to Weights & Biases |
| **HF Registry** | Main only | Push model card to HuggingFace (private) |
| **GPU Training** | Manual dispatch | Train any model on any server |

### Experiment Tracking
- **Weights & Biases**: All training metrics, model artifacts, and code logged to `fleet-safe-vla` project
- **HuggingFace**: Private model registry at `FrankAsanteVanLaarhoven/fleet-safe-vla`
- **Blockchain Certification**: SHA-256 ledger for ISA SIL-3 safety compliance

---

## 🔒 Security

- API keys stored in `.env` (gitignored) locally, encrypted GitHub Secrets for CI/CD
- No secrets in committed code — CI/CD jobs gracefully skip if secrets unavailable
- `bandit` security scan on every push
- Blockchain audit trail for all deployed models

---

## License

MIT License — see [LICENSE](LICENSE)

