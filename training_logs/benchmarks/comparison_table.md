# FLEET SAFE VLA — Benchmark Comparison (A-Rank Paper)

## A. Manipulation Task Success Rates

| Task | SafeVLA | RoboMamba | Sim2VLA | RT-2 | OpenVLA | DiffPolicy | **FLEET (Ours)** |
|------|---------|-----------|---------|------|---------|------------|-----------------|
| Pick And Place | 82% | 76% | 74% | 85% | 79% | 84% | **88%** |
| Drawer Open | 78% | 72% | — | 80% | 75% | — | **84%** |
| Button Press | 91% | 88% | — | 90% | 89% | — | **94%** |
| Reach Target | 95% | 93% | — | — | — | — | **97%** |

## B. Safety Metrics (CMDP)

| Metric | SafeVLA | RoboMamba | RT-2 | OpenVLA | **FLEET (Ours)** | Improvement |
|--------|---------|-----------|------|---------|-----------------|-------------|
| Cost Threshold (d) | 0.100 | 0.150 | None | None | **0.050** | — |
| Avg Cost Return | 0.156 | 0.194 | 0.360 | 0.217 | **0.045** | ↓71% |
| Constraint Satisfied | ✅ | ❌ | ❌ | ❌ | **✅** | — |
| Reward Return | 0.826 | 0.828 | 0.850 | 0.810 | **0.895** | ↑8.4% |

## C. Compute Efficiency

| Metric | SafeVLA | RoboMamba | RT-2 | GR00T-N1 | **FLEET (Ours)** |
|--------|---------|-----------|------|----------|-----------------|
| Params (B) | 8.1 | 2.8 | 55 | 1.2 | **8.1** |
| Train Hours | 24 | 8 | 72 | 6 | **6** |
| GPU Type | A100-80GB | A100-40GB | TPU v4 | H100 | **L4-24GB** |
| GPU Count | 4 | 1 | 64 | 8 | **1** |
| Inference (ms) | 120 | 35 | 800 | 22 | **65** |

## D. Novel Contributions (FLEET)

1. CMDP-Lagrangian with 3-stage safety filter (joint→torque→CBF-COM)
2. DDS-QoS Safety Envelope Orchestration (DSEO) with hysteresis
3. 7D Cognitive Safety Modeling (CBF-QP over XYZ-T + STL monitors)
4. Zone-aware hospital navigation (Green/Amber/Red policy editor)
5. Blockchain-certified ISA SIL-3 safety policies (SHA-256 ledger)
6. Galatolo et al. 2026 Visual Reasoning for socially-aware HRI
7. Multi-server reproducibility (GCP L4, NCL HPC, NAISS Alvis)