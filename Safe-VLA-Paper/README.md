# FLEET-Safe VLA — LaTeX Submission Package

## Authors
**Frank Asante Van Laarhoven** — Newcastle University, School of Computing

## Quick Start

### Prerequisites
- LaTeX distribution: [TeX Live 2024+](https://www.tug.org/texlive/) or [MacTeX](https://www.tug.org/mactex/)
- Required packages: `IEEEtran`, `tikz`, `booktabs`, `siunitx`, `algorithmic`, `subcaption`, `hyperref`

### Compile Main Paper
```bash
# Navigate to the latex directory
cd preprint/latex

# Compile (run twice for references)
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Compile Supplementary Materials
```bash
pdflatex supplementary.tex
bibtex supplementary
pdflatex supplementary.tex
pdflatex supplementary.tex
```

### One-Command Build (both documents)
```bash
cd preprint/latex
for doc in main supplementary; do
  pdflatex "$doc.tex" && bibtex "$doc" && pdflatex "$doc.tex" && pdflatex "$doc.tex"
done
echo "Build complete — see main.pdf and supplementary.pdf"
```

## Package Contents

```
preprint/latex/
├── main.tex                    # 8-page IEEE conference paper
├── supplementary.tex           # Extended hyperparams, ablations, data stats
├── references.bib              # 45 BibTeX entries
├── figures/
│   ├── fig1_g1_primary_panels.png       # G1 CMDP training curves
│   ├── fig2_fastbot_secondary_panels.png # FastBot training curves
│   ├── fig3_final_training_panels.png    # COM margin + action jitter
│   └── fig4_wandb_overview.png           # Combined W&B dashboard
└── README.md                   # This file
```

## Paper Structure

| Section | Title | Content |
|---------|-------|---------|
| I | Introduction | Problem statement, 3 gaps identified, 4 contributions |
| II | Related Work | VLA models, safe RL, CBFs, hospital robotics, foundation models |
| III | Problem Formulation | CMDP definition, STL specs, CBF forward invariance |
| IV | Methodology | Architecture (TikZ diagram), GR00T backbone, CBF-QP, DSEO, Algorithm 1 |
| V | Experimental Design | Simulation setup, 9 baselines, 8 metrics, hardware specs |
| VI | Results | Main benchmark (Table II), convergence (Fig. 2), ablations (Table III), cross-embodiment (Table IV) |
| VII | Discussion | Limitations, sim-to-real strategy, ethical considerations |
| VIII | Conclusion | Summary of contributions and key numbers |

## Key Results

| Metric | FLEET-Safe VLA | SafeVLA | Improvement |
|--------|---------------|---------|-------------|
| Safety Cost | 0.0007 | 0.156 | ↓88.5% |
| SVR | 5×10⁻⁵ | 0.030 | ↓99.8% |
| Latency | <8ms | 120ms | ↓93.3% |
| Compute Cost | $0.49 | $96 | ↓14× |
| Provable Safety | ✅ Yes | ❌ No | — |

## Target Venues

| Venue | Page Limit | Deadline (2026) | Template |
|-------|-----------|-----------------|----------|
| CoRL 2026 | 8 pages | Jul 15 | OpenReview |
| NeurIPS 2026 | 9 pages | May 22 | NeurIPS LaTeX |
| ICRA 2027 | 6–8 pages | Sep 15 | IEEE |
| RSS 2027 | 8 pages | Feb 1 (2027) | LaTeX |

## Online Resources

- **Code**: [github.com/FrankAsanteVanLaarhoven/Fleet-Safe-VLA-FastBots-G1](https://github.com/FrankAsanteVanLaarhoven/Fleet-Safe-VLA-FastBots-G1)
- **W&B**: [wandb.ai/f-a-v-l/fleet-safe-vla](https://wandb.ai/f-a-v-l/fleet-safe-vla)
- **FastBot run**: [runs/hscmj967](https://wandb.ai/f-a-v-l/fleet-safe-vla/runs/hscmj967)
- **G1 CMDP run**: [runs/19k6gj6p](https://wandb.ai/f-a-v-l/fleet-safe-vla/runs/19k6gj6p)

## Licence

© 2026 Frank Asante Van Laarhoven. All rights reserved.
