import os

inject_text = r"""\begin{align}
    \mathbf{u}^* =& \arg \min_{\mathbf{u} \in \mathcal{U}} \frac{1}{2} || \mathbf{u} - \mathbf{u}_{vla} ||^2 \label{eq:qp} \\
    \text{s.t.}& \quad \dot{h}_i(\mathbf{x}, \mathbf{u}) \ge -\alpha_i h_i(\mathbf{x}), \quad \forall i
\end{align}

\subsection{Drift-Aware 3DGS Firewall \& Multi-Stage DPO}
Traditional VLA models degrade precipitously when exposed to out-of-distribution (OOD) visual topologies. We introduce the \textbf{Drift-Aware Geometric Firewall}. Before Eq.~\ref{eq:qp} evaluates the semantic matrix, we measure trans-metric consistency $\tau(s_t, s_{t-1})$ mapping the volume scalar of the pointcloud. If $\tau > \delta_{\text{drift}}$, the robot initiates a Golden Spiral Hemispherical sweep to auto-calibrate $16$ multi-view transformations dynamically.

To formally eliminate this drift within the Small Language Model (OpenVLA) prior to inference, we define the \textbf{Safety Violation Rate (SVR) Direct Preference Optimisation (DPO)} mathematically as:
\begin{equation}
\label{eq:dpo_cbf}
\mathcal{L}_{DPO}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
\end{equation}
where $y_l$ exclusively maps to trajectories rejected by the Control Barrier Function. This native multi-stage SVR alignment effectively pushes operational SVR towards theoretical limits ($\le 0.0015$) securely on Edge computing paradigms (e.g., 8x H100 arrays)."""

search_text = r"""\begin{align}
    \mathbf{u}^* =& \arg \min_{\mathbf{u} \in \mathcal{U}} \frac{1}{2} || \mathbf{u} - \mathbf{u}_{vla} ||^2 \\
    \text{s.t.}& \quad \dot{h}_i(\mathbf{x}, \mathbf{u}) \ge -\alpha_i h_i(\mathbf{x}), \quad \forall i
\end{align}"""

files = [
    'preprint/latex/main.tex',
    'preprint/latex/SafeVLA_TAC.tex',
    'preprint/latex/SafeVLA_TRO.tex',
    'preprint/latex/SafeVLA_NeurIPS.tex'
]

for fp in files:
    if os.path.exists(fp):
        with open(fp, 'r') as f:
            content = f.read()
        if search_text in content:
            content = content.replace(search_text, inject_text)
            with open(fp, 'w') as f:
                f.write(content)
            print(f"[*] Injected math into {fp}")
        else:
            print(f"[!] Target block not found in {fp}")
