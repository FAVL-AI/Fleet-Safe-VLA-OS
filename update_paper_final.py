import re

with open("Safe-VLA-Paper/main.tex", "r") as f:
    content = f.read()

# 1. Sim-to-Twin in Abstract
content = content.replace(
    r"This framework provides provable safety guarantees via CBF-QP whilst demonstrating cross-embodiment generalisation from wheeled to bipedal platforms.",
    r"All experiments are conducted in a high-fidelity simulation-to-twin (Sim-to-Twin) digital twin, with controlled perturbations to evaluate robustness under realistic deployment conditions. This framework provides provable safety guarantees via CBF-QP whilst demonstrating cross-embodiment generalisation from wheeled to bipedal platforms."
)

# 2. Modify FastBot to Yahboom (preserve casing where possible, just a straight replace)
content = content.replace("FastBot", "Yahboom")
content = content.replace("FastBots", "Yahbooms")

# 3. Add Figure Reference in Text
main_results = r"""Table~\ref{tab:main} summarises our main results.
FLEET-Safe VLA achieves the lowest safety cost (0.0007), lowest SVR ($5\times10^{-5}$), highest reward (0.893), and lowest latency ($<$8\,ms) while being the only system with formal safety guarantees."""

new_main_results = r"""Table~\ref{tab:main} summarises our main results.
FLEET-Safe VLA achieves the lowest safety cost (0.0007), lowest SVR ($5\times10^{-5}$), highest reward (0.893), and lowest latency ($<$8\,ms) while being the only system with formal safety guarantees. As shown in Figure~\ref{fig:robustness}, FleetSafe-VLA maintains low SVR under increasing latency."""
content = content.replace(main_results, new_main_results)

# 4. Modify Experiments Text
eval_text = r"""We evaluate in a hospital digital twin built in NVIDIA Isaac Sim~\cite{isaacgym} with 12 distinct zones (lobby, corridor, ward, ICU, pharmacy, lift, stairwell, consultation room, reception, staff room, operating theatre, emergency department)."""
new_eval_text = r"""We evaluate in a high-fidelity digital twin, and stress-test it with controlled perturbations. The hospital digital twin encompasses 12 distinct zones (lobby, corridor, ward, ICU, pharmacy, lift, stairwell, consultation room, reception, staff room, operating theatre, emergency department)."""
content = content.replace(eval_text, new_eval_text)

# 5. Fix Multi-Panel Figure
old_figs = r"""\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/latency.png}
\caption{Latency distributions demonstrating the strict under \SI{8}{\milli\second} preemption capability of DSEO compared to standard VLA baselines.}
\label{fig:latency}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/trajectory.png}
\caption{Safe vs. Unsafe trajectories during evaluation. FLEET-Safe VLA strictly adheres to the CBF safe set avoiding semantic zone violations.}
\label{fig:trajectory}
\end{figure}"""

new_figs = r"""\begin{figure}[t]
\centering
\includegraphics[width=0.32\linewidth]{figures/svr_latency.png}
\includegraphics[width=0.32\linewidth]{figures/dmr_noise.png}
\includegraphics[width=0.32\linewidth]{figures/ttp_latency.png}
\caption{
\textbf{Robustness under latency and noise.}
(Left) Safety Violation Rate (SVR) as latency increases.
(Middle) Deadline Miss Rate (DMR) under sensor noise.
(Right) Time-To-Preempt (TTP) demonstrating proactive safety behavior.
FleetSafe-VLA maintains stability under perturbations, whereas static baselines degrade rapidly.
}
\label{fig:robustness}
\end{figure}"""
content = content.replace(old_figs, new_figs)

with open("Safe-VLA-Paper/main.tex", "w") as f:
    f.write(content)

print("Text replacements completed successfully.")
