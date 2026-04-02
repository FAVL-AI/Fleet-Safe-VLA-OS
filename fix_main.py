import re
with open("Safe-VLA-Paper/main.tex", "r") as f:
    text = f.read()

# Fix the broken architecture line (line 169 originally)
text = text.replace(
r"""\begin{figure}[t]
\centering
\includegraphics[width=0.32\linewidth]{figures/svr_latency.png}
\includegraphics[width=0.32\linewidth]{figures/dmr_noise.png}
\includegraphics[width=0.32\linewidth]{figures/ttp_latency.png}
\caption{
\textbf{FleetSafe-VLA Architecture.}""",
r"""\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{figures/architecture.png}
\caption{
\textbf{FleetSafe-VLA Architecture.}""")

# Fix the plots block (line ~437)
text = text.replace(
r"""\includegraphics[width=0.32\linewidth]{svr_latency.png}
\includegraphics[width=0.32\linewidth]{dmr_noise.png}
\includegraphics[width=0.32\linewidth]{ttp_latency.png}""",
r"""\includegraphics[width=0.32\linewidth]{figures/svr_latency.png}
\includegraphics[width=0.32\linewidth]{figures/dmr_noise.png}
\includegraphics[width=0.32\linewidth]{figures/ttp_latency.png}""")

with open("Safe-VLA-Paper/main.tex", "w") as f:
    f.write(text)
