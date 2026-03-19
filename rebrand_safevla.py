import os
import re

fp = "Safe-VLA-Paper/main.tex"

if os.path.exists(fp):
    with open(fp, 'r') as f:
        content = f.read()

    # Title adjustment
    content = content.replace(r"\title{FLEET-Safe VLA: A Layered Safety Architecture for \\", r"\title{SafeVLA: A Cross-Embodiment Safety Architecture for \\")
    content = content.replace(r"Vision-Language-Action Robot Fleets in Hospital Environments}", r"Vision-Language-Action Robots in Hospital Environments}")
    
    # Core system re-branding
    content = content.replace("FLEET-Safe VLA", "SafeVLA")
    content = content.replace("FLEET-Safe", "SafeVLA")
    
    # Adjust sentence mentioning fleet limitations
    content = content.replace("While FLEET targets multi-robot fleets, current results are single-agent. Fleet-level coordination with shared safety envelopes is future work.", "While later stages will target multi-robot environments, current results evaluate strict cross-embodiment hardware generalisation.")
    
    # "robot fleets" to "robots"
    content = content.replace("robot fleets", "robots")
    
    # Replace any straggling uppercase FLEET where SafeVLA is more appropriate
    content = re.sub(r'\bFLEET\b', 'SafeVLA', content)

    with open(fp, 'w') as f:
        f.write(content)
        
    print("[*] Successfully stripped FLEET branding for Cross-Embodiment Scope.")
