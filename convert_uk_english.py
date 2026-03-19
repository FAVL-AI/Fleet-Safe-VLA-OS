import os
import re

replacements = {
    r'\boptimization\b': 'optimisation',
    r'\bOptimization\b': 'Optimisation',
    r'\boptimize\b': 'optimise',
    r'\bOptimize\b': 'Optimise',
    r'\boptimized\b': 'optimised',
    r'\bOptimized\b': 'Optimised',
    r'\boptimizing\b': 'optimising',
    r'\bOptimizing\b': 'Optimising',
    r'\bmodeling\b': 'modelling',
    r'\bModeling\b': 'Modelling',
    r'\bmodeled\b': 'modelled',
    r'\bModeled\b': 'Modelled',
    r'\banalyze\b': 'analyse',
    r'\bAnalyze\b': 'Analyse',
    r'\banalyzed\b': 'analysed',
    r'\banalyzing\b': 'analysing',
    r'\bbehavior\b': 'behaviour',
    r'\bBehavior\b': 'Behaviour',
    r'\bbehaviors\b': 'behaviours',
    r'\bcolor\b': 'colour',
    r'\bColor\b': 'Colour',
    r'\blabeled\b': 'labelled',
    r'\bLabeled\b': 'Labelled',
    r'\blabeling\b': 'labelling',
    r'\bgeneralization\b': 'generalisation',
    r'\bGeneralization\b': 'Generalisation',
    r'\bgeneralize\b': 'generalise',
    r'\bparameterized\b': 'parameterised',
    r'\bparameterize\b': 'parameterise',
    r'\bstandardization\b': 'standardisation',
    r'\bsynchronization\b': 'synchronisation',
    r'\bSynchronization\b': 'Synchronisation',
    r'\brecognize\b': 'recognise',
    r'\brecognized\b': 'recognised',
    r'\brealize\b': 'realise',
    r'\bvisualize\b': 'visualise',
    r'\bVisualization\b': 'Visualisation',
    r'\bvisualization\b': 'visualisation',
    r'\bminimize\b': 'minimise',
    r'\bMinimize\b': 'Minimise',
    r'\bmaximize\b': 'maximise',
    r'\bMaximize\b': 'Maximise',
    r'\binferences\b': 'inferences',
    r'\breward\b': 'reward',
    r'\bprogrammatic\b': 'programmatic'
}

tex_files = [
    'Safe-VLA-Paper/main.tex',
    'preprint/latex/SafeVLA_TAC.tex',
    'preprint/latex/SafeVLA_TRO.tex',
    'preprint/latex/main.tex'
]

for fp in tex_files:
    if os.path.exists(fp):
        with open(fp, 'r') as f:
            content = f.read()
        for us, uk in replacements.items():
            content = re.sub(us, uk, content)
        with open(fp, 'w') as f:
            f.write(content)
        print(f"[*] Standardised UK English syntax mapped for: {fp}")
