"""Synchronize the requested paper/ entrypoints with canonical IEEE artifacts.
Run after regeneration and compilation; no model training or data mutation.
"""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
IEEE=ROOT/'ieee'
for folder,names in {
    'figures':['fig_arch.pdf','evaluation_protocol.pdf','fig_data.pdf','fig_forest.pdf'],
    'tables':['main_p9.tex','seed_variability.tex','protocols.tex','ablation_p3.tex','intervals.tex'],
}.items():
    (ROOT/folder).mkdir(exist_ok=True)
    for name in names:shutil.copy2(IEEE/folder/name,ROOT/folder/name)
shutil.copy2(IEEE/'refs.bib',ROOT/'refs.bib')
for name in ['main.pdf','supplement.pdf']:
    if (IEEE/name).exists():shutil.copy2(IEEE/name,ROOT/name)
print('Synchronized paper/ PDF, references, figures and tables; canonical sources remain in paper/ieee/.')
