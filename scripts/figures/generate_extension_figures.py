"""
Publication figures for the session's new results
=================================================
Reads verified JSONs and produces vector (PDF) + raster (PNG) figures:
  fig_cross_dataset.pdf   — zero-shot transfer Rank-1/EER on both external sets
  fig_snorm.pdf           — cross-domain EER: baseline vs AdaBN vs S-norm
  fig_faithfulness.pdf    — Fidelity+/- per explainer

Usage:  python scripts/figures/generate_extension_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.parent
STATS = ROOT / 'outputs' / 'stats'
OUT = ROOT / 'outputs' / 'figures' / 'extension'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 200})


def _load(name):
    p = STATS / name
    return json.load(open(p)) if p.exists() else {}


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{stem}.png', bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {stem}.pdf/.png')


def fig_cross_dataset():
    a = _load('cross_dataset_kaggle24.json'); b = _load('cross_dataset_pakistan459.json')
    if not a or not b:
        print('  [skip] cross-dataset jsons missing'); return
    sets = ['Set A\n(24 IDs)', 'Set B\n(308 IDs)']
    r1 = [a['closed_set']['rank1'] * 100, b['closed_set']['rank1'] * 100]
    eer = [a['closed_set']['eer'] * 100, b['closed_set']['eer'] * 100]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
    x = np.arange(2)
    ax[0].bar(x, r1, color='#2a9d8f', width=0.55)
    for i, v in enumerate(r1): ax[0].text(i, v + 1, f'{v:.1f}', ha='center')
    ax[0].set_ylim(0, 105); ax[0].set_xticks(x); ax[0].set_xticklabels(sets)
    ax[0].set_ylabel('Rank-1 (%)'); ax[0].set_title('Zero-shot identification')
    ax[1].bar(x, eer, color='#e76f51', width=0.55)
    for i, v in enumerate(eer): ax[1].text(i, v + 0.4, f'{v:.1f}', ha='center')
    ax[1].set_xticks(x); ax[1].set_xticklabels(sets)
    ax[1].set_ylabel('EER (%)'); ax[1].set_title('Zero-shot verification')
    fig.suptitle('Zero-shot cross-dataset transfer (no fine-tuning)', y=1.02, fontweight='bold')
    save(fig, 'fig_cross_dataset')


def fig_snorm():
    ka = _load('cross_dataset_kaggle24.json'); kb = _load('cross_dataset_pakistan459.json')
    ka_ad = _load('cross_dataset_kaggle24_adabn.json'); kb_ad = _load('cross_dataset_pakistan459_adabn.json')
    ka_sn = _load('cross_dataset_kaggle24_snorm.json'); kb_sn = _load('cross_dataset_pakistan459_snorm.json')
    if not (ka and kb and ka_sn and kb_sn):
        print('  [skip] snorm jsons missing'); return
    methods = ['Baseline', '+AdaBN', '+S-norm']
    setA = [ka['closed_set']['eer'] * 100, ka_ad.get('closed_set', {}).get('eer', 0) * 100, ka_sn['closed_set']['eer'] * 100]
    setB = [kb['closed_set']['eer'] * 100, kb_ad.get('closed_set', {}).get('eer', 0) * 100, kb_sn['closed_set']['eer'] * 100]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(x - w / 2, setA, w, label='Set A (24 IDs)', color='#264653')
    ax.bar(x + w / 2, setB, w, label='Set B (308 IDs)', color='#e9c46a')
    for i, v in enumerate(setA): ax.text(i - w / 2, v + 0.2, f'{v:.1f}', ha='center', fontsize=9)
    for i, v in enumerate(setB): ax.text(i + w / 2, v + 0.2, f'{v:.1f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylabel('EER (%) — lower is better')
    ax.set_title('Label-free test-time calibration recovers\ncross-domain verification', fontweight='bold')
    ax.legend()
    save(fig, 'fig_snorm')


def fig_faithfulness():
    f = _load('explainability_faithfulness.json')
    methods = f.get('methods', {})
    if not methods:
        print('  [skip] faithfulness json missing'); return
    names = list(methods.keys())
    fp = [methods[m]['fidelity_plus']['mean'] for m in names]
    fm = [methods[m]['fidelity_minus']['mean'] for m in names]
    x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(x - w / 2, fp, w, label='Fidelity+ (higher=better)', color='#2a9d8f')
    ax.bar(x + w / 2, fm, w, label='Fidelity- (lower=better)', color='#e76f51')
    ax.set_xticks(x); ax.set_xticklabels([n.replace('_', '\n') for n in names])
    ax.set_ylabel('Predicted-class prob. change')
    ax.set_title('Explanation faithfulness (GNN)', fontweight='bold')
    ax.legend(fontsize=9)
    save(fig, 'fig_faithfulness')


if __name__ == '__main__':
    print('Generating extension figures ->', OUT)
    fig_cross_dataset(); fig_snorm(); fig_faithfulness()
    print('Done.')
