"""
Severity-vs-Rank-1 and severity-vs-EER curves for the quality-aware fusion study
(Part 1 of the extension plan). Reads outputs/stats/quality_fusion_results.json.
Outputs outputs/figures/extension/fig_corruption_{rank1,eer}.(png|pdf)
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
J = ROOT / 'outputs' / 'stats' / 'quality_fusion_results.json'
OUT = ROOT / 'outputs' / 'figures' / 'extension'
OUT.mkdir(parents=True, exist_ok=True)

res = json.load(open(J))
CORR = ['blur', 'brightness', 'spatter']
SEV = [0, 1, 3, 5]
METHODS = [('cnn', 'CNN', '#264653', 'o'), ('hybrid', 'Hybrid', '#2a9d8f', 's'),
           ('val_tuned_alpha', 'Fixed-α fusion', '#e9c46a', '^'),
           ('quality_learned', 'Quality-aware fusion', '#e76f51', 'D')]


def val(tag, method, metric):
    r = res.get(tag, {}).get(method, {})
    return r.get(metric) if isinstance(r, dict) and metric in r else None


def plot(metric, ylabel, stem, lower_better):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), dpi=200, sharey=True)
    for ax, c in zip(axes, CORR):
        for key, name, col, mk in METHODS:
            ys = []
            for s in SEV:
                tag = 'clean' if s == 0 else f'{c}_s{s}'
                v = val(tag, key, metric)
                ys.append(v * 100 if v is not None else None)
            xs = [s for s, y in zip(SEV, ys) if y is not None]
            yy = [y for y in ys if y is not None]
            if yy:
                ax.plot(xs, yy, marker=mk, color=col, label=name, lw=1.8, ms=6)
        ax.set_title(c, fontsize=11); ax.set_xlabel('severity'); ax.set_xticks(SEV)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(fontsize=8, loc='best')
    fig.suptitle(f'Corruption robustness: {ylabel} vs severity'
                 + (' (lower better)' if lower_better else ''), y=1.03, fontweight='bold')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{stem}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('saved', stem)


if __name__ == '__main__':
    plot('rank1', 'Rank-1 (%)', 'fig_corruption_rank1', False)
    plot('eer', 'EER (%)', 'fig_corruption_eer', True)
