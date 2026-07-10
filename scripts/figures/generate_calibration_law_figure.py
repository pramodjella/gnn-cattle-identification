"""
The paper's central figure: S-norm EER recovery vs. baseline EER (mis-calibration).
Visualizes the conditional law — recovery grows with baseline mis-calibration, and
significant recoveries occur only in the mis-calibrated (large-EER) regime.

Data points are the measured (baseline EER, +S-norm EER) pairs from the study
(scripts/wildlife_probe.py, wildlife_natural_shift.py, calibration_bakeoff.py,
evaluate_cross_dataset.py). Outputs outputs/figures/extension/fig_calibration_law.(png|pdf)
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent.parent.parent / 'outputs' / 'figures' / 'extension'
OUT.mkdir(parents=True, exist_ok=True)

# (label, backbone, baseline_EER%, snorm_EER%, significant)
D = [
    ("cattle X-dataset B", "CNN",  12.2, 7.9,  True),
    ("cattle X-dataset A", "CNN",  14.8, 11.4, True),
    ("macaque corrupt",    "Mega", 6.24, 5.34, None),
    ("Friesian corrupt",   "Mega", 2.77, 1.15, None),
    ("panda natural",      "Mega", 2.63, 2.29, False),
    ("macaque natural",    "Mega", 0.08, 0.08, False),
    ("panda natural",      "DINOv2", 40.8, 35.3, True),
    ("macaque natural",    "DINOv2", 29.2, 20.7, True),
    ("chimp corrupt",      "DINOv2", 35.75, 22.69, True),
    ("chimp clean",        "DINOv2", 27.54, 16.87, True),
    ("panda corrupt",      "DINOv2", 42.84, 38.00, None),
    ("macaque corrupt",    "DINOv2", 32.39, 18.24, True),
    ("Friesian corrupt",   "DINOv2", 39.02, 31.58, None),
]

COL = {"CNN": "#2a9d8f", "Mega": "#264653", "DINOv2": "#e76f51"}
MK = {True: "o", False: "s", None: "^"}  # significant / n.s. / not-CI-tested

fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=220)
for lab, bb, base, sn, sig in D:
    rec = base - sn
    ax.scatter(base, rec, s=85, c=COL[bb], marker=MK[sig], edgecolor='white',
               linewidth=0.8, zorder=3, alpha=0.9)

# trend line (recovery ~ baseline)
b = np.array([d[2] for d in D]); r = np.array([d[2]-d[3] for d in D])
z = np.polyfit(b, r, 1); xs = np.linspace(0, 44, 50)
ax.plot(xs, np.polyval(z, xs), '--', color='#888', lw=1.3, zorder=1,
        label=f'trend (slope={z[0]:.2f})')

ax.axhline(0, color='#ccc', lw=0.8)
ax.set_xlabel('Baseline EER (%) — mis-calibration', fontsize=11)
ax.set_ylabel('S-norm EER recovery (pts)', fontsize=11)
ax.set_title('Calibration recovery scales with baseline mis-calibration', fontsize=11, fontweight='bold')

# legends: backbone colours + significance markers
from matplotlib.lines import Line2D
bb_leg = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=9, label=k)
          for k, c in COL.items()]
sig_leg = [Line2D([0],[0], marker=MK[True], color='#555', markersize=9, label='significant', linestyle=''),
           Line2D([0],[0], marker=MK[False], color='#555', markersize=9, label='n.s. / no gap', linestyle=''),
           Line2D([0],[0], marker=MK[None], color='#555', markersize=9, label='not CI-tested', linestyle='')]
l1 = ax.legend(handles=bb_leg, title='backbone', loc='upper left', fontsize=8, title_fontsize=8)
ax.add_artist(l1)
ax.legend(handles=sig_leg, loc='lower right', fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'fig_calibration_law.{ext}', bbox_inches='tight')
plt.close(fig)
print('saved', OUT / 'fig_calibration_law.png')
