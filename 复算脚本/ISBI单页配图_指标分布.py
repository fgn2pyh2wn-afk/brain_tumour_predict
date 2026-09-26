#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISBI 单页摘要用配图：51 例进展模拟四项指标的 2×2 分布图。

数据来源：复算结果/表3_进展模拟_per_case.csv（与论文表3、图7同一份逐例数据）。
输出：核验材料/ISBI单页/fig_metrics_2x2.png
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / '复算结果' / '表3_进展模拟_per_case.csv'
OUT = ROOT / '核验材料' / 'ISBI单页'
OUT.mkdir(parents=True, exist_ok=True)

with open(SRC, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
spec = [('soft_contour_iou', 'Soft contour IoU'), ('hard_contour_iou', 'Hard contour IoU'),
        ('texture_corr', 'High-freq. texture corr.'), ('lesion_mean_change', 'Lesion mean change')]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.linewidth': 0.9})
fig, axes = plt.subplots(2, 2, figsize=(5.2, 2.45), dpi=300)
for ax, (key, name) in zip(axes.ravel(), spec):
    v = np.array([float(r[key]) for r in rows])
    bp = ax.boxplot([v], widths=0.42, patch_artist=True, showfliers=True,
                    medianprops=dict(color='#c0392b', linewidth=1.2),
                    flierprops=dict(marker='o', markersize=2.4, markerfacecolor='none',
                                    markeredgecolor='#4a5568', markeredgewidth=0.7),
                    boxprops=dict(facecolor='#dbe7f7', edgecolor='#2f5d9e', linewidth=0.9),
                    whiskerprops=dict(color='#2f5d9e', linewidth=0.9),
                    capprops=dict(color='#2f5d9e', linewidth=0.9))
    ax.scatter(np.random.RandomState(20260925).normal(1, 0.045, v.size), v, s=2.6,
               color='#12243d', alpha=0.55, linewidths=0)
    ax.set_title(f'{name}\nmean={v.mean():.4f}, sd={v.std(ddof=1):.4f}', fontsize=8.4, pad=2.5)
    ax.set_xticks([]); ax.set_xlim(0.72, 1.28)
    ax.tick_params(axis='y', labelsize=7.4, length=2, width=0.8, pad=1.5)
    ax.grid(axis='y', color='#dfe4ea', linewidth=0.7)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.35, w_pad=1.2, h_pad=0.9)
out = OUT / 'fig_metrics_2x2.png'
fig.savefig(out, dpi=300, facecolor='white')
print('→', out)
