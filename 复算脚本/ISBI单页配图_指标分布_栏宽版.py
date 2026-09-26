#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISBI 单页摘要用配图：51 例进展模拟四项指标的 2×2 分布图。

数据来源：复算结果/表3_进展模拟_per_case.csv（与论文表3、图7同一份逐例数据）。
输出：核验材料/ISBI单页/fig_metrics_2x2_small.png

排版约定：图宽 = PDF 单栏满宽 243.8 pt = 3.386 in（×1.00，1:1 不缩放），600 dpi；
字号同步放大 243.8/170.7 = 1.428 倍，保证纸上字号不小于旧版；
出图后**逐条检查文字包围盒是否越出画布**（此前版本 "sd=0.0035" 被裁掉过），
越界即报错，避免把裁字的图直接排进 PDF。
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

W_IN, H_IN = 3.386, 1.596          # = 243.8 pt × 1.00，与 PDF 中实际显示尺寸一致（保持原长宽比）
FS = 1.428                         # 字号放大系数

with open(SRC, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
spec = [('soft_contour_iou', 'Soft contour IoU'), ('hard_contour_iou', 'Hard contour IoU'),
        ('texture_corr', 'High-freq. texture corr.'), ('lesion_mean_change', 'Lesion mean change')]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.linewidth': 0.5})
fig, axes = plt.subplots(2, 2, figsize=(W_IN, H_IN), dpi=600)
for ax, (key, name) in zip(axes.ravel(), spec):
    v = np.array([float(r[key]) for r in rows])
    ax.boxplot([v], widths=0.42, patch_artist=True, showfliers=True,
               medianprops=dict(color='#c0392b', linewidth=0.7),
               flierprops=dict(marker='o', markersize=1.2*FS, markerfacecolor='none',
                               markeredgecolor='#4a5568', markeredgewidth=1.0),
               boxprops=dict(facecolor='#dbe7f7', edgecolor='#2f5d9e', linewidth=0.5),
               whiskerprops=dict(color='#2f5d9e', linewidth=0.5),
               capprops=dict(color='#2f5d9e', linewidth=0.5))
    ax.scatter(np.random.RandomState(20260925).normal(1, 0.045, v.size), v, s=0.9*FS*FS,
               color='#12243d', alpha=0.55, linewidths=0)
    ax.set_title(f'{name}\nmean={v.mean():.4f}, sd={v.std(ddof=1):.4f}',
                 fontsize=4.4*FS, color='#1B3A6B', pad=1.2*FS)
    ax.set_xticks([]); ax.set_xlim(0.72, 1.28)
    ax.tick_params(axis='y', labelsize=4.2*FS, length=1.2*FS, width=0.7, pad=0.8*FS)
    ax.grid(axis='y', color='#dfe4ea', linewidth=0.4)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.16*FS, w_pad=0.7*FS, h_pad=0.45*FS)

fig.canvas.draw()
rend = fig.canvas.get_renderer()
fw, fh = fig.canvas.get_width_height()
worst, who = 0.0, None
for ax in axes.ravel():
    lo, hi = ax.get_ylim()
    shown = [t for loc, t in zip(ax.get_yticks(), ax.get_yticklabels()) if lo <= loc <= hi]
    for t in list(ax.texts) + [ax.title] + shown:
        bb = t.get_window_extent(renderer=rend)
        over = max(-bb.x0, -bb.y0, bb.x1 - fw, bb.y1 - fh)
        if over > worst:
            worst, who = over, t.get_text().replace('\n', ' | ')
if worst > 0.5:
    raise SystemExit(f'文字越出画布 {worst:.1f} px: {who!r}')
print(f'画布越界检查通过（最大越界 {worst:.2f} px）')

out = OUT / 'fig_metrics_2x2_small.png'
fig.savefig(out, dpi=600, facecolor='white')
print('→', out)
