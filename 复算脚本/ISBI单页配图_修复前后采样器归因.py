# -*- coding: utf-8 -*-
"""ISBI 单页摘要 Fig. 4：修复前 / 修复后"采样器是否影响报告指标"的对照图。

数据来源（全部为代码实跑产物，本图不做任何再加工）：
  修复前：核验材料/p0_variants_full.csv                      （论文现有权重，51 例 × 7 替身）
  修复后：核验材料/VAE重训试点/试点51例_修复后指标_per_case.csv（pilot_ldm.pt，51 例 × 7 替身）

面板 (a)：五路"采样器侧"替身（真采样器 / 换种子 / 冻结采样 / 他例结果 / 常数潜变量）
          的逐例极差（max−min）在四项指标上的均值 —— 修复前恒为 0，修复后非零。
面板 (b)：修复后，相对真采样器的后处理输出差（候选核内，均值；须线为逐例最大值的均值）。

排版：图宽 = PDF 单栏 243.8 pt × 0.94 = 229.2 pt = 3.183 in，600 dpi，1:1 排入。
"""
import csv, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BEFORE = ROOT / '核验材料' / 'p0_variants_full.csv'
AFTER = ROOT / '核验材料' / 'VAE重训试点' / '试点51例_修复后指标_per_case.csv'
OUT = ROOT / '核验材料' / 'ISBI单页' / 'fig_sampler_attribution.png'

W_IN, H_IN = 3.183, 1.160          # = 229.2 pt × 0.94
SIDE = ['dit_seedA', 'dit_seedB', 'frozen', 'shuffled', 'mean_latent']
METRICS = [('soft_iou', 'soft\nIoU'), ('hard_iou', 'hard\nIoU'),
           ('tex_corr', 'texture\ncorr.'), ('lesion_change', 'mean\nchange')]
EN = ['seed B', 'frozen', 'other case', 'mean latent', 'source', 'noise']


def load(p):
    with open(p, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def spread(rows, key):
    by = collections.defaultdict(dict)
    for r in rows:
        by[r['case']][r['variant']] = float(r[key])
    return np.array([max(d[v] for v in SIDE) - min(d[v] for v in SIDE) for d in by.values()])


b = load(BEFORE); a = load(AFTER)
assert len({r['case'] for r in b}) == len({r['case'] for r in a}) == 51

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.linewidth': 0.5})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_IN, H_IN), dpi=600,
                               gridspec_kw=dict(width_ratios=[1.12, 1.0], wspace=0.62))
fig.subplots_adjust(left=0.115, right=0.968, top=0.945, bottom=0.300)

# ---------------- (a) 逐例极差：修复前 vs 修复后
vals_b = [spread(b, k).mean() for k, _ in METRICS]
vals_a = [spread(a, k).mean() for k, _ in METRICS]
x = np.arange(len(METRICS)); w = 0.36
ax1.bar(x - w / 2, vals_b, w, color='#c0392b', label='as trained (before fix)')
ax1.bar(x + w / 2, vals_a, w, color='#2f5d9e', label='after fix (pilot)')
for xi, v in zip(x - w / 2, vals_b):
    ax1.text(xi, v + 0.0006, '0.0000', ha='center', va='bottom', fontsize=3.5, color='#c0392b')
for xi, v in zip(x + w / 2, vals_a):
    ax1.text(xi, v + 0.0006, f'{v:.4f}', ha='center', va='bottom', fontsize=3.5, color='#2f5d9e')
ax1.set_xticks(x); ax1.set_xticklabels([n for _, n in METRICS], fontsize=3.6)
ax1.set_ylabel('per-case spread\n(5 sampler-side variants)', fontsize=3.6)
ax1.tick_params(axis='y', labelsize=3.5, length=1.3, width=0.5, pad=0.7)
ax1.set_ylim(0, max(vals_a) * 1.34)
ax1.legend(fontsize=3.3, frameon=False, loc='upper left', handlelength=1.1, borderpad=0.1)
ax1.grid(axis='y', color='#dfe4ea', linewidth=0.4); ax1.set_axisbelow(True)
for sp in ('top', 'right'):
    ax1.spines[sp].set_visible(False)

# ---------------- (b) 修复后：相对真采样器的核内差
rows = ['dit_seedB', 'frozen', 'shuffled', 'mean_latent', 'identity', 'noise']
mm = [np.mean([float(r['fin_diff_core_mean']) for r in a if r['variant'] == v]) for v in rows]
xx = [np.max([float(r['fin_diff_core_max']) for r in a if r['variant'] == v]) for v in rows]
y = np.arange(len(rows))
ax2.barh(y, mm, 0.55, color='#2f5d9e', xerr=[[0] * len(mm), [hi - lo for hi, lo in zip(xx, mm)]],
         error_kw=dict(ecolor='#8aa4c8', elinewidth=0.5, capsize=1.0), label='mean (bar) / max (whisker)')
ax2.set_yticks(y); ax2.set_yticklabels(EN, fontsize=3.5)
ax2.invert_yaxis()
ax2.set_xlabel('core |Δ| vs. the true sampler (after fix)', fontsize=3.7)
ax2.tick_params(axis='x', labelsize=3.5, length=1.3, width=0.5, pad=0.7)
ax2.set_xlim(0, max(xx) * 1.10)
ax2.grid(axis='x', color='#dfe4ea', linewidth=0.4); ax2.set_axisbelow(True)
for sp in ('top', 'right'):
    ax2.spines[sp].set_visible(False)

fig.canvas.draw(); rend = fig.canvas.get_renderer(); fw, fh = fig.canvas.get_width_height()
worst, who = 0.0, None
for ax in (ax1, ax2):
    xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
    shown_x = [t for loc, t in zip(ax.get_xticks(), ax.get_xticklabels()) if min(xlo, xhi) <= loc <= max(xlo, xhi)]
    shown_y = [t for loc, t in zip(ax.get_yticks(), ax.get_yticklabels()) if min(ylo, yhi) <= loc <= max(ylo, yhi)]
    for t in list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label] + shown_x + shown_y:
        if not t.get_text():
            continue
        bb = t.get_window_extent(renderer=rend)
        over = max(-bb.x0, -bb.y0, bb.x1 - fw, bb.y1 - fh)
        if over > worst:
            worst, who = over, t.get_text().replace('\n', ' | ')
if worst > 0.5:
    raise SystemExit(f'文字越出画布 {worst:.1f} px: {who!r}')
print(f'画布越界检查通过（最大越界 {worst:.2f} px）')
fig.savefig(OUT, dpi=600, facecolor='white')
print('→', OUT)
