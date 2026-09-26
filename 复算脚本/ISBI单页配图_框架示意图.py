#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISBI 单页摘要配图：两阶段潜空间扩散总体框架示意（对应论文图1）。

与论文图1同构：输入体数据 → 3D VAE 编码 → 潜变量 z0 → 自适应噪声注入(AdaNI t_start)
→ 条件 DiT 去噪 → 第二阶段低噪声细化(0.15T) → 3D VAE 解码 → 结构保持后处理
→ 第 0 至第 N 刻度演化序列；条件分支 c = pool(z0)。

**尺寸约定**：按 PDF 中的最终物理尺寸出图——单栏满宽 243.8 pt = 3.386 in。
论文原图是 13.2 in 宽的画布，直接缩到 3.386 in 会让框内文字只剩约 2.3 pt（不可读），
故本脚本按目标宽度重排为 2 行 × 4 框的蛇形流（第二行自右向左），字体按最终尺寸取 4.4 pt。
序列标注统一用 "Step 0 … Step N"，与正文口径一致（不使用 day）。

输出：核验材料/ISBI单页/fig_framework.png（3.386 in 宽 × 1.250 in 高，600 dpi）
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / '核验材料' / 'ISBI单页' / 'fig_framework.png'
OUT.parent.mkdir(parents=True, exist_ok=True)

W_PT, H_PT = 243.8, 78.0            # = 单栏满宽 × 高度；排入 PDF 后不再缩放
BW, BH = 54.45, 22.0                # 框宽 / 框高
GX = [4.0, 64.45, 124.9, 185.35]    # 四个框的 x 起点
ROW_A, ROW_B = 56.0, 26.0           # 上排 / 下排的 y
OUTY, OUTH = 2.0, 16.0

BLUE, EDGE, INK = '#eaf1fb', '#2f5d9e', '#12243d'
CREAM, CEDGE = '#fdf6e6', '#b08a3e'
GREEN, GEDGE = '#eaf7ee', '#2e7d4f'

plt.rcParams['font.family'] = 'DejaVu Sans'
fig = plt.figure(figsize=(W_PT / 72.0, H_PT / 72.0), dpi=600)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_PT); ax.set_ylim(0, H_PT); ax.axis('off')


def box(x, y, w, h, text, fc=BLUE, ec=EDGE, fs=4.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.6',
                                linewidth=0.55, facecolor=fc, edgecolor=ec))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            color=INK, linespacing=1.35)


def arrow(p, q, color=EDGE, lw=0.6, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=4.0, linewidth=lw,
                                 color=color, shrinkA=0.6, shrinkB=0.6,
                                 connectionstyle=f'arc3,rad={rad}'))


# 上排：左 → 右
for x, t in zip(GX, ['Input volume\n1×64×64×64', '3D VAE\nencoder',
                     'Latent $z_0$\n4×16×16×16', 'Noise injection\n(AdaNI: $t_{start}$)']):
    box(x, ROW_A, BW, BH, t)
# 下排：右 → 左（蛇形），使条件 DiT 正好落在噪声注入下方
for x, t in zip(GX, ['Structure-\npreserving post', '3D VAE\ndecoder',
                     'Stage-2 refine\n(low noise 0.15$T$)',
                     'Conditional\nDiT $\\epsilon_\\theta$\n(denoise $t-1\\to 0$)']):
    box(x, ROW_B, BW, BH, t)

# 上排内箭头
for i in range(3):
    arrow((GX[i] + BW, ROW_A + BH / 2), (GX[i + 1], ROW_A + BH / 2))
# 换行：噪声注入 → 条件 DiT（竖直向下）
arrow((GX[3] + BW / 2, ROW_A), (GX[3] + BW / 2, ROW_B + BH))
# 下排内箭头（右 → 左）
for i in range(3, 0, -1):
    arrow((GX[i], ROW_B + BH / 2), (GX[i - 1] + BW, ROW_B + BH / 2))
# 条件分支 c = pool(z0)：潜变量 → 条件 DiT
arrow((GX[2] + BW * 0.70, ROW_A), (GX[3] + BW * 0.20, ROW_B + BH), color=CEDGE, rad=-0.2)
ax.text(GX[2] + BW * 0.55, ROW_A - 5.2, 'cond  $c=\\mathrm{pool}(z_0)$',
        ha='right', va='center', fontsize=3.8, color=CEDGE)
# 输出
box(4.0, OUTY, 235.8, OUTH, 'Step 0 … Step N   evolution sequence', fc=GREEN, ec=GEDGE, fs=5.0)
arrow((GX[0] + BW / 2, ROW_B), (GX[0] + BW / 2, OUTY + OUTH), color=GEDGE)

fig.canvas.draw()
rend = fig.canvas.get_renderer()
fw, fh = fig.canvas.get_width_height()
worst, who = 0.0, None
for t in list(ax.texts) + list(ax.patches):
    bb = t.get_window_extent(renderer=rend)
    over = max(-bb.x0, -bb.y0, bb.x1 - fw, bb.y1 - fh)
    if over > worst:
        worst, who = over, getattr(t, 'get_text', lambda: type(t).__name__)()
if worst > 0.5:
    raise SystemExit(f'元素越出画布 {worst:.1f} px: {who!r}')
print(f'画布越界检查通过（最大越界 {worst:.2f} px）；物理尺寸 {W_PT/72:.3f}×{H_PT/72:.3f} in')
fig.savefig(OUT, dpi=600, facecolor='white')
print('→', OUT)
