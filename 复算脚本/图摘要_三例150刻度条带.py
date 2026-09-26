#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复算论文首页摘要配图：三例测试病例（Mets_020 / Mets_049 / Mets_042）在时间网格 N = 150 上的
轴向切面条带。每行一例，8 个刻度（Step 0/21/42/64/85/107/128/150），图内标题为 "Step k"
（时间网格第 k 个刻度，生成条件，非临床随访时间）。

复用 notebook 导出模块，固定随机种子，保证与图6、表3 的数值同一来源。

用法：python3 图摘要_三例150刻度条带.py [--seed 20260925]
输出：核验材料/论文配图/新摘要配图_三例150刻度_轴向.png
"""
import argparse, random, re, time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODULE = ROOT / '核验材料/P0_生成模块隔离/dit_nb.py'
OUT = ROOT / '核验材料/论文配图/新摘要配图_三例150刻度_轴向.png'
CASES = ['Mets_020', 'Mets_049', 'Mets_042']
CELLS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13]


def load_namespace(module_path=MODULE):
    text = module_path.read_text(encoding='utf-8')
    toks = re.split(r'# ===== CELL (\d+) =====\n', text)
    cells = {int(toks[i]): toks[i + 1] for i in range(1, len(toks) - 1, 2)}
    ns = {'__name__': 'notebook_module'}
    for n in CELLS:
        exec(compile(cells[n], f'<cell{n}>', 'exec'), ns)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--grid-steps', dest='grid_steps', type=int, default=150,
                    help='时间网格刻度数 N（生成条件，非临床随访时间）')
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--module', type=Path, default=MODULE)
    a = ap.parse_args()

    ns = load_namespace(a.module)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    cases = {c.name: c for c in ns['_case_dirs'](ns['TEST_ROOT'])}
    sel = np.linspace(0, a.grid_steps, min(a.steps, a.grid_steps + 1), dtype=int)
    frames = {}
    for name in CASES:
        t0 = time.time()
        sequence, metrics = ns['predict_tumor_progression'](
            cases[name], grid_steps=a.grid_steps, change_strength=0.35)
        displayed = ns['_normalize_for_display'](ns['_blackout_tumor_growth'](sequence))
        frames[name] = [ns['_get_three_plane_slices'](displayed[step, 0])[0] for step in sel]
        print(f"{name}: softIoU={metrics['contour_iou']:.4f} 纹理相关性={metrics['texture_correlation']:.4f} "
              f"病灶区均值变化={metrics['lesion_mean_change']:+.4f} 耗时{time.time()-t0:.1f}s", flush=True)

    # 画布与面板几何：与论文内嵌图一致（dpi=100 -> 2717 x 901 px，面板 261 x 260 px，列距 317 px）
    dpi = 100
    canvas_px = (2717, 901)
    panel_px = (261, 260)
    col_x0_px, col_pitch_px = 235, 317
    row_top_px = [25, 330, 635]
    fig = plt.figure(figsize=(canvas_px[0] / dpi, canvas_px[1] / dpi), dpi=dpi)
    for row, name in enumerate(CASES):
        top = row_top_px[row]
        bottom = canvas_px[1] - (top + panel_px[1])
        for column, step in enumerate(sel):
            left = col_x0_px + column * col_pitch_px
            ax = fig.add_axes([left / canvas_px[0], bottom / canvas_px[1],
                               panel_px[0] / canvas_px[0], panel_px[1] / canvas_px[1]])
            ax.imshow(frames[name][column], cmap='gray', vmin=0, vmax=1, aspect='auto')
            ax.axis('off')
            ax.set_title(f'Step {int(step)}', fontsize=12)
        fig.text(213 / canvas_px[0], (canvas_px[1] - (top + panel_px[1] / 2)) / canvas_px[1], name,
                 fontsize=31, ha='right', va='center')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=dpi, facecolor='white')
    plt.close(fig)
    from PIL import Image
    print(f'已保存 {OUT} 尺寸={Image.open(OUT).size}')


if __name__ == '__main__':
    main()
