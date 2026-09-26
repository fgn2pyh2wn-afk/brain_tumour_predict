#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复算论文图6（a）–（c）：三例测试病例在时间网格 N = 150 上的进展模拟三联图。

图内标题为 "Step k"（时间网格第 k 个刻度，生成条件，非临床随访时间）。
直接复用 notebook 导出模块（默认 核验材料/P0_生成模块隔离/dit_nb.py），
固定随机种子以保证与论文数值一致、可复现。

用法：python3 图6_150刻度三联图.py [--cases Mets_020,Mets_049,Mets_042] [--seed 20260925]
输出：核验材料/论文配图/{病例}_step150.png、核验材料/复现_150刻度/{病例}_step150.png
"""
import argparse, json, random, re, time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODULE = ROOT / '核验材料/P0_生成模块隔离/dit_nb.py'
FIG_DIR = ROOT / '核验材料/论文配图'
RAW_DIR = ROOT / '核验材料/复现_150刻度'
# 与 notebook 推理单元一致：这些单元定义配置、数据、模型、生长律、推理与绘图
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
    ap.add_argument('--cases', default='Mets_020,Mets_049,Mets_042')
    ap.add_argument('--grid-steps', dest='grid_steps', type=int, default=150,
                    help='时间网格刻度数 N（生成条件，非临床随访时间）')
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--module', type=Path, default=MODULE)
    a = ap.parse_args()

    ns = load_namespace(a.module)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cases = {c.name: c for c in ns['_case_dirs'](ns['TEST_ROOT'])}
    summary = {}
    for name in [s.strip() for s in a.cases.split(',') if s.strip()]:
        t0 = time.time()
        sequence, metrics = ns['predict_tumor_progression'](
            cases[name], grid_steps=a.grid_steps, change_strength=0.35)
        blackout = ns['_blackout_tumor_growth'](sequence)
        fig = ns['plot_full_progression'](
            blackout, grid_steps=list(range(a.grid_steps + 1)),
            max_steps=min(8, a.grid_steps + 1), show_lesion_box=False)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for out in (FIG_DIR / f'{name}_step{a.grid_steps}.png', RAW_DIR / f'{name}_step{a.grid_steps}.png'):
            fig.savefig(out, dpi=100, bbox_inches='tight')  # 与论文内嵌图一致：紧边界裁剪
        plt.close(fig)
        summary[name] = {'seq_shape': list(sequence.shape),
                         'soft_iou': float(metrics['contour_iou']),
                         'texture_correlation': float(metrics['texture_correlation']),
                         'lesion_mean_change': float(metrics['lesion_mean_change']),
                         'seconds': round(time.time() - t0, 1)}
        print(f"{name}: 形状={tuple(sequence.shape)} softIoU={metrics['contour_iou']:.4f} "
              f"纹理相关性={metrics['texture_correlation']:.4f} "
              f"病灶区均值变化={metrics['lesion_mean_change']:+.4f} 耗时{time.time()-t0:.1f}s", flush=True)
    out_json = ROOT / '复算脚本/复算结果/图6_三联图汇总.json'
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('已保存汇总 ->', out_json)


if __name__ == '__main__':
    main()
