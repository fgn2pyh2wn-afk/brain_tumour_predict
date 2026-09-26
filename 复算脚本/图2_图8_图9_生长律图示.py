#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复算论文图2、图8、图9（测试病例 Mets_001，时间网格 N = 100，生成条件非临床随访时间）。

图2  生长律参数化时间权重曲线 w(t)，t 为时间网格刻度（Step t）
图8  时间网格第50个刻度的中间帧及其相对线性权重的有符号差异（同一组端点、同一轴向切片）
图9  各生长律对应的病灶负荷轨迹（由命题1的仿射关系与各生长律权重给出）

复用 notebook 导出模块与固定随机种子，保证与表3/表4及正文数值同源。

用法：python3 图2_图8_图9_生长律图示.py [--case Mets_001] [--grid-steps 100] [--seed 20260925]
输出：核验材料/论文配图/{图2_生长律权重曲线,图8_生长律中间帧与差异,图9_病灶负荷轨迹}.png
"""
import argparse, contextlib, io, random, re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODULE = ROOT / '核验材料/P0_生成模块隔离/dit_nb.py'
OUT_DIR = ROOT / '核验材料/论文配图'
CELLS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13]
LAWS = [('linear', 'volume', 'linear'),
        ('exponential', 'volume', 'exponential-volume'),
        ('exponential', 'radius', 'exponential-radius'),
        ('logistic_fkpp', 'volume', 'logistic_fkpp-volume')]
# 图9 的列/图例顺序与图2、图8 不同（原图如此）：linear, exponential-volume, logistic_fkpp-volume, exponential-radius
BURDEN_ORDER = ['linear', 'exponential-volume', 'logistic_fkpp-volume', 'exponential-radius']


def load_namespace(module_path=MODULE):
    text = module_path.read_text(encoding='utf-8')
    toks = re.split(r'# ===== CELL (\d+) =====\n', text)
    cells = {int(toks[i]): toks[i + 1] for i in range(1, len(toks) - 1, 2)}
    ns = {'__name__': 'notebook_module'}
    for n in CELLS:
        exec(compile(cells[n], f'<cell{n}>', 'exec'), ns)
    return ns


def figure_weights(ns, weights, out):
    """图2：权重曲线。"""
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    steps = np.arange(len(weights[LAWS[0][2]]))
    for _, _, label in LAWS:
        ax.plot(steps, weights[label], label=label)
    ax.set_title('Growth-law-parameterized temporal weights (Mets_001)')
    ax.set_xlabel('Time-grid step t')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figure_burden(ns, burden, out):
    """图9：病灶负荷轨迹。"""
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    steps = np.arange(len(burden[LAWS[0][2]]))
    for label in BURDEN_ORDER:
        ax.plot(steps, burden[label], label=label)
    ax.set_title('Lesion-burden trajectory implied by each growth law (Mets_001)')
    ax.set_xlabel('Time-grid step t')
    ax.set_ylabel('Lesion burden of the blended frame')
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def figure_midframe(frames, devs, out, mid_step):
    """图8：第 mid_step 个刻度的中间帧与相对线性权重的有符号差异。"""
    diff_scale = max(float(np.abs(frames[label] - frames['linear']).max()) for _, _, label in LAWS)
    fig, axes = plt.subplots(2, len(LAWS), figsize=(11, 4.2))
    for column, (_, _, label) in enumerate(LAWS):
        axes[0, column].imshow(frames[label], cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        axes[0, column].set_title(f'{label}\nmax|w-lin|={devs[label]:.4f}')
        image = axes[1, column].imshow(frames[label] - frames['linear'], cmap='RdBu_r',
                                        vmin=-diff_scale, vmax=diff_scale, interpolation='bilinear')
        for row in range(2):
            axes[row, column].set_xticks([]); axes[row, column].set_yticks([])
    axes[0, 0].set_ylabel(f'Step {mid_step} frame')
    axes[1, 0].set_ylabel('Δ vs linear')
    fig.colorbar(image, ax=axes[1, 3], label=f'Δ intensity (full scale ±{diff_scale:.4f})')
    fig.suptitle(f'Growth-law-weighted temporal progress at grid step {mid_step} and its difference '
                 'from linear weights, Mets_001 (same endpoints, axial slice)')
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return diff_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', default='Mets_001')
    ap.add_argument('--grid-steps', dest='grid_steps', type=int, default=100,
                    help='时间网格刻度数 N（生成条件，非临床随访时间）')
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--module', type=Path, default=MODULE)
    a = ap.parse_args()

    ns = load_namespace(a.module)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    cases = {c.name: c for c in ns['_case_dirs'](ns['TEST_ROOT'])}
    with contextlib.redirect_stdout(io.StringIO()):
        sequence, _ = ns['predict_tumor_progression'](cases[a.case], grid_steps=a.grid_steps,
                                                      change_strength=0.35)
    source, target = sequence[0:1], sequence[-1:]
    mask = ns['_make_tumor_candidate_mask'](source)

    mid_step = a.grid_steps // 2
    weights, devs, burden, frames = {}, {}, {}, {}
    for law, space, label in LAWS:
        with contextlib.redirect_stdout(io.StringIO()):
            weight_tensor, report = ns['temporal_lerp_weights'](source, target, mask, a.grid_steps,
                                                                law=law, space=space, verbose=False)
        w = weight_tensor.reshape(-1).double().numpy()
        weights[label], devs[label] = w, float(report['max_abs_deviation_from_linear'])
        burden[label] = np.array([ns['lesion_burden'](source, target, mask, float(x)) for x in w])
        frame = torch.lerp(source.double(), target.double(), float(w[mid_step]))[0, 0].numpy()
        frames[label] = np.rot90(frame[:, :, frame.shape[2] // 2])
        print(f'{label:22s} max|w-lin|={devs[label]:.4f} w({mid_step})={w[mid_step]:.4f} '
              f'病灶负荷={burden[label][0]:.5f}->{burden[label][-1]:.5f}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_weights(ns, weights, OUT_DIR / '图2_生长律权重曲线.png')
    figure_burden(ns, burden, OUT_DIR / '图9_病灶负荷轨迹.png')
    scale = figure_midframe(frames, devs, OUT_DIR / '图8_生长律中间帧与差异.png', mid_step)
    print(f'图8 差异色标 = ±{scale:.4f}（原强度域，取各生长律 |Δ| 的最大值）')
    print('已输出 ->', OUT_DIR)


if __name__ == '__main__':
    main()
