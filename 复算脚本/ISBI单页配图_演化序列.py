#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISBI 单页摘要配图：三例脑转移瘤演化序列（Mets_020 / 049 / 042，N=100 刻度）。

图片全部来自代码实跑，显示链与 notebook 一致（未自造任何像素）：

    seq  = predict_tumor_progression(case, grid_steps=100)      # 3D VAE + 两阶段 DiT 采样
    blk  = _blackout_tumor_growth(seq)                          # 按生长律进度 w(t) 涂黑病灶候选区
    disp = _normalize_slice_sequence(blk[:, 0, :, :, Z])        # 逐切片百分位拉伸（不改数据）
    resid= clip(|raw(step100) - raw(step0)| * 10, 0, 1)         # 残差用**原始**强度算

显示切片 Z 的选择规则（**公开、可复核，非人工挑选**）：
    取"脑组织面积不小于最大层面 60%"的轴位层，令
        Z(case) = argmax_k #{ 掩膜>0.35 且 距切片背景至少 1 体素 }
    即该层内**候选核内部**体素最多的切面。规则只看掩膜几何，与后续指标无关，
    目的在于让"处方式叠加"在印刷尺寸下可辨认；换切片不改变任何被引用的数字。

必须写明的三点（已同步写入 PDF 题注与正文）：
  1. 黑色区域 = `_make_tumor_candidate_mask` 给出的**病灶候选区** ∩ 生长律进度 w(t) > 0.35。
     它是"处方式"进度叠加显示，**不是**模型预测出的病灶生长；
  2. 51 例测试集**没有专家分割标注**（仅 105 例训练集有 seg），因此这里的黑色区域
     只能来自启发式候选掩膜；该掩膜与专家病灶差异很大（24.39% vs 0.20% 脑体积，
     Dice 0.0071），故不得读作肿瘤分割；
  3. 模型自身造成的强度变化极小且有界（|Δ_gen| ≤ s·c = 0.028），且非病灶局部化
     —— 这正是最右残差列要展示、并由 Fig. 3 审计确认的事实。残差列用**模型原始输出**
     （未做任何显示变换）计算 |Δ|×10 并裁剪到 [0,1]；实测仅 0.25%–0.55% 脑体素 |Δ| ≥ 0.1，
     故裁剪影响可忽略。

输出：核验材料/ISBI单页/fig_progression_N100.png（3.386 in 宽 = 单栏满宽，600 dpi）
"""
import re, tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parent.parent
DITNB = PROJ / '核验材料' / 'P0_生成模块隔离' / 'dit_nb.py'
OUTDIR = PROJ / '核验材料' / 'ISBI单页'
OUTDIR.mkdir(parents=True, exist_ok=True)

toks = re.split(r'# ===== CELL (\d+) =====\n', DITNB.read_text(encoding='utf-8'))
cells = {int(toks[i]): toks[i + 1] for i in range(1, len(toks) - 1, 2)}
ns = {'__name__': 'isbifig'}
for n in [0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13]:
    exec(compile(cells[n], f'<cell{n}>', 'exec'), ns)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
ns['device'] = device
predict = ns['predict_tumor_progression']
blackout = ns['_blackout_tumor_growth']
norm_slice = ns['_normalize_slice_sequence']
make_mask = ns['_make_tumor_candidate_mask']

CASES = ['Mets_020', 'Mets_049', 'Mets_042']
STEPS = [0, 20, 40, 60, 80, 100]
N = 100
CACHE = Path(tempfile.gettempdir()) / 'isbi_seq_N100.npz'   # 约 56 MB，刻意不入库


def select_display_slice(volume_np):
    """公开规则：脑面积足够大的层面中，候选核内部体素最多的那一层。"""
    mask = make_mask(torch.from_numpy(volume_np).float()[None, None])[0, 0].numpy()
    hard = mask > 0.35
    brain = volume_np > 0.04
    outside = F.max_pool3d(
        torch.from_numpy((~brain).astype(np.float32))[None, None], 3, 1, 1
    )[0, 0].numpy() > 0
    interior = hard & (~outside)
    areas = np.array([int(brain[:, :, k].sum()) for k in range(brain.shape[2])])
    keep = areas >= 0.60 * areas.max()
    scores = np.array([int(interior[:, :, k].sum()) if keep[k] else -1 for k in range(len(areas))])
    return int(np.argmax(scores))


if CACHE.exists():
    data = {k: np.load(CACHE)[k] for k in np.load(CACHE).files}
    print('读取序列缓存', CACHE)
else:
    data = {}
    for case in CASES:
        torch.manual_seed(20260926)
        seq, metrics = predict(ns['TEST_ROOT'] / case, grid_steps=N)
        data[case] = seq.detach().cpu().numpy()[:, 0]
        print(f'{case}: softIoU={metrics["contour_iou"]:.4f}')
    np.savez_compressed(CACHE, **data)
    print('写入序列缓存', CACHE)

disp, resid, slices = {}, {}, {}
for case in CASES:
    Z = select_display_slice(data[case][0])
    slices[case] = Z
    seq5 = torch.from_numpy(data[case]).float().unsqueeze(1)      # [N+1,1,64,64,64]
    blk = blackout(seq5)                                          # 真实代码：生长律涂黑
    disp[case] = norm_slice(blk[:, 0, :, :, Z]).numpy()            # 真实代码：逐切片拉伸
    raw = data[case][:, :, :, Z]                                   # 模型原始输出（未做任何显示变换）
    resid[case] = np.clip(np.abs(raw[STEPS[-1]] - raw[STEPS[0]]) * 10, 0, 1)
    print(f'{case}: 显示切片 Z={Z} | 末刻度涂黑像素占全切片 {float((disp[case][-1] == 0).mean()):.3f} '
          f'| 残差 max={float(np.abs(raw[STEPS[-1]] - raw[STEPS[0]]).max()):.4f}')

FS = (3.386, 1.42)
fig, axes = plt.subplots(3, len(STEPS) + 1, figsize=FS, dpi=600,
                         gridspec_kw=dict(wspace=0.06, hspace=0.05))
for r, case in enumerate(CASES):
    for k, st in enumerate(STEPS):
        ax = axes[r, k]
        ax.imshow(np.rot90(disp[case][st]), cmap='gray', vmin=0, vmax=1, interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(f'Step {st}', fontsize=5.8, pad=2.2, color='#1B3A6B')
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color('#888888')
    ax = axes[r, len(STEPS)]
    ax.imshow(np.rot90(resid[case]), cmap='magma', vmin=0, vmax=1, interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    if r == 0:
        ax.set_title(r'$|\Delta|\times 10$', fontsize=5.8, pad=2.2, color='#8B1A1A')
    for sp in ax.spines.values():
        sp.set_linewidth(0.4); sp.set_color('#888888')
    axes[r, 0].set_ylabel(case, fontsize=5.2, rotation=0, labelpad=14, va='center', color='#1B3A6B')

fig.subplots_adjust(left=0.115, right=0.997, top=0.90, bottom=0.005)
out = OUTDIR / 'fig_progression_N100.png'
fig.savefig(out, dpi=600, facecolor='white')
plt.close(fig)
(OUTDIR / 'fig_progression_N100_切片.txt').write_text(
    '\n'.join(f'{c}\t{Z}' for c, Z in slices.items()) + '\n', encoding='utf-8')
print('→', out)
print('显示切片:', slices)
