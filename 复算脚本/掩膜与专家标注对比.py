# -*- coding: utf-8 -*-
"""候选掩膜 vs 专家分割标注（BrainMetShare 训练集 105 例，均带 seg）。

只读评估，不改动算法：直接复用 notebook/dit_nb.py 中的 _make_tumor_candidate_mask 与 _load_png_volume，
在 64^3 统一体数据口径上比较“候选核（掩膜 > 0.35）”“掩膜非零支撑（掩膜 > 0）”
与专家标注（seg，三线性下采样后 > 0.5；另以最近邻下采样做稳健性对照）。

输出：复算脚本/复算结果/掩膜与专家标注对比_per_case.csv 与同目录 .txt 汇总。
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODULE = ROOT / '核验材料' / 'P0_生成模块隔离' / 'dit_nb.py'
OUT_DIR = HERE / '复算结果'
IMG = (64, 64, 64)

# 复用代码本体：仅执行到 CELL 14 之前（不含训练/推理的运行块）
src = MODULE.read_text(encoding='utf-8').split('# ===== CELL 14 =====')[0]
NS = {}
exec(compile(src, str(MODULE), 'exec'), NS)
_load_png_volume = NS['_load_png_volume']
_make_tumor_candidate_mask = NS['_make_tumor_candidate_mask']
_mask_device = NS['device']

def load(case_dir, modality, mode='trilinear'):
    """与代码一致的读取路径，但可指定插值方式（用于 seg 的稳健性对照）。"""
    files = sorted((Path(case_dir) / modality).glob('*.png'), key=NS['_slice_number'])
    slices = [np.asarray(Image.open(p).convert('L').resize((IMG[1], IMG[0]), Image.Resampling.BILINEAR),
                         dtype=np.float32) / 255.0 for p in files]
    vol = torch.from_numpy(np.stack(slices, axis=0)).unsqueeze(0).unsqueeze(0)
    kw = dict(align_corners=False) if mode == 'trilinear' else {}
    return F.interpolate(vol, size=IMG, mode=mode, **kw)[0]

def load_seg_native(case_dir):
    """原生分辨率专家标注（256x256x150，>127 为病灶），用于最宽松（最大池化）参照。"""
    files = sorted((Path(case_dir) / 'seg').glob('*.png'), key=NS['_slice_number'])
    slices = [np.asarray(Image.open(p).convert('L'), dtype=np.float32) / 255.0 for p in files]
    vol = torch.from_numpy(np.stack(slices, axis=0)).unsqueeze(0).unsqueeze(0)
    return vol


def main():
    train_root = Path(NS['TRAIN_ROOT'])
    cases = sorted(p for p in train_root.glob('Mets_*') if p.is_dir())
    rows = []
    for case in cases:
        source = load(case, 't1_gd')
        seg_tri = load(case, 'seg', 'trilinear')[0].numpy() > 0.5
        seg_near = load(case, 'seg', 'nearest')[0].numpy() > 0.5
        # 最宽松参照：原生分辨率标注做最大池化（任一子体素为病灶则该体素计为病灶）
        seg_max = F.adaptive_max_pool3d(load_seg_native(case), IMG)[0, 0].numpy() > 0.5
        # 与推理链一致：函数按 5D (N,C,D,H,W) 调用，掩膜在同一 64^3 网格上
        mask = _make_tumor_candidate_mask(source.unsqueeze(0))[0, 0].numpy()
        brain = source[0].numpy() > 0.035
        core = mask > 0.35
        support = mask > 0

        def pr(pred, gt):
            inter = int((pred & gt).sum())
            dice = 2.0 * inter / max(1, int(pred.sum() + gt.sum()))
            rec = inter / max(1, int(gt.sum()))
            prec = inter / max(1, int(pred.sum()))
            return dice, rec, prec

        d_tri, r_tri, p_tri = pr(core, seg_tri)
        d_near, r_near, p_near = pr(core, seg_near)
        d_sup, r_sup, _ = pr(support, seg_tri)
        d_max, r_max, p_max = pr(core, seg_max)
        d_max_sup, r_max_sup, _ = pr(support, seg_max)
        rows.append(dict(
            case=case.name, brain_voxels=int(brain.sum()),
            core_voxels=int(core.sum()), support_voxels=int(support.sum()),
            seg_voxels_tri=int(seg_tri.sum()), seg_voxels_near=int(seg_near.sum()),
            core_brain_frac=core.sum() / max(1, brain.sum()),
            seg_brain_frac=seg_tri.sum() / max(1, brain.sum()),
            dice_tri=d_tri, recall_tri=r_tri, precision_tri=p_tri,
            dice_near=d_near, recall_near=r_near, precision_near=p_near,
            dice_support=d_sup, recall_support=r_sup,
            seg_voxels_max=int(seg_max.sum()),
            dice_max=d_max, recall_max=r_max, precision_max=p_max,
            dice_support_max=d_max_sup, recall_support_max=r_max_sup,
        ))
        if len(rows) % 20 == 0:
            print(f'  已处理 {len(rows)}/{len(cases)} 例 ...', flush=True)

    import csv
    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / '掩膜与专家标注对比_per_case.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def ms(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return v.mean(), v.std(ddof=1), v.min(), v.max()

    lines = ['候选掩膜 vs 专家分割标注（BrainMetShare 训练集，n=%d，64^3 体数据口径）' % len(rows), '=' * 96]
    for key, label in [
        ('core_brain_frac', '候选核(>0.35)占脑区比例'),
        ('seg_brain_frac', '专家病灶(seg>0.5)占脑区比例'),
        ('dice_tri', 'Dice（候选核 vs 三线性下采样 seg）'),
        ('recall_tri', '召回（专家病灶被候选核覆盖比例）'),
        ('precision_tri', '精确率'),
        ('dice_near', 'Dice（候选核 vs 最近邻下采样 seg，稳健性对照）'),
        ('recall_support', '召回（掩膜非零支撑 >0）'),
        ('dice_support', 'Dice（掩膜非零支撑 vs seg）'),
        ('dice_max', 'Dice（候选核 vs 最大池化 seg，最宽松参照）'),
        ('recall_max', '召回（候选核 vs 最大池化 seg，最宽松参照）'),
        ('dice_support_max', 'Dice（掩膜非零支撑 vs 最大池化 seg）'),
        ('recall_support_max', '召回（掩膜非零支撑 vs 最大池化 seg）'),
    ]:
        m, s, lo, hi = ms(key)
        lines.append('%-46s 均值=%.4f ± %.4f | 范围 %.4f – %.4f' % (label, m, s, lo, hi))
    core_frac, _, _, _ = ms('core_brain_frac')
    seg_frac, _, _, _ = ms('seg_brain_frac')
    lines.append('')
    lines.append('候选核体积 / 专家病灶体积 = %.1f 倍（按占脑区比例之比）' % (core_frac / seg_frac))
    vol_ratio = np.array([r['core_voxels'] / max(1, r['seg_voxels_tri']) for r in rows])
    lines.append('逐例体积倍数：中位数 %.1f 倍，范围 %.1f – %.1f 倍' % (float(np.median(vol_ratio)), vol_ratio.min(), vol_ratio.max()))
    report = '\n'.join(lines)
    (OUT_DIR / '掩膜与专家标注对比.txt').write_text(report + '\n', encoding='utf-8')
    print('\n' + report)
    print('\n报告 ->', OUT_DIR / '掩膜与专家标注对比.txt')

if __name__ == '__main__':
    main()
