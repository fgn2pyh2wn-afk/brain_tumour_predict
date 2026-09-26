#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""病灶内部指标（专家标注口径）交叉验证脚本。

动机：论文 §4 报告的 "lesion mean intensity change" 与 "outside/inside change ratio"
实际是在**启发式候选核**（_make_tumor_candidate_mask > 0.35，占脑区约 24%）上统计的，
而 §8 又写明该候选核远大于专家病灶（占脑区约 0.2%，Dice 0.0071）。两处口径同名不同义。
本脚本在同一批病例、同一次生成序列上，把同一个指标**同时**在三种区域定义下算一遍：

  R1 候选核      m > 0.35（论文当前口径，启发式）
  R2 专家病灶    seg 三线性下采样到 64^3 后 > 0.5（主口径）
  R3 专家病灶    seg 原生分辨率最大池化到 64^3 后 > 0.5（最宽松上界口径）

输出逐例 CSV 与汇总 TXT，用于把 §4 的指标改名/改口径，或直接替换为病灶内部数值。

重要限制（写进结论时必须保留）：专家标注只存在于**发布方训练集**（105 例），
而这 105 例正是本模型训练所用数据，因此本表是**口径审计（audit），不是留出集性能**。

用法：
  python3 病灶内部指标_105例交叉验证.py                # 全部 105 例
  python3 病灶内部指标_105例交叉验证.py --cases 5      # 冒烟
"""
import argparse, contextlib, csv, io, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
NB_DEFAULT = HERE.parent / '脑肿瘤预测_DiT算法.ipynb'
CKPT_DEFAULT = Path('/Users/chris/Desktop/brain_tumor_predict/brain_tumor_ldm.pt')
CELLS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13]
IMG = (64, 64, 64)


def load_notebook_ns(nb_path, device):
    raw = json.loads(Path(nb_path).read_text(encoding='utf-8'))
    code_cells = [''.join(c['source']) for c in raw['cells'] if c['cell_type'] == 'code']
    ns = {'__name__': 'lesion_internal_audit'}
    for idx in CELLS:
        exec(compile(code_cells[idx], f'<cell{idx}>', 'exec'), ns)
    ns['device'] = device
    return ns


def seed_all(ns, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); ns['torch'].manual_seed(seed)


def stats(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return dict(mean=float('nan'), std=float('nan'), min=float('nan'), max=float('nan'), n=0)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                min=float(v.min()), max=float(v.max()), n=int(v.size))


def form(s, fmt='.4f'):
    return '—' if s['n'] == 0 else f"{s['mean']:{fmt}}±{s['std']:{fmt}}"


def load_seg(case_dir, mode):
    """专家标注下采样到 64^3：mode='trilinear' / 'nearest'。"""
    files = sorted((Path(case_dir) / 'seg').glob('*.png'))
    slices = [np.asarray(Image.open(p).convert('L').resize((IMG[1], IMG[0]), Image.Resampling.BILINEAR),
                         dtype=np.float32) / 255.0 for p in files]
    vol = torch.from_numpy(np.stack(slices, axis=0)).unsqueeze(0).unsqueeze(0)
    kw = dict(align_corners=False) if mode == 'trilinear' else {}
    return F.interpolate(vol, size=IMG, mode=mode, **kw)[0, 0].numpy()


def load_seg_maxpool(case_dir):
    """最宽松上界口径：原生分辨率标注做最大池化（任一子体素为病灶则该体素计为病灶）。"""
    files = sorted((Path(case_dir) / 'seg').glob('*.png'))
    slices = [np.asarray(Image.open(p).convert('L'), dtype=np.float32) / 255.0 for p in files]
    vol = torch.from_numpy(np.stack(slices, axis=0)).unsqueeze(0).unsqueeze(0)
    return F.adaptive_max_pool3d(vol, IMG)[0, 0].numpy()


def region_metrics(x0, xN, region, brain):
    """在给定区域内计算论文同款指标（有符号均值变化、平均绝对变化、区域外/内变化比）。"""
    d = (xN - x0)
    ad = d.abs()
    r = torch.from_numpy(region)
    b = torch.from_numpy(brain)
    outside = (b & (~r))
    n_in, n_out = int(r.sum()), int(outside.sum())
    out = dict(n_voxels=n_in, n_brain_outside=n_out,
               signed_mean_change=float(d[r].mean()) if n_in else float('nan'),
               abs_mean_change=float(ad[r].mean()) if n_in else float('nan'),
               abs_mean_change_outside=float(ad[outside].mean()) if n_out else float('nan'))
    out['outside_inside_ratio'] = (out['abs_mean_change_outside'] / out['abs_mean_change']
                                   if n_in and n_out and out['abs_mean_change'] > 0 else float('nan'))
    # 区域内高频纹理相关（与 compare_structure_quality 同一算法，但限制在区域内）
    x0b, xNb = x0.unsqueeze(0).unsqueeze(0), xN.unsqueeze(0).unsqueeze(0)
    s_det = (x0b - F.avg_pool3d(x0b, kernel_size=3, stride=1, padding=1))[0, 0]
    t_det = (xNb - F.avg_pool3d(xNb, kernel_size=3, stride=1, padding=1))[0, 0]
    sv, tv = s_det[r], t_det[r]
    if n_in > 2 and float(sv.std()) > 0 and float(tv.std()) > 0:
        out['texture_corr_in_region'] = float(torch.corrcoef(torch.stack([sv, tv]))[0, 1])
    else:
        out['texture_corr_in_region'] = float('nan')
    out['brain_frac'] = n_in / max(1, int(b.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=0, help='病例数上限，0=全部 105 例')
    ap.add_argument('--grid-steps', dest='grid_steps', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--notebook', type=Path, default=NB_DEFAULT)
    ap.add_argument('--out', type=Path, default=HERE / '复算结果')
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps', 'auto'])
    a = ap.parse_args()

    if a.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(a.device)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    a.out.mkdir(parents=True, exist_ok=True)
    print(f'设备={device}  notebook={a.notebook}', flush=True)
    ns = load_notebook_ns(a.notebook, device)
    cases = sorted(p for p in Path(ns['TRAIN_ROOT']).glob('Mets_*') if p.is_dir())
    assert all((c / 'seg').exists() for c in cases), '训练集应全部带 seg 标注'
    cases = cases[:a.cases] if a.cases else cases
    print(f'训练集病例数={len(cases)}（全部带专家 seg；注意这 105 例即训练数据）  N={a.grid_steps}', flush=True)

    rows = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        seed_all(ns, a.seed + i)
        with contextlib.redirect_stdout(io.StringIO()):
            seq, metrics = ns['predict_tumor_progression'](case, grid_steps=a.grid_steps)
        x0 = torch.as_tensor(seq[0, 0], dtype=torch.float32)
        xN = torch.as_tensor(seq[-1, 0], dtype=torch.float32)
        src5 = x0.unsqueeze(0).unsqueeze(0)
        mask = ns['_make_tumor_candidate_mask'](src5.to(device)).cpu()[0, 0].numpy()
        seg_tri = (load_seg(case, 'trilinear') > 0.5)
        seg_max = (load_seg_maxpool(case) > 0.5)
        brain = (x0.numpy() > 0.035)

        core = mask > 0.35
        m_core = region_metrics(x0, xN, core, brain)
        m_tri = region_metrics(x0, xN, seg_tri, brain)
        m_max = region_metrics(x0, xN, seg_max, brain)
        inter = int((core & seg_tri).sum())
        row = dict(case=case.name,
                   brain_voxels=int(brain.sum()),
                   core_voxels=m_core['n_voxels'], seg_voxels=int(seg_tri.sum()), segmax_voxels=int(seg_max.sum()),
                   core_brain_frac=m_core['brain_frac'], seg_brain_frac=m_tri['brain_frac'],
                   segmax_brain_frac=m_max['brain_frac'],
                   recall_core_vs_seg=(inter / max(1, int(seg_tri.sum()))),
                   precision_core_vs_seg=(inter / max(1, int(core.sum()))),
                   paper_signed_change_core=m_core['signed_mean_change'],
                   paper_ratio_core=m_core['outside_inside_ratio'],
                   paper_abschange_core=m_core['abs_mean_change'],
                   texture_corr_core=m_core['texture_corr_in_region'])
        for tag, m in (('seg', m_tri), ('segmax', m_max)):
            row[f'signed_change_{tag}'] = m['signed_mean_change']
            row[f'abschange_{tag}'] = m['abs_mean_change']
            row[f'abschange_outside_{tag}'] = m['abs_mean_change_outside']
            row[f'ratio_{tag}'] = m['outside_inside_ratio']
            row[f'texture_corr_{tag}'] = m['texture_corr_in_region']
        rows.append(row)
        if i % 10 == 0 or i == len(cases):
            print(f'  {i}/{len(cases)}  累计 {time.time()-t0:.0f}s', flush=True)

    csv_path = a.out / '病灶内部指标_105例_per_case.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def S(k):
        return stats([r[k] for r in rows])

    L = []
    L.append('病灶内部指标（专家标注口径）交叉验证 —— BrainMetShare 训练集 n=%d，N=%d 刻度' % (len(rows), a.grid_steps))
    L.append('=' * 108)
    L.append('⚠ 本表病例 = 模型训练病例（发布方训练集），专家标注仅存在于该划分；本表是口径审计，不是留出集性能。')
    L.append('')
    L.append('[区域规模]')
    for k, lab in (('core_brain_frac', 'R1 候选核 m>0.35 占脑区'),
                   ('seg_brain_frac', 'R2 专家病灶(三线性) 占脑区'),
                   ('segmax_brain_frac', 'R3 专家病灶(最大池化上界) 占脑区')):
        s = S(k); L.append('  %-34s 均值=%.4f ± %.4f | 范围 %.4f – %.4f' % (lab, s['mean'], s['std'], s['min'], s['max']))
    s = S('recall_core_vs_seg'); L.append('  %-34s 均值=%.4f ± %.4f | 范围 %.4f – %.4f' % ('专家病灶被候选核覆盖比例', s['mean'], s['std'], s['min'], s['max']))
    s = S('precision_core_vs_seg'); L.append('  %-34s 均值=%.4f ± %.4f | 范围 %.4f – %.4f' % ('候选核中确为病灶的比例', s['mean'], s['std'], s['min'], s['max']))
    L.append('')
    L.append('[同一指标、三种区域定义]（R1=论文当前口径；R2/R3=专家病灶口径）')
    for key, lab, fmt in (('signed_change', '病灶/区域平均灰度变化', '.4f'),
                          ('abschange', '区域平均绝对变化', '.5f'),
                          ('abschange_outside', '区域外脑区平均绝对变化', '.5f'),
                          ('ratio', '区域外/区域内变化比', '.4f'),
                          ('texture_corr', '区域内高频纹理相关', '.4f')):
        if key == 'signed_change':
            s1 = S('paper_signed_change_core')
        elif key == 'abschange':
            s1 = S('paper_abschange_core')
        elif key == 'ratio':
            s1 = S('paper_ratio_core')
        elif key == 'texture_corr':
            s1 = S('texture_corr_core')
        else:
            s1 = stats([r['abschange_outside_seg'] for r in rows])  # 占位，下面单独打印
        L.append('  %s' % lab)
        for tag, s in (('R1 候选核      ', s1), ('R2 专家病灶    ', S(f'{key}_seg')), ('R3 专家上界口径', S(f'{key}_segmax'))):
            if s['n'] == 0:
                continue
            if key == 'abschange_outside':
                continue
            L.append('    %s 均值=%s | 范围 %s – %s' % (tag, form(s, fmt), f"{s['min']:{fmt}}", f"{s['max']:{fmt}}"))
    L.append('')
    L.append('[关键对比] 同一个 "mean change" 指标，换区域定义后是否还测得出来（论文 §4 vs §8 的口径冲突）')
    for lab, k1, k2 in (('R1 候选核', 'paper_signed_change_core', None),
                        ('R2 专家病灶(三线性)', 'signed_change_seg', None),
                        ('R3 专家病灶(最大池化)', 'signed_change_segmax', None)):
        s = S(k1); L.append('  %-22s 平均灰度变化=%s' % (lab, form(s, '.4f')))
    L.append('')
    L.append('[变化是否在病灶内更集中？] 区域外/内变化比 = 1 表示无任何局部化')
    for lab, k in (('R1 候选核', 'paper_ratio_core'), ('R2 专家病灶(三线性)', 'ratio_seg'),
                   ('R3 专家病灶(最大池化)', 'ratio_segmax')):
        s = S(k); L.append('  %-22s 比值=%s' % (lab, form(s, '.4f')))
    report = '\n'.join(L)
    (a.out / '病灶内部指标_105例.txt').write_text(report + '\n', encoding='utf-8')
    print('\n' + report)
    print('\n逐例 ->', csv_path)


if __name__ == '__main__':
    main()
