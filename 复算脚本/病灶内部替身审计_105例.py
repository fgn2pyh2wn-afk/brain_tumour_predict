#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""病灶内部指标的替身审计（专家标注口径，BrainMetShare 训练集 105 例）。

与 `病灶内部指标_105例交叉验证.py` 同一批病例，但把生成器依次替换为
  dit_seedA / dit_seedB  真实 DiT 采样（两个随机种子）
  frozen                 冻结采样（固定种子，等价于确定性单链）
  shuffled               用上一例的生成结果
  identity               源图本身（不生成）
  noise                  随机噪声
  mean_latent            全局平均潜变量解码得到的常数图
再走**同一套**结构保持后处理，然后在三种区域定义上算同款指标：
  R1 候选核 m>0.35（论文当前口径） R2 专家病灶(三线性) R3 专家病灶(最大池化上界)

目的：回答"换成真正的病灶区域后，指标能否区分真采样器与替身"。
输出：复算脚本/复算结果/病灶内部替身审计_105例_per_case.csv 与汇总 txt。
"""
import argparse, contextlib, csv, io, json, random, time
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
VARIANTS = ['dit_seedA', 'dit_seedB', 'frozen', 'shuffled', 'identity', 'noise', 'mean_latent']


def load_notebook_ns(nb_path, device):
    raw = json.loads(Path(nb_path).read_text(encoding='utf-8'))
    code_cells = [''.join(c['source']) for c in raw['cells'] if c['cell_type'] == 'code']
    ns = {'__name__': 'surrogate_lesion_audit'}
    for idx in CELLS:
        exec(compile(code_cells[idx], f'<cell{idx}>', 'exec'), ns)
    ns['device'] = device
    return ns


def stats(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return dict(mean=float('nan'), std=float('nan'), min=float('nan'), max=float('nan'), n=0)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                min=float(v.min()), max=float(v.max()), n=int(v.size))


def form(s, fmt='.4f'):
    return '—' if s['n'] == 0 else f"{s['mean']:{fmt}}±{s['std']:{fmt}}"


def load_seg_maxpool(case_dir):
    files = sorted((Path(case_dir) / 'seg').glob('*.png'))
    slices = [np.asarray(Image.open(p).convert('L'), dtype=np.float32) / 255.0 for p in files]
    vol = torch.from_numpy(np.stack(slices, axis=0)).unsqueeze(0).unsqueeze(0)
    return F.adaptive_max_pool3d(vol, IMG)[0, 0].numpy()


def load_seg_tri(case_dir):
    files = sorted((Path(case_dir) / 'seg').glob('*.png'))
    slices = [np.asarray(Image.open(p).convert('L').resize((IMG[1], IMG[0]), Image.Resampling.BILINEAR),
                         dtype=np.float32) / 255.0 for p in files]
    vol = torch.from_numpy(np.stack(slices, axis=0)).unsqueeze(0).unsqueeze(0)
    return F.interpolate(vol, size=IMG, mode='trilinear', align_corners=False)[0, 0].numpy()


def region_metrics(x0, xN, region, brain):
    r = torch.from_numpy(region); b = torch.from_numpy(brain)
    outside = (b & (~r))
    d = xN - x0; ad = d.abs()
    n_in, n_out = int(r.sum()), int(outside.sum())
    out = dict(n_voxels=n_in,
               signed_mean_change=float(d[r].mean()) if n_in else float('nan'),
               abs_mean_change=float(ad[r].mean()) if n_in else float('nan'))
    out['abs_mean_change_outside'] = float(ad[outside].mean()) if n_out else float('nan')
    out['outside_inside_ratio'] = (out['abs_mean_change_outside'] / out['abs_mean_change']
                                   if n_in and n_out and out['abs_mean_change'] > 0 else float('nan'))
    x0b, xNb = x0.unsqueeze(0).unsqueeze(0), xN.unsqueeze(0).unsqueeze(0)
    s_det = (x0b - F.avg_pool3d(x0b, 3, 1, 1))[0, 0]; t_det = (xNb - F.avg_pool3d(xNb, 3, 1, 1))[0, 0]
    sv, tv = s_det[r], t_det[r]
    out['texture_corr_in_region'] = (float(torch.corrcoef(torch.stack([sv, tv]))[0, 1])
                                     if n_in > 2 and float(sv.std()) > 0 and float(tv.std()) > 0 else float('nan'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=0)
    ap.add_argument('--grid-steps', dest='grid_steps', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--notebook', type=Path, default=NB_DEFAULT)
    ap.add_argument('--ckpt', type=Path, default=CKPT_DEFAULT)
    ap.add_argument('--out', type=Path, default=HERE / '复算结果')
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps', 'auto'])
    a = ap.parse_args()

    device = torch.device('mps' if (a.device == 'auto' and torch.backends.mps.is_available())
                          else a.device) if a.device != 'auto' else torch.device('cpu')
    torch.set_num_threads(min(8, torch.get_num_threads()))
    a.out.mkdir(parents=True, exist_ok=True)
    print(f'设备={device}', flush=True)
    ns = load_notebook_ns(a.notebook, device)
    ck = torch.load(a.ckpt, map_location=device, weights_only=True)
    vae = ns['SimpleVAE3D']().to(device).eval(); vae.load_state_dict(ck['vae'])
    unet = ns['DiT3D']().to(device).eval(); unet.load_state_dict(ck['unet'])
    sched = ns['LDMScheduler3D']()
    cases = sorted(p for p in Path(ns['TRAIN_ROOT']).glob('Mets_*') if p.is_dir())
    cases = cases[:a.cases] if a.cases else cases
    print(f'训练集病例数={len(cases)}  N={a.grid_steps}  变体={len(VARIANTS)}', flush=True)

    def sample(z0, cond, s):
        torch.manual_seed(s); z1 = sched.pass1_adani_infer(unet, z0, a.grid_steps, cond)
        torch.manual_seed(s + 1); zf = sched.pass2_refine(unet, z1, z0, cond)
        return vae.decode(zf).clamp(0, 1)

    rows = []; prev = None; t0 = time.time()
    for i, case in enumerate(cases, 1):
        source = ns['_load_inference_volume'](case).unsqueeze(0).to(device)
        mask = ns['_make_tumor_candidate_mask'](source)
        brain = (source.cpu() > 0.035)[0, 0].numpy()
        core = mask.cpu()[0, 0].numpy() > 0.35
        seg_tri = load_seg_tri(case) > 0.5
        seg_max = load_seg_maxpool(case) > 0.5
        with torch.no_grad():
            z0 = vae.encode(source); cond = F.adaptive_avg_pool3d(z0, 1).flatten(1)
            gen = {'dit_seedA': sample(z0, cond, a.seed + i),
                   'dit_seedB': sample(z0, cond, a.seed + i + 1),
                   'frozen': sample(z0, cond, 0),
                   'identity': source.clone(),
                   'noise': torch.rand_like(source),
                   'mean_latent': vae.decode(z0.mean(dim=(2, 3, 4), keepdim=True).expand_as(z0)).clamp(0, 1)}
        gen['shuffled'] = prev if prev is not None else gen['dit_seedA']
        prev = gen['dit_seedA'].clone()
        for v in VARIANTS:
            with torch.no_grad():
                tgt = ns['_preserve_brain_structure'](source, gen[v], mask, 0.35)
                raw = (gen[v].cpu() - source.cpu())
                gen_cpu = gen[v].cpu()
                tgt_cpu = tgt.cpu()
            # 论文同款 4 指标（整幅口径），用于与 §6 表6 对齐
            x0 = source.cpu()[0, 0]; xN = tgt_cpu[0, 0]
            mc = region_metrics(x0, xN, core, brain)
            m_tri = region_metrics(x0, xN, seg_tri, brain)
            m_max = region_metrics(x0, xN, seg_max, brain)
            with contextlib.redirect_stdout(io.StringIO()):
                mm = ns['compare_structure_quality'](source.cpu(), tgt_cpu)
            row = dict(case=case.name, variant=v,
                       soft_iou=mm['contour_iou'], hard_iou=mm['hard_contour_iou'],
                       tex_corr=mm['texture_correlation'], candidate_region_change=mm['lesion_mean_change'],
                       candidate_abs_change=mc['abs_mean_change'], candidate_ratio=mc['outside_inside_ratio'],
                       seg_change=m_tri['signed_mean_change'], seg_abs_change=m_tri['abs_mean_change'],
                       seg_ratio=m_tri['outside_inside_ratio'], seg_texture=m_tri['texture_corr_in_region'],
                       segmax_change=m_max['signed_mean_change'], segmax_abs_change=m_max['abs_mean_change'],
                       segmax_ratio=m_max['outside_inside_ratio'], segmax_texture=m_max['texture_corr_in_region'],
                       preclip_abs_mean=float(raw.abs().mean()),
                       gen_std=float(gen_cpu.std()), tgt_std=float(tgt_cpu.std()))
            rows.append(row)
        if i % 5 == 0 or i == len(cases):
            print(f'  {i}/{len(cases)}  累计 {time.time()-t0:.0f}s', flush=True)

    csv_path = a.out / '病灶内部替身审计_105例_per_case.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    metric_disp = [('soft_iou', 'Soft 脑轮廓 IoU', '.4f'), ('hard_iou', 'Hard 脑轮廓 IoU', '.4f'),
                   ('tex_corr', '高频纹理相关(整幅)', '.4f'),
                   ('candidate_region_change', 'R1 候选核 平均灰度变化', '.4f'),
                   ('seg_change', 'R2 专家病灶 平均灰度变化', '.4f'),
                   ('seg_abs_change', 'R2 专家病灶 平均绝对变化', '.5f'),
                   ('seg_ratio', 'R2 专家病灶 外/内变化比', '.3f'),
                   ('seg_texture', 'R2 专家病灶 纹理相关', '.4f'),
                   ('segmax_change', 'R3 专家上界 平均灰度变化', '.4f'),
                   ('segmax_abs_change', 'R3 专家上界 平均绝对变化', '.5f'),
                   ('segmax_ratio', 'R3 专家上界 外/内变化比', '.3f'),
                   ('preclip_abs_mean', '裁剪前 |residual| 均值', '.5f'),
                   ('gen_std', '生成图解码 std', '.4f')]
    L = ['病灶内部指标的替身审计（BrainMetShare 训练集 n=%d，N=%d，7 种生成模块设置）' % (len(cases), a.grid_steps),
         '=' * 112,
         '每格 = 均值±标准差（105 例）；"极差" = 逐例跨 7 种设置的最大−最小']
    for key, lab, fmt in metric_disp:
        per_variant = {v: [r[key] for r in rows if r['variant'] == v] for v in VARIANTS}
        cell = ' | '.join('%s %s' % (v, form(stats(per_variant[v]), fmt)) for v in VARIANTS)
        # 逐例跨设置极差
        per_case = {}
        for r in rows:
            per_case.setdefault(r['case'], []).append(r[key])
        spread = [max(x) - min(x) for x in per_case.values() if all(np.isfinite(x))]
        L.append('')
        L.append('[%s]' % lab)
        for v in VARIANTS:
            s = stats(per_variant[v]); L.append('    %-12s %s | 范围 %s – %s' % (v, form(s, fmt), f"{s['min']:{fmt}}", f"{s['max']:{fmt}}"))
        if spread:
            L.append('    逐例跨设置极差: 中位 %.6f | 最大 %.6f' % (float(np.median(spread)), float(np.max(spread))))
        del cell
    report = '\n'.join(L)
    (a.out / '病灶内部替身审计_105例.txt').write_text(report + '\n', encoding='utf-8')
    print('\n' + report)
    print('\n逐例 ->', csv_path)


if __name__ == '__main__':
    main()
