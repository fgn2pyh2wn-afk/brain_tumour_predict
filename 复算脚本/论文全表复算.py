#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文全表复算：只使用 notebook（脑肿瘤预测_DiT算法.ipynb）中的模型/函数定义 + 项目内 checkpoint，
在真实数据（BRAINMETASTASIS/test，51 例）上重跑论文里的每一张表，并把逐例结果落盘。

覆盖：
  表2  自编码器重建（51 例，整幅口径 MSE/MAE/PSNR）
  表3  时间网格 N = 100 刻度的进展模拟结构保持指标（51 例；刻度是生成条件，非临床随访时间）
  表4  生长律时间权重消融（前 10 例）
  表5  四条命题核验（51 例） + 三随机种子核验（51 例）
  表6  生成模块替身对照（51 例）
  表7  结构保持后处理开关消融（51 例）
  图11 替身对照点图 / 图12 裁剪前残差饱和口径（保存 PNG 与 CSV）

用法：
  python3 论文全表复算.py                     # 全部跑（约 8-10 分钟，CPU）
  python3 论文全表复算.py --cases 5           # 只跑前 5 例（冒烟）
  python3 论文全表复算.py --only 表2,表3      # 只跑指定表
  python3 论文全表复算.py --out ~/Desktop/复算结果
参数：
  --notebook  notebook 路径（默认与本脚本同级的上一级目录下的 脑肿瘤预测_DiT算法.ipynb）
  --ckpt      checkpoint 路径（默认 /Users/chris/Desktop/brain_tumor_ldm.pt）
  --grid-steps 时间网格刻度数 N（默认 100，与论文表3/表4 一致）
  --seed      主随机种子（默认 20260925，与论文一致）
  --seeds     种子核验用的三组种子（默认 20260925,20260926,20260927）
"""
import argparse, contextlib, csv, io, json, random, re, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
NB_DEFAULT = HERE.parent / '脑肿瘤预测_DiT算法.ipynb'
CKPT_DEFAULT = Path('/Users/chris/Desktop/brain_tumor_ldm.pt')
CELLS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 12, 13]


# ----------------------------------------------------------------- 工具
def load_notebook_ns(nb_path, device):
    """把 notebook 的相关 cell 按顺序 exec 进一个命名空间，返回该命名空间。"""
    raw = json.loads(Path(nb_path).read_text(encoding='utf-8'))
    code_cells = [''.join(c['source']) for c in raw['cells'] if c['cell_type'] == 'code']
    ns = {'__name__': 'paper_recompute'}
    for idx in CELLS:
        exec(compile(code_cells[idx], f'<cell{idx}>', 'exec'), ns)
    ns['device'] = device
    return ns


def seed_all(ns, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); ns['torch'].manual_seed(seed)


def load_models(ns, ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=True)
    vae = ns['SimpleVAE3D']().to(device).eval(); vae.load_state_dict(ck['vae'])
    unet = ns['DiT3D']().to(device).eval(); unet.load_state_dict(ck['unet'])
    return vae, unet, ck


def stats(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return dict(mean=float('nan'), std=float('nan'), min=float('nan'), max=float('nan'), n=0)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                min=float(v.min()), max=float(v.max()), n=int(v.size))


def form(s, fmt='.4f'):
    if s['n'] == 0:
        return '—'
    return f"{s['mean']:{fmt}}±{s['std']:{fmt}}"


def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


# ----------------------------------------------------------------- 表2 / 表3 / 表5
def run_recon_progression(ns, vae, cases, grid_steps, seed, out):
    rows2, rows3, rows5 = [], [], []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        # 表2：整幅体数据重建
        vol = ns['_load_png_volume'](case).to(ns['device'])
        with torch.no_grad():
            rec = vae.decode(vae.encode(vol.unsqueeze(0))).clamp(0, 1)[0]
        mse = F.mse_loss(rec, vol).item(); mae = F.l1_loss(rec, vol).item()
        rows2.append(dict(case=case.name, mse=mse, mae=mae,
                          psnr_db=-10.0 * np.log10(mse) if mse > 0 else float('inf')))

        # 表3：时间网格 N = 100 刻度的进展模拟（走 notebook 原函数）
        seed_all(ns, seed + i)
        with contextlib.redirect_stdout(io.StringIO()):
            seq, metrics = ns['predict_tumor_progression'](case, grid_steps=grid_steps)
        source, target = seq[0:1], seq[-1:]
        mask = ns['_make_tumor_candidate_mask'](source).cpu()
        m_bin = (mask > 0.35).float(); brain = (source.cpu() > 0.035).float()
        delta = (target.cpu() - source.cpu()).abs()
        in_les = delta[m_bin.expand_as(delta) > 0]; out_br = delta[((brain - m_bin).clamp_min(0)).expand_as(delta) > 0]
        rows3.append(dict(case=case.name,
                          soft_contour_iou=metrics['contour_iou'],
                          hard_contour_iou=metrics['hard_contour_iou'],
                          texture_corr=metrics['texture_correlation'],
                          lesion_mean_change=metrics['lesion_mean_change'],
                          abs_change_in_lesion=float(in_les.mean()) if in_les.numel() else 0.0,
                          abs_change_in_brain_outside_lesion=float(out_br.mean()) if out_br.numel() else 0.0))

        # 表5：命题核验（复用同一序列，避免重复推理）
        seq_c = seq.cpu()
        with contextlib.redirect_stdout(io.StringIO()):
            w, rep = ns['temporal_lerp_weights'](source.cpu(), target.cpu(), mask, grid_steps)
        seqf = np.asarray(seq_c.reshape(seq_c.shape[0], -1), dtype=np.float64)
        w_np = np.asarray(w.detach().cpu() if torch.is_tensor(w) else w, dtype=np.float64).reshape(-1)
        inf_norm = np.abs(seqf - seqf[0]).max(axis=1)
        ref = float(inf_norm[-1])
        rel2 = float((np.abs(inf_norm - w_np * ref) / max(ref, 1e-12)).max()) if ref > 0 else 0.0
        w32 = w_np.astype(np.float32)
        inc = np.abs(np.diff(w32))
        rows5.append(dict(case=case.name,
                          prop1_affinity_residual=float(rep.get('burden_affinity_residual', np.nan)),
                          prop2_rel_dev=rel2,
                          prop3_lipschitz_measured=float(inc.max()),
                          prop4_max_dev_from_linear=float(rep.get('max_abs_deviation_from_linear', np.nan)),
                          prop4_argmax_step=int(np.argmax(np.abs(w_np - np.arange(len(w_np)) / max(len(w_np) - 1, 1)))),
                          monotone=bool(np.all(np.diff(w_np) >= -1e-12))))
        if i % 5 == 0 or i == len(cases):
            print(f'  [表2/表3/表5] {i}/{len(cases)}  累计 {time.time()-t0:.0f}s', flush=True)
    write_csv(out / '表2_重建_per_case.csv', rows2)
    write_csv(out / '表3_进展模拟_per_case.csv', rows3)
    write_csv(out / '表5_命题核验_per_case.csv', rows5)
    s2 = {k: stats([r[k] for r in rows2]) for k in ('mse', 'mae', 'psnr_db')}
    s3 = {k: stats([r[k] for r in rows3]) for k in
          ('soft_contour_iou', 'hard_contour_iou', 'texture_corr', 'lesion_mean_change',
           'abs_change_in_lesion', 'abs_change_in_brain_outside_lesion')}
    print(f"  [表2] 整幅 MSE={form(s2['mse'],'.6f')}  MAE={form(s2['mae'],'.6f')}  PSNR={form(s2['psnr_db'],'.3f')} dB")
    print(f"  [表3] SoftIoU={form(s3['soft_contour_iou'])}  HardIoU={form(s3['hard_contour_iou'])}  "
          f"纹理相关={form(s3['texture_corr'])}  病灶灰度变化={form(s3['lesion_mean_change'])}")
    print(f"  [表3] 病灶内平均绝对变化={form(s3['abs_change_in_lesion'])}  "
          f"病灶外脑区={form(s3['abs_change_in_brain_outside_lesion'])}")
    p1 = stats([r['prop1_affinity_residual'] for r in rows5]); p2 = stats([r['prop2_rel_dev'] for r in rows5])
    p3 = stats([r['prop3_lipschitz_measured'] for r in rows5]); p4 = stats([r['prop4_max_dev_from_linear'] for r in rows5])
    print(f"  [表5-命题1] 负荷仿射最大相对残差={p1['max']:.3e}")
    print(f"  [表5-命题2] 凸组合恒等式最大相对偏差={p2['max']:.3e}")
    print(f"  [表5-命题3] Lipschitz 实测={p3['max']:.9f}")
    print(f"  [表5-命题4] 与线性最大偏差（整数日）={p4['max']:.9f}  最大处刻度中位="
          f"{int(np.median([r['prop4_argmax_step'] for r in rows5]))}")
    print(f"  [表5] 权重单调性: {sum(r['monotone'] for r in rows5)}/{len(rows5)} 例成立")
    return s2, s3, rows5


# ----------------------------------------------------------------- 表4
def run_growth_ablation(ns, cases, grid_steps, seed, out, limit=10):
    laws = ['linear', 'exponential', 'gompertz', 'logistic_fkpp']
    spaces = ['volume', 'radius']
    rows = []
    for case in cases[:limit]:
        seed_all(ns, seed)
        with contextlib.redirect_stdout(io.StringIO()):
            seq, _ = ns['predict_tumor_progression'](case, grid_steps=grid_steps)
        source, target = seq[0:1], seq[-1:]
        mask = ns['_make_tumor_candidate_mask'](source)
        for law in laws:
            for space in spaces:
                if law == 'linear' and space == 'radius':
                    continue
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        _, rep = ns['temporal_lerp_weights'](source, target, mask, grid_steps,
                                                             law=law, space=space, verbose=False)
                except Exception as exc:
                    rows.append(dict(case=case.name, law=law, space=space, error=str(exc))); continue
                rows.append(dict(case=case.name, law=law, space=space, error='',
                                 dev_from_linear=float(rep['max_abs_deviation_from_linear']),
                                 volume_ratio=float(rep.get('volume_ratio', np.nan)),
                                 doubling_time_days=float(rep.get('doubling_time_days', np.nan)),
                                 affinity_residual=float(rep.get('burden_affinity_residual', np.nan))))
    write_csv(out / '表4_生长律消融_per_case.csv', rows)
    print('  [表4] 生长律 / 进度空间            与线性最大偏差   V(N)/V(0)   等效倍增时间(天)')
    for law in laws:
        for space in spaces:
            sub = [r for r in rows if r['law'] == law and r['space'] == space and not r['error']]
            if not sub:
                continue
            print(f"        {law+'/'+space:26s} {np.mean([r['dev_from_linear'] for r in sub]):13.6f}   "
                  f"{np.mean([r['volume_ratio'] for r in sub]):9.4f}   "
                  f"{np.mean([r['doubling_time_days'] for r in sub]):13.2f}")
    return rows


# ----------------------------------------------------------------- 表5-种子
def run_seed_check(ns, vae, unet, cases, grid_steps, seeds, out):
    sched = ns['LDMScheduler3D']()
    device = ns['device']
    rows = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        source = ns['_load_inference_volume'](case).unsqueeze(0).to(device)
        mask = ns['_make_tumor_candidate_mask'](source).cpu()
        core = (mask > 0.35)
        zs, vols, mets = [], [], []
        with torch.no_grad():
            z0 = vae.encode(source); cond = F.adaptive_avg_pool1d(z0.flatten(2), 1)
            cond = F.adaptive_avg_pool3d(z0, 1).flatten(1)
            for sd in seeds:
                torch.manual_seed(sd); z1 = sched.pass1_adani_infer(unet, z0, grid_steps, cond)
                torch.manual_seed(sd + 1); zf = sched.pass2_refine(unet, z1, z0, cond)
                zs.append(zf.cpu()); vols.append(vae.decode(zf).clamp(0, 1).cpu())
            for v in vols:
                with contextlib.redirect_stdout(io.StringIO()):
                    tgt = ns['_preserve_brain_structure'](source.cpu(), v, mask, 0.35)
                    mets.append(ns['compare_structure_quality'](source.cpu(), tgt))
        r = dict(case=case.name)
        for a in range(len(seeds)):
            for b in range(a + 1, len(seeds)):
                tag = f'{a}{b}'
                d = (zs[a] - zs[b]).abs()
                r[f'latent2_mean_{tag}'] = float(d.mean()); r[f'latent2_max_{tag}'] = float(d.max())
                dv = (vols[a] - vols[b]).abs()
                r[f'vol_mean_{tag}'] = float(dv.mean()); r[f'vol_max_{tag}'] = float(dv.max())
                r[f'metric_identical_{tag}'] = int(all(abs(mets[a][k] - mets[b][k]) == 0.0 for k in mets[0]))
                r[f'metric_maxdiff_{tag}'] = max(abs(mets[a][k] - mets[b][k]) for k in mets[0])
                r[f'sat_frac_core_{tag}'] = float((dv[core] > 0.08).float().mean())
        rows.append(r)
        if i % 10 == 0 or i == len(cases):
            print(f'  [表5-种子] {i}/{len(cases)}  累计 {time.time()-t0:.0f}s', flush=True)
    write_csv(out / '表5_三随机种子核验_per_case.csv', rows)
    pairs = [f'{a}{b}' for a in range(len(seeds)) for b in range(a + 1, len(seeds))]
    for tag in pairs:
        ok = sum(r[f'metric_identical_{tag}'] for r in rows)
        print(f"  [表5-种子] 种子对{tag}: 报告指标逐例一致 {ok}/{len(rows)}，"
              f"最大绝对差 {max(r[f'metric_maxdiff_{tag}'] for r in rows):.3e}")
    print(f"  [表5-种子] 第二阶段潜变量平均绝对差={form(stats([r[f'latent2_mean_{pairs[0]}'] for r in rows]))}，"
          f"最大={form(stats([r[f'latent2_max_{pairs[0]}'] for r in rows]),'.3f')}")
    print(f"  [表5-种子] 解码体数据平均绝对差={form(stats([r[f'vol_mean_{pairs[0]}'] for r in rows]),'.6g')}，"
          f"最大={form(stats([r[f'vol_max_{pairs[0]}'] for r in rows]))}")
    return rows


# ----------------------------------------------------------------- 表6 / 表7
def preserve_cfg(ns, source_vol, generated_vol, tumor_mask, change_strength=0.35,
                 clip=0.08, denoise=0.35, sharpen=0.24, lesion_red=0.22, lock_contour=True):
    """与 notebook 的 _preserve_brain_structure 逐位一致的可配置版本（用于开关消融）。"""
    src = source_vol.cpu(); src_smooth = F.avg_pool3d(src, 5, 1, 2); src_detail = src - src_smooth
    clean_detail = F.avg_pool3d(src_detail, 3, 1, 1)
    brain_mask = (src > 0.035).float(); lesion_mask = tumor_mask.cpu().clamp(0, 1)
    raw_res = generated_vol.cpu() - src
    res = raw_res if clip is None else raw_res.clamp(-clip, clip)
    res = res * lesion_mask * float(change_strength)
    tgt = (src * (1.0 - denoise) + src_smooth * denoise + res).clamp(0, 1)
    tgt = (tgt + sharpen * clean_detail * brain_mask).clamp(0, 1)
    tgt = (tgt - lesion_red * lesion_mask * (tgt - src_smooth).clamp_min(0)).clamp(0, 1)
    if lock_contour:
        tgt = torch.where(brain_mask > 0, tgt.clamp_min(0.04), src)
    return tgt, raw_res


def run_substitution_and_switches(ns, vae, unet, cases, grid_steps, seed, out):
    sched = ns['LDMScheduler3D'](); device = ns['device']
    def sample(z0, cond, s):
        torch.manual_seed(s); z1 = sched.pass1_adani_infer(unet, z0, grid_steps, cond)
        torch.manual_seed(s + 1); zf = sched.pass2_refine(unet, z1, z0, cond)
        return vae.decode(zf).clamp(0, 1)
    VAR = ['dit_seedA', 'dit_seedB', 'frozen', 'shuffled', 'identity', 'noise', 'mean_latent']
    SW = [('default', {}), ('no_clip', dict(clip=None)), ('clip_0.04', dict(clip=0.04)),
          ('clip_0.15', dict(clip=0.15)), ('no_detail', dict(sharpen=0.0)),
          ('no_expo_reduce', dict(lesion_red=0.0)), ('no_contour_lock', dict(lock_contour=False)),
          ('strength_0.10', dict(change_strength=0.10)), ('strength_1.00', dict(change_strength=1.0))]
    vrows, srows, rrows = [], [], []
    prev = None; t0 = time.time()
    for i, case in enumerate(cases, 1):
        source = ns['_load_inference_volume'](case).unsqueeze(0).to(device)
        mask = ns['_make_tumor_candidate_mask'](source)
        brain = (source.cpu() > 0.035); core = (mask.cpu() > 0.35); supp = (mask.cpu() > 0)
        with torch.no_grad():
            z0 = vae.encode(source); cond = F.adaptive_avg_pool3d(z0, 1).flatten(1)
            gen = {'dit_seedA': sample(z0, cond, seed), 'dit_seedB': sample(z0, cond, seed + 1),
                   'frozen': sample(z0, cond, 0), 'identity': source.clone(),
                   'noise': torch.rand_like(source),
                   'mean_latent': vae.decode(z0.mean(dim=(2, 3, 4), keepdim=True).expand_as(z0)).clamp(0, 1)}
        gen['shuffled'] = prev if prev is not None else gen['dit_seedA']; prev = gen['dit_seedA']
        for v in VAR:
            tgt, raw = preserve_cfg(ns, source, gen[v], mask)
            with contextlib.redirect_stdout(io.StringIO()):
                mm = ns['compare_structure_quality'](source.cpu(), tgt)
            vrows.append(dict(case=case.name, variant=v, soft_iou=mm['contour_iou'], hard_iou=mm['hard_contour_iou'],
                              tex_corr=mm['texture_correlation'], lesion_change=mm['lesion_mean_change'],
                              resid_mean_abs=float(raw.abs().mean()),
                              psnr_brain_target=float(-10 * np.log10(F.mse_loss(tgt[brain.expand_as(tgt)],
                                                                                source.cpu()[brain.expand_as(tgt)]).item()))))
        for name, kw in SW:
            tgt, raw = preserve_cfg(ns, source, gen['dit_seedA'], mask, **kw)
            with contextlib.redirect_stdout(io.StringIO()):
                mm = ns['compare_structure_quality'](source.cpu(), tgt)
            srows.append(dict(case=case.name, switch=name, soft_iou=mm['contour_iou'], hard_iou=mm['hard_contour_iou'],
                              tex_corr=mm['texture_correlation'], lesion_change=mm['lesion_mean_change'],
                              psnr_brain_target=float(-10 * np.log10(F.mse_loss(tgt[brain.expand_as(tgt)],
                                                                                source.cpu()[brain.expand_as(tgt)]).item()))))
        rawA = (gen['dit_seedA'].cpu() - source.cpu()).abs()
        rr = dict(case=case.name)
        for name, sel in [('vol', torch.ones_like(brain, dtype=torch.bool)), ('brain', brain),
                          ('support', supp), ('core', core)]:
            v = rawA[sel]
            rr[f'sat08_{name}'] = float((v > 0.08).float().mean()); rr[f'meanabs_{name}'] = float(v.mean())
        rrows.append(rr)
        if i % 10 == 0 or i == len(cases):
            print(f'  [表6/表7] {i}/{len(cases)}  累计 {time.time()-t0:.0f}s', flush=True)
    write_csv(out / '表6_替身对照_per_case.csv', vrows)
    write_csv(out / '表7_开关消融_per_case.csv', srows)
    write_csv(out / '图12_残差饱和口径_per_case.csv', rrows)
    print('  [表6] 生成模块设置                          Soft IoU      Hard IoU     纹理相关性    病灶灰度变化')
    for v in VAR:
        sub = [r for r in vrows if r['variant'] == v]
        print(f"        {v:38s} {form(stats([r['soft_iou'] for r in sub]))}  {form(stats([r['hard_iou'] for r in sub]))}"
              f"  {form(stats([r['tex_corr'] for r in sub]))}  {form(stats([r['lesion_change'] for r in sub]))}")
    print('  [表7] 配置                     Soft IoU      纹理相关性    脑区PSNR/dB')
    for name, _ in SW:
        sub = [r for r in srows if r['switch'] == name]
        print(f"        {name:22s} {form(stats([r['soft_iou'] for r in sub]))}  "
              f"{form(stats([r['tex_corr'] for r in sub]))}  {form(stats([r['psnr_brain_target'] for r in sub]),'.2f')}")
    print('  [图12] 裁剪前残差 |Δ|>0.08 比例（按分母）')
    for k, lab in [('vol', '整幅'), ('brain', '脑掩膜'), ('support', '掩膜非零支撑'), ('core', '候选核 mask>0.35')]:
        print(f"        {lab:16s} {form(stats([r[f'sat08_{k}'] for r in rrows]))}")
    return vrows, srows, rrows


# ----------------------------------------------------------------- 图
def make_figures(out, vrows, srows, rrows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    ORDER = ['dit_seedA', 'dit_seedB', 'frozen', 'shuffled', 'mean_latent', 'identity', 'noise']
    LABEL = {'dit_seedA': 'DiT (seed A)', 'dit_seedB': 'DiT (seed B)', 'frozen': 'frozen sampler',
             'shuffled': 'shuffled case', 'mean_latent': 'constant latent',
             'identity': 'identity (source copy)', 'noise': 'pure noise input'}
    METRICS = [('soft_iou', '(a) soft contour IoU'), ('hard_iou', '(b) hard contour IoU'),
               ('tex_corr', '(c) high-freq texture corr.'), ('lesion_change', '(d) lesion mean change')]
    def st(v, c):
        x = np.array([r[c] for r in vrows if r['variant'] == v], dtype=float)
        return x.mean(), x.std(ddof=1)
    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.85), dpi=400, gridspec_kw={'wspace': 0.34})
    for c, (col, lab) in enumerate(METRICS):
        ax = axes[c]; means = []
        for n in ORDER:
            m, s = st(n, col); means.append(m)
            y = ORDER.index(n)
            col_c = '#12805c' if n == 'identity' else ('#c0392b' if n == 'noise' else
                    ('#1f4e9c' if n == 'dit_seedA' else '#7d9dc4'))
            ax.errorbar([m], [y], xerr=[s], fmt='o', ms=5.0, lw=1.1, capsize=2.4, color=col_c,
                        markeredgecolor='white', markeredgewidth=0.5, zorder=3)
        means = np.array(means)
        ax.set_yticks(list(range(len(ORDER))))
        ax.set_yticklabels([LABEL[n] for n in ORDER] if c == 0 else [], fontsize=9)
        ax.set_ylim(6.7, -0.7); ax.set_title(lab, fontsize=10.5)
        ax.grid(axis='x', ls=':', lw=0.5, alpha=0.55)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        pad = (means.max() - means.min()) * 0.30 + 2e-4
        ax.set_xlim(means.min() - pad, means.max() + pad)
        if col == 'lesion_change':
            ax.axvline(0, color='k', lw=0.7, ls=':')
    fig.legend(handles=[Line2D([], [], color='#1f4e9c', marker='o', ls='none', ms=5, label='DiT sampling'),
                        Line2D([], [], color='#7d9dc4', marker='o', ls='none', ms=5, label='substitute generator'),
                        Line2D([], [], color='#12805c', marker='o', ls='none', ms=5, label='no generation'),
                        Line2D([], [], color='#c0392b', marker='o', ls='none', ms=5, label='pure noise')],
               fontsize=9, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.savefig(out / '图11_生成模块替身对照.png', bbox_inches='tight', facecolor='white'); plt.close(fig)

    dens = [('vol', 'whole volume'), ('brain', 'brain mask'), ('support', 'mask support\n(mask>0)'),
            ('core', 'candidate core\n(mask>0.35)')]
    fig2, ax2 = plt.subplots(figsize=(5.6, 2.9), dpi=400)
    xs = np.arange(len(dens))
    means = [np.mean([r[f'sat08_{k}'] for r in rrows]) for k, _ in dens]
    sds = [np.std([r[f'sat08_{k}'] for r in rrows], ddof=1) for k, _ in dens]
    ax2.bar(xs, means, yerr=sds, color=['#b8c6d9', '#5d8cc0', '#c9a227', '#1f4e9c'], capsize=4, width=0.62)
    for x, m in zip(xs, means):
        ax2.text(x, m + 0.025, f'{m*100:.2f}%', ha='center', fontsize=11)
    ax2.axhline(1.0, color='k', ls=':', lw=0.9)
    ax2.set_xticks(xs); ax2.set_xticklabels([l for _, l in dens], fontsize=10.5)
    ax2.set_ylabel('fraction with $|\\Delta|>0.08$', fontsize=11); ax2.set_ylim(0, 1.16)
    ax2.grid(axis='y', ls=':', lw=0.5, alpha=0.55)
    for sp in ('top', 'right'):
        ax2.spines[sp].set_visible(False)
    fig2.savefig(out / '图12_残差饱和口径.png', bbox_inches='tight', facecolor='white'); plt.close(fig2)
    print('  [图] 已保存 图11 / 图12 到', out)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=0, help='病例数上限，0=全部51例')
    ap.add_argument('--grid-steps', dest='grid_steps', type=int, default=100,
                    help='时间网格刻度数 N（生成条件，非临床随访时间）')
    ap.add_argument('--seed', type=int, default=20260925)
    ap.add_argument('--seeds', type=str, default='20260925,20260926,20260927')
    ap.add_argument('--notebook', type=Path, default=NB_DEFAULT)
    ap.add_argument('--ckpt', type=Path, default=CKPT_DEFAULT)
    ap.add_argument('--out', type=Path, default=HERE / '复算结果')
    ap.add_argument('--only', type=str, default='')
    ap.add_argument('--device', type=str, default='auto')
    a = ap.parse_args()

    device = torch.device('mps' if (a.device == 'auto' and torch.backends.mps.is_available())
                          else ('cuda' if (a.device == 'auto' and torch.cuda.is_available()) else 'cpu'))
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    print(f'设备={device}  notebook={a.notebook}  checkpoint={a.ckpt}')
    ns = load_notebook_ns(a.notebook, device)
    seeds = [int(s) for s in a.seeds.split(',')]
    all_cases = sorted(ns['_case_dirs'](ns['TEST_ROOT']))
    cases = all_cases[:a.cases] if a.cases else all_cases
    print(f'测试病例数={len(cases)}  时间网格刻度数 N={a.grid_steps}（生成条件，非临床随访时间）  主种子={a.seed}  种子核验={seeds}')
    vae, unet, _ = load_models(ns, a.ckpt, device)
    print(f'参数: 自编码器 {sum(p.numel() for p in vae.parameters())}  '
          f'DiT {sum(p.numel() for p in unet.parameters())}')

    only = [s.strip() for s in a.only.split(',') if s.strip()]
    def want(tag):
        return (not only) or any(tag.startswith(o) for o in only)

    t0 = time.time(); summary = dict(device=str(device), cases=len(cases), grid_steps=a.grid_steps, seed=a.seed, seeds=seeds)
    if want('表2') or want('表3') or want('表5-命题'):
        s2, s3, p5 = run_recon_progression(ns, vae, cases, a.grid_steps, a.seed, a.out)
        summary['表2'] = s2; summary['表3'] = s3
        summary['表5_命题'] = dict(
            prop1_max_affinity_residual=max(r['prop1_affinity_residual'] for r in p5),
            prop2_max_rel_dev=max(r['prop2_rel_dev'] for r in p5),
            prop3_lipschitz_measured=max(r['prop3_lipschitz_measured'] for r in p5),
            prop4_max_dev_from_linear=max(r['prop4_max_dev_from_linear'] for r in p5),
            prop4_argmax_step=int(np.median([r['prop4_argmax_step'] for r in p5])),
            monotone_cases=int(sum(r['monotone'] for r in p5)))
    if want('表4'):
        run_growth_ablation(ns, cases, a.grid_steps, a.seed, a.out)
    if want('表5-种子'):
        run_seed_check(ns, vae, unet, cases, a.grid_steps, seeds, a.out)
    if want('表6') or want('表7') or want('图'):
        vrows, srows, rrows = run_substitution_and_switches(ns, vae, unet, cases, a.grid_steps, a.seed, a.out)
        if want('图'):
            make_figures(a.out, vrows, srows, rrows)
    summary['total_seconds'] = round(time.time() - t0, 1)
    (a.out / '复算汇总.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n完成，总耗时 {summary['total_seconds']}s -> {a.out}")


if __name__ == '__main__':
    main()
