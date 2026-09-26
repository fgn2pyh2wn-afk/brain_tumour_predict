#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重建口径分解复算：核对论文 §4.2 的口径描述（整幅 / 脑区 / 背景）。

在 51 例测试集上，用 notebook 里定义的自编码器与它实际加载的 checkpoint
（默认 /Users/chris/Desktop/brain_tumor_ldm.pt）做一次重建，分别统计：
  整幅体数据  MSE / MAE / PSNR
  脑区（源体数据 > 0.035 的体素）MSE / MAE / PSNR
  背景（脑区补集）MSE / MAE
以及脑区体素占比，供论文数字一致性核对脚本引用。

输出：复算结果/表2_重建口径分解.json
用法：python3 重建口径分解复算.py [--cases N] [--ckpt PATH] [--notebook PATH] [--out DIR]
"""
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
HELPER = HERE / '论文全表复算.py'


def load_helper():
    spec = importlib.util.spec_from_file_location('paper_recompute', HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=int, default=0, help='病例数上限，0=全部51例')
    ap.add_argument('--notebook', type=Path, default=HERE.parent / '脑肿瘤预测_DiT算法.ipynb')
    ap.add_argument('--ckpt', type=Path, default=Path('/Users/chris/Desktop/brain_tumor_ldm.pt'),
                    help='notebook 实际加载的 checkpoint（桌面版）')
    ap.add_argument('--out', type=Path, default=HERE / '复算结果')
    a = ap.parse_args()

    mod = load_helper()
    device = torch.device('cpu')
    torch.set_num_threads(min(8, torch.get_num_threads()))
    a.out.mkdir(parents=True, exist_ok=True)
    print(f'设备={device}  checkpoint={a.ckpt}')
    ns = mod.load_notebook_ns(a.notebook, device)
    vae, _, _ = mod.load_models(ns, a.ckpt, device)

    cases = sorted(ns['_case_dirs'](ns['TEST_ROOT']))
    if a.cases:
        cases = cases[:a.cases]

    vol_mse, vol_mae, brain_mse, brain_mae, bg_mse, bg_mae, frac = [], [], [], [], [], [], []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        vol = ns['_load_png_volume'](case).to(device)
        with torch.no_grad():
            rec = vae.decode(vae.encode(vol.unsqueeze(0))).clamp(0, 1)[0]
        m = (vol > 0.035)                                     # 脑区掩膜（与论文口径一致）
        bg = ~m
        vol_mse.append(F.mse_loss(rec, vol).item())
        vol_mae.append(F.l1_loss(rec, vol).item())
        brain_mse.append(F.mse_loss(rec[m], vol[m]).item())
        brain_mae.append(F.l1_loss(rec[m], vol[m]).item())
        bg_mse.append(F.mse_loss(rec[bg], vol[bg]).item())
        bg_mae.append(F.l1_loss(rec[bg], vol[bg]).item())
        frac.append(m.float().mean().item())
        if i % 10 == 0 or i == len(cases):
            print(f'  {i}/{len(cases)}  累计 {time.time()-t0:.0f}s', flush=True)

    def st(v):
        v = np.asarray(v, dtype=np.float64)
        return dict(mean=float(v.mean()), std=float(v.std(ddof=1)), min=float(v.min()), max=float(v.max()), n=int(v.size))

    def psnr(mse_mean):
        return float(-10.0 * np.log10(mse_mean)) if mse_mean > 0 else float('inf')

    s_vol_mse, s_brain_mse, s_bg_mse = st(vol_mse), st(brain_mse), st(bg_mse)
    out = dict(
        device=str(device), cases=len(cases), checkpoint=str(a.ckpt),
        notebook=str(a.notebook),
        brain_mask_rule='源体数据 > 0.035',
        vol_mse=s_vol_mse, vol_mae=st(vol_mae),
        vol_psnr_db_mean=float(np.mean([-10.0 * np.log10(x) for x in vol_mse])),
        brain_mse=s_brain_mse, brain_mae=st(brain_mae),
        brain_psnr_db_from_mean_mse=psnr(s_brain_mse['mean']),
        bg_mse=s_bg_mse, bg_mae=st(bg_mae),
        brain_frac_of_volume=st(frac),
        total_seconds=round(time.time() - t0, 1))
    (a.out / '表2_重建口径分解.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"整幅 MSE={s_vol_mse['mean']:.6f}±{s_vol_mse['std']:.6f}  "
          f"PSNR(逐例均值)={out['vol_psnr_db_mean']:.3f} dB")
    print(f"脑区 MSE={s_brain_mse['mean']:.6f}  -> {out['brain_psnr_db_from_mean_mse']:.2f} dB  "
          f"占整幅体素 {100*st(frac)['mean']:.1f}%")
    print(f"背景 MSE={s_bg_mse['mean']:.3e}")
    print(f"耗时 {out['total_seconds']}s -> {a.out / '表2_重建口径分解.json'}")


if __name__ == '__main__':
    main()
