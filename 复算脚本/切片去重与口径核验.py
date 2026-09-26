#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""切片级去重核验 + 重建指标口径核验（论文 §3.1 / §3.3 的支撑产物）。

一、去重：对全部 t1_gd 切片做内容哈希，比较跨集合同名病例对与集合内部相邻病例对的
    切片哈希交集，用于排除同一影像跨集合重复。
二、口径：比较整幅体数据 MSE 与逐切片 MSE 的等权平均（应严格相等），并量化代码演示单元
    中"按内容挑选信息量最大切片"的偏置；另给出"先逐例算 PSNR 再平均"与"按合并 MSE 换算"之差。

输出：复算结果/切片去重与口径核验.json
用法：python3 切片去重与口径核验.py
"""
import hashlib, json, re, sys
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / '复算结果'
DATA = Path('/Users/chris/Desktop/BRAINMETASTASIS')
MODULE = ROOT / '核验材料' / 'P0_生成模块隔离' / 'dit_nb.py'


def case_dirs(root):
    return sorted(p for p in Path(root).iterdir() if p.is_dir())


def slice_hashes(root, case):
    return {f.name: hashlib.md5(np.asarray(Image.open(f)).tobytes()).hexdigest()
            for f in sorted((Path(root) / case / 't1_gd').glob('*.png'))}


def dedup_check():
    tr, te = case_dirs(DATA / 'train'), case_dirs(DATA / 'test')
    Htr = {c.name: slice_hashes(DATA / 'train', c.name) for c in tr}
    Hte = {c.name: slice_hashes(DATA / 'test', c.name) for c in te}
    n_pair = min(len(tr), len(te))
    same = [len(set(Htr[f'Mets_{i:03d}'].values()) & set(Hte[f'Mets_{i:03d}'].values()))
            for i in range(1, n_pair + 1)]
    wtr = [len(set(Htr[f'Mets_{i:03d}'].values()) & set(Htr[f'Mets_{i+1:03d}'].values()))
           for i in range(1, len(tr))]
    wte = [len(set(Hte[f'Mets_{i:03d}'].values()) & set(Hte[f'Mets_{i+1:03d}'].values()))
           for i in range(1, len(te))]
    return {
        'n_slices_hashed': int(sum(len(v) for v in Htr.values()) + sum(len(v) for v in Hte.values())),
        'slices_per_case': 150,
        'same_name_pairs': len(same), 'same_name_overlap_mean': float(np.mean(same)),
        'same_name_overlap_max': int(max(same)),
        'within_train_pairs': len(wtr), 'within_train_overlap_mean': float(np.mean(wtr)),
        'within_train_overlap_max': int(max(wtr)),
        'within_test_pairs': len(wte), 'within_test_overlap_mean': float(np.mean(wte)),
        'within_test_overlap_max': int(max(wte)),
        'pairs_with_full_overlap': int(sum(1 for v in same + wtr + wte if v == 150)),
        'total_pairs': len(same) + len(wtr) + len(wte),
    }


def caliber_check():
    src = MODULE.read_text(encoding='utf-8').split('# ===== CELL 14 =====')[0].replace(
        "device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))",
        "device = torch.device('cpu')")
    NS = {}
    exec(compile(src, str(MODULE), 'exec'), NS)
    import random, torch, torch.nn.functional as F
    torch.manual_seed(20260925); random.seed(20260925); np.random.seed(20260925)
    ck = torch.load(Path('/Users/chris/Desktop/brain_tumor_ldm.pt'), map_location='cpu', weights_only=True)
    vae = NS['SimpleVAE3D'](); vae.load_state_dict(ck['vae']); vae.eval()
    cases = NS['_case_dirs'](NS['TEST_ROOT'])
    rows = []
    with torch.no_grad():
        for c in cases:
            x = NS['_load_png_volume'](c, modality='t1_gd').unsqueeze(0)
            r = vae.decode(vae.encode(x)).clamp(0, 1)
            vol_mse = F.mse_loss(r, x).item()
            per_slice = [F.mse_loss(r[0, :, :, k], x[0, :, :, k]).item() for k in range(x.shape[-1])]
            si = NS['_select_informative_slice'](x[0])
            rows.append({'case': c.name, 'vol_mse': vol_mse,
                         'mean_all_slice_mse': float(np.mean(per_slice)),
                         'informative_slice_mse': float(per_slice[si]), 'slice_index': int(si)})
    v = np.array([r['vol_mse'] for r in rows])
    a = np.array([r['mean_all_slice_mse'] for r in rows])
    s = np.array([r['informative_slice_mse'] for r in rows])
    ps_case = -10 * np.log10(v)
    ps_inf = -10 * np.log10(s)
    return {
        'cases': len(rows),
        'vol_mse_mean': float(v.mean()), 'vol_mse_std': float(v.std(ddof=1)),
        'mean_all_slice_mse_mean': float(a.mean()),
        'equal_check_max_abs_diff': float(np.abs(v - a).max()),
        'informative_slice_mse_mean': float(s.mean()),
        'informative_over_volume_ratio': float(s.mean() / v.mean()),
        'psnr_per_case_mean': float(ps_case.mean()), 'psnr_per_case_std': float(ps_case.std(ddof=1)),
        'psnr_pooled': float(-10 * np.log10(v.mean())),
        'psnr_pooled_gap_db': float(ps_case.mean() + 10 * np.log10(v.mean())),
        'psnr_informative_slice_mean': float(ps_inf.mean()),
        'per_case': rows,
    }


def main():
    RES.mkdir(parents=True, exist_ok=True)
    out = {'dedup': dedup_check(), 'caliber': caliber_check()}
    (RES / '切片去重与口径核验.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    d, c = out['dedup'], out['caliber']
    print('=== 切片去重 ===')
    print(f"哈希切片 {d['n_slices_hashed']} | 跨集合同名对 {d['same_name_pairs']} 对 交集均值 "
          f"{d['same_name_overlap_mean']:.2f}/150 最大 {d['same_name_overlap_max']}")
    print(f"集内相邻(训练) {d['within_train_overlap_mean']:.2f}/150 | 集内相邻(测试) "
          f"{d['within_test_overlap_mean']:.2f}/150 | 全重复病例对 {d['pairs_with_full_overlap']}/{d['total_pairs']}")
    print('=== 重建口径 ===')
    print(f"整幅MSE {c['vol_mse_mean']:.6f} | 逐片MSE等权平均 {c['mean_all_slice_mse_mean']:.6f} | 最大差 "
          f"{c['equal_check_max_abs_diff']:.3e}")
    print(f"内容挑选切片MSE {c['informative_slice_mse_mean']:.6f} = 整幅的 {c['informative_over_volume_ratio']:.2f} 倍")
    print(f"逐例PSNR均值 {c['psnr_per_case_mean']:.3f}±{c['psnr_per_case_std']:.3f} | 合并MSE换算 "
          f"{c['psnr_pooled']:.3f} | 差 {c['psnr_pooled_gap_db']:.3f} dB | 逐片PSNR "
          f"{c['psnr_informative_slice_mean']:.2f}")
    print('产物 ->', RES / '切片去重与口径核验.json')


if __name__ == '__main__':
    main()
