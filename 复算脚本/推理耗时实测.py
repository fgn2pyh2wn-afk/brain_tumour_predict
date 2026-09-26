#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""推理耗时实测（论文 §3.2 的支撑产物）：51 例测试集的整幅重建 + N=100 刻度进展模拟。

同一流程分别在 CPU 与 MPS 后端各测两轮，取中位数。注意：墙钟时间与机器负载相关，
不同机器上的绝对值会变化，本产物只用于说明本机量级。

输出：复算结果/推理耗时实测.json
"""
import json, os, random, re, statistics as st, time
os.environ.setdefault('MPLBACKEND', 'Agg')
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / '复算结果'
MODULE = ROOT / '核验材料' / 'P0_生成模块隔离' / 'dit_nb.py'
SRC = MODULE.read_text(encoding='utf-8').split('# ===== CELL 14 =====')[0]
CKPT = Path('/Users/chris/Desktop/brain_tumor_ldm.pt')


def build(devname):
    src = re.sub(r"device = torch\.device\([^\n]*\)", f"device = torch.device('{devname}')", SRC, count=1)
    NS = {}; exec(compile(src, str(MODULE), 'exec'), NS)
    dev = NS['device']
    torch.manual_seed(20260925); random.seed(20260925); np.random.seed(20260925)
    ck = torch.load(CKPT, map_location='cpu', weights_only=True)
    vae = NS['SimpleVAE3D']().to(dev); vae.load_state_dict(ck['vae']); vae.eval()
    unet = NS['LatentUNet3D']().to(dev); unet.load_state_dict(ck['unet']); unet.eval()
    return NS, dev, vae, unet, NS['LDMScheduler3D'](), NS['_case_dirs'](NS['TEST_ROOT'])


def main():
    out = {}
    for devname in ('cpu', 'mps'):
        if devname == 'mps' and not torch.backends.mps.is_available():
            continue
        NS, dev, vae, unet, sched, cases = build(devname)

        def rec(c):
            x = NS['_load_png_volume'](c, modality='t1_gd').unsqueeze(0).to(dev)
            with torch.no_grad():
                return x, vae.decode(vae.encode(x)).clamp(0, 1)

        def sim(c, N):
            x = NS['_load_png_volume'](c, modality='t1_gd').unsqueeze(0).to(dev)
            with torch.no_grad():
                z0 = vae.encode(x); cond = F.adaptive_avg_pool3d(z0, 1).flatten(1)
                z1 = sched.pass1_adani_infer(unet, z0, N, cond)
                zf = sched.pass2_refine(unet, z1, z0, cond)
                g = vae.decode(zf).clamp(0, 1)
            NS['_preserve_brain_structure'](x, g, NS['_make_tumor_candidate_mask'](x), 0.35)

        rec(cases[0]); sim(cases[0], 100)          # 预热
        rec_t, sim_t = [], []
        for _ in range(2):
            t0 = time.perf_counter()
            for c in cases: rec(c)
            rec_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            for c in cases: sim(c, 100)
            sim_t.append(time.perf_counter() - t0)
        out[devname] = {'cases': len(cases), 'grid_steps': 100,
                        'recon_s': rec_t, 'sim_s': sim_t,
                        'recon_median_s': st.median(rec_t), 'sim_median_s': st.median(sim_t),
                        'total_median_s': st.median(rec_t) + st.median(sim_t)}
        print(f"{devname}: 重建 {out[devname]['recon_median_s']:.1f}s | 进展模拟(N=100) "
              f"{out[devname]['sim_median_s']:.1f}s | 合计 {out[devname]['total_median_s']:.1f}s", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / '推理耗时实测.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('产物 ->', RES / '推理耗时实测.json')


if __name__ == '__main__':
    main()
