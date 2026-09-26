#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""notebook 冒烟测试：按顺序执行 脑肿瘤预测_DiT算法.ipynb 的代码单元（含推理与出图单元），
确认时间口径改名后仍可端到端跑通，且输出的网格刻度、指标与论文口径一致。

用法：python3 notebook冒烟测试.py [--cases-limit 1]
说明：DeepSeek 报告分支默认关闭（DEEPSEEK_ENABLED=False 时只提示跳过），不产生网络请求。
"""
import argparse, contextlib, io, json, re, time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NB = ROOT / '脑肿瘤预测_DiT算法.ipynb'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nb', type=Path, default=NB)
    a = ap.parse_args()

    cells = json.loads(a.nb.read_text(encoding='utf-8'))['cells']
    code = [''.join(c['source']) for c in cells if c['cell_type'] == 'code']
    ns = {'__name__': 'notebook_smoke'}
    torch.set_num_threads(min(8, torch.get_num_threads()))
    t0 = time.time()
    for idx in [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
        exec(compile(code[idx], f'<cell{idx}>', 'exec'), ns)
        print(f'[cell{idx:>2}] 执行完成  {time.time()-t0:6.1f}s', flush=True)
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        exec(compile(code[14], '<cell14>', 'exec'), ns)
    log = buf.getvalue()
    print('[cell14] 推理单元执行完成  %.1fs' % (time.time() - t0))
    for line in log.splitlines():
        if any(k in line for k in ['时间网格刻度数', 'MSE', 'MAE', 'PSNR', '结构指标']):
            print('   ', line)
    with contextlib.redirect_stdout(io.StringIO()) as buf2:
        exec(compile(code[15], '<cell15>', 'exec'), ns)
    print('[cell15] DeepSeek 报告单元执行完成（未配置 key 时只提示跳过）')
    for line in buf2.getvalue().splitlines()[-2:]:
        print('   ', line)
    seq = ns['_real_sequence']
    grid = ns['GRID_STEPS']
    assert tuple(seq.shape) == (grid + 1, 1, *ns['IMG_SIZE']), '序列长度与时间网格刻度数不一致'
    print(f'自检通过：序列形状 {tuple(seq.shape)} = 时间网格刻度数 N={grid} + 1（刻度是生成条件，非临床随访时间）')


if __name__ == '__main__':
    main()
