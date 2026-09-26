#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重跑 脑肿瘤预测_DiT算法.ipynb 的推理与出图单元，把 cell 14 的显示输出刷新为
"Step k（时间网格刻度）"口径的真实运行结果（含两张图与标准输出）。

用法：python3 刷新notebook输出.py [--nb 路径] [--clear]
      --clear 只清空输出，不重跑。
"""
import argparse, base64, contextlib, io, json, time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NB = ROOT / '脑肿瘤预测_DiT算法.ipynb'
RUN_CELLS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nb', type=Path, default=NB)
    ap.add_argument('--clear', action='store_true', help='只清空全部输出')
    a = ap.parse_args()

    nb = json.loads(a.nb.read_text(encoding='utf-8'))
    if a.clear:
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
        a.nb.write_text(json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')
        print('已清空全部单元输出 ->', a.nb)
        return

    cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    captured, log = [], io.StringIO()
    original_show = plt.show

    def show(*args, **kwargs):
        for number in plt.get_fignums():
            figure = plt.figure(number)
            buffer = io.BytesIO()
            figure.savefig(buffer, format='png', dpi=100)
            captured.append(buffer.getvalue())
        plt.close('all')

    plt.show = show
    ns = {'__name__': 'notebook_refresh'}
    torch.set_num_threads(min(8, torch.get_num_threads()))
    t0 = time.time()
    try:
        for idx in RUN_CELLS:
            if idx == 14:                      # 仅捕获 cell 14 自身的标准输出与图
                with contextlib.redirect_stdout(log):
                    exec(compile(cells[idx], f'<cell{idx}>', 'exec'), ns)
                print(f'[cell{idx}] 执行完成 {time.time()-t0:.1f}s | 捕获图 {len(captured)} 张')
            else:
                with contextlib.redirect_stdout(io.StringIO()) as _discard:
                    exec(compile(cells[idx], f'<cell{idx}>', 'exec'), ns)
    finally:
        plt.show = original_show

    outputs = [{'name': 'stdout', 'output_type': 'stream',
                'text': log.getvalue().splitlines(keepends=True)}]
    for image in captured:
        outputs.append({'data': {'image/png': base64.b64encode(image).decode('ascii'),
                                 'text/plain': ['<Figure size 2560x900 with 24 Axes>']},
                        'metadata': {}, 'output_type': 'display_data'})
    nb['cells'][14]['outputs'] = outputs
    nb['cells'][14]['execution_count'] = 18
    a.nb.write_text(json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + '\n', encoding='utf-8')
    print('已刷新 cell 14 输出 ->', a.nb)
    for line in log.getvalue().splitlines():
        if any(k in line for k in ('时间网格刻度数', 'MSE', 'MAE', 'PSNR', '结构指标', '本次随机测试病例')):
            print('   ', line)


if __name__ == '__main__':
    main()
