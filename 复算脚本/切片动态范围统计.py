# -*- coding: utf-8 -*-
"""统计 BrainMetShare 切片（本文使用的 t1_gd 模态）的灰度动态范围。

目的：论文 §3.1 需说明输入切片是否为逐片归一化——即各切片的灰度极值是否恒为满量程。
逐文件读取 PNG 极值（Image.getextrema），不修改任何数据。输出 复算结果/切片动态范围统计.json。
"""
import json
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path('~/Desktop/BRAINMETASTASIS').expanduser()
OUT = HERE / '复算结果' / '切片动态范围统计.json'


def scan(modality='t1_gd'):
    total = full = blank = 0
    others = []
    for split in ('train', 'test'):
        for case in sorted((DATA / split).glob('Mets_*')):
            for f in sorted((case / modality).glob('*.png')):
                lo, hi = Image.open(f).convert('L').getextrema()
                total += 1
                if (lo, hi) == (0, 255):
                    full += 1
                elif (lo, hi) == (0, 0):
                    blank += 1
                else:
                    others.append({'file': str(f.relative_to(DATA)), 'extrema': [lo, hi]})
    return {'modality': modality, 'root': str(DATA), 'total': total,
            'full_range': full, 'blank': blank, 'other': others,
            'full_range_fraction': full / total, 'blank_fraction': blank / total}


if __name__ == '__main__':
    rep = scan('t1_gd')
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"模态 {rep['modality']} | 切片总数 {rep['total']}")
    print(f"满量程(0,255) {rep['full_range']} | 全零空白片 {rep['blank']} | 其它 {len(rep['other'])}")
    print(f"满量程占比 {100*rep['full_range_fraction']:.4f}% | 空白片占比 {100*rep['blank_fraction']:.4f}%")
    print('报告 ->', OUT)
