#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文 docx 与配图/标注的一致性工具（幂等，可反复运行）。

1) 配图：把 核验材料/论文配图 中重生成的结果写回 docx 对应 media 槽位，并逐张比对 md5；
2) 标注：校验正文与图注使用“时间网格第 k 个刻度（图中标注 Step k）”口径，
   不再出现日序标注；同时列出仍含“天”的句子（应只有生长律物理量 T_d/倍增时间）。
用法：
  python3 docx_配图写回与校验.py            # 只校验
  python3 docx_配图写回与校验.py --write    # 写回配图并校验
"""
import argparse
import hashlib
import os
import re
import zipfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DOC = ROOT / '脑转移瘤局部进展演化的两阶段潜空间扩散模型DiT.docx'
ASSETS = ROOT / '核验材料/论文配图'

MEDIA = {
    'word/media/image1.png': ASSETS / '新摘要配图_三例150刻度_轴向.png',   # 首页摘要配图
    'word/media/image7.png': ASSETS / 'Mets_020_step150.png',              # 图6(a)
    'word/media/image8.png': ASSETS / 'Mets_049_step150.png',              # 图6(b)
    'word/media/image9.png': ASSETS / 'Mets_042_step150.png',              # 图6(c)
    'word/media/image3.png': ASSETS / '图2_生长律权重曲线.png',            # 图2
    'word/media/image11.png': ASSETS / '图8_生长律中间帧与差异.png',       # 图8
    'word/media/image12.png': ASSETS / '图9_病灶负荷轨迹.png',             # 图9
    'word/media/image16.png': ASSETS / '图13_裁剪前残差口径与生成器旁路.png',  # 图13
    'word/media/image17.png': ASSETS / '图14_参数扫描临界曲线.png',          # 图14
}


def md5(data):
    return hashlib.md5(data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='把配图写回 docx')
    args = ap.parse_args()

    z = zipfile.ZipFile(DOC)
    names = z.namelist()
    doc = z.read('word/document.xml').decode('utf-8')
    bad = 0

    print('== 配图 ==')
    for slot, path in MEDIA.items():
        want = md5(path.read_bytes())
        got = md5(z.read(slot))
        flag = 'OK ' if want == got else '需写回'
        if want != got:
            bad += 1
        print(f'  {flag} {slot:24s} <- {path.name}  {want}')

    print('== 标注 ==')
    text = ''.join(re.findall(r'<w:t(?: [^>]*)?>(.*?)</w:t>', doc, re.S))
    for kw, ok in [('Day', text.count('Day') == 0), ('Step k', 'Step k' in text), ('时间网格', '时间网格' in text)]:
        print(f'  {"OK " if ok else "异常"} {kw}: {text.count(kw)}')
        bad += 0 if ok else 1
    print('  仍含“天”的句子（应仅为生长律物理量）：')
    for i, p in enumerate(re.findall(r'<w:p[ >].*?</w:p>', doc, re.S)):
        t = ''.join(re.findall(r'<w:t(?: [^>]*)?>(.*?)</w:t>', p, re.S))
        if '天' in t:
            print(f'    [{i}] {t[:80]}')

    # 图8 版面比例应与新图一致
    w, h = Image.open(MEDIA['word/media/image11.png']).size
    ext = re.findall(r'<wp:extent cx="(\d+)" cy="(\d+)"/>', doc)
    ok8 = any(abs(int(cy) / int(cx) - h / w) < 1e-4 for cx, cy in ext)
    print(f'  {"OK " if ok8 else "异常"} 图8 版面比例 {h}/{w}')
    bad += 0 if ok8 else 1

    if args.write:
        tmp = DOC.with_suffix('.tmp.docx')
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zo:
            for n in names:
                zo.writestr(n, MEDIA[n].read_bytes() if n in MEDIA else z.read(n))
        os.replace(tmp, DOC)
        print('已写回配图 ->', DOC)
    z.close()
    print('校验', '通过' if bad == 0 else f'有 {bad} 项待处理')
    return bad


if __name__ == '__main__':
    raise SystemExit(main())
