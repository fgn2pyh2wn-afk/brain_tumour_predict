#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 脑肿瘤预测_DiT算法.ipynb 的代码单元导出为可 import/exec 的模块文件（带 CELL 分隔标记），
供 复算脚本/ 与 核验材料/ 下的复算脚本复用同一份 notebook 代码。

用法：python3 导出notebook模块.py [--nb 路径] [--out 路径]
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nb', type=Path, default=ROOT / '脑肿瘤预测_DiT算法.ipynb')
    ap.add_argument('--out', type=Path, default=ROOT / '核验材料/P0_生成模块隔离/dit_nb.py')
    a = ap.parse_args()
    cells = json.loads(a.nb.read_text(encoding='utf-8'))['cells']
    parts = []
    for i, cell in enumerate(cells):
        if cell['cell_type'] != 'code':
            continue
        parts.append(f"# ===== CELL {i} =====\n" + ''.join(cell['source']))
    a.out.write_text('\n'.join(parts) + '\n', encoding='utf-8')
    print(f'导出 {len(parts)} 个代码单元 -> {a.out}')


if __name__ == '__main__':
    main()
