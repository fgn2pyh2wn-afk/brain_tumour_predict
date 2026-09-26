#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""论文数字一致性核对：把 docx 论文里的每个定量结论，与代码复算产物逐项对齐。

论文文本取自 docx 的 <w:t> 节点（段落换行、单元格以 | 分隔）；复算产物取自：
  复算结果/复算汇总.json                （论文全表复算.py 的主输出）
  复算结果/表2_重建口径分解.json         （重建口径分解复算.py）
  复算结果/表3..表7_*.csv                （逐例明细）
  核验材料/P0_生成模块隔离/p0_*.csv       （替身对照 / 开关消融 / 残差口径 / 三随机种子）
  核验材料/指标口径/psnr_origin.json      （notebook 打印 PSNR 的口径来源）
  核验材料/训练/history_种子*.json        （表1 训练损失）

判定规则：论文中必须出现该字符串，且其数值与代码复算值之差不超过论文显示精度允许的容差。
用法：python3 论文数字一致性核对.py [--docx PATH] [--out PATH]
"""
import argparse, csv, html, json, math, re, statistics as st, zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / '复算结果'
MAT = ROOT / '核验材料'
DOCX_DEFAULT = ROOT / '脑转移瘤局部进展演化的两阶段潜空间扩散模型DiT.docx'

SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def paper_text(docx):
    raw = zipfile.ZipFile(docx).read('word/document.xml').decode('utf-8')
    # 只匹配 <w:t> / <w:t xml:space="preserve">，不得匹配 <w:tbl>/<w:tc>/<w:tr>
    txt = re.sub(r'<w:t(?: [^>]*)?>(.*?)</w:t>', lambda m: m.group(1), raw, flags=re.S)
    txt = txt.replace('</w:p>', '\n').replace('</w:tc>', ' | ')
    return html.unescape(txt)


def sci(x, sig=3):
    """1.9156e-07 → '1.92×10⁻⁷'（用论文里的上标写法）。"""
    if x == 0:
        return '0'
    mant, exp = f'{x:.{sig-1}e}'.split('e')
    e = int(exp)
    return f'{mant}×10{("⁻" if e < 0 else "")}{str(abs(e)).translate(SUP)}'


def mean_sd(vals, nd=4):
    m, s = st.mean(vals), st.stdev(vals)
    return (f'−{abs(m):.{nd}f}±{s:.{nd}f}') if m < 0 else f'{m:.{nd}f}±{s:.{nd}f}'


def pct_sd(vals, nd=2):
    return f'{100*st.mean(vals):.{nd}f}%±{100*st.stdev(vals):.{nd}f}%'


def parse_num(s):
    s = s.replace('−', '-').replace('≤', '').replace('%', '').strip()
    m = re.match(r'^-?\d+(\.\d+)?(×10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)?', s)
    if not m:
        return None
    head, exp = m.group(0), ''
    if '×10' in s:
        sup = s.split('×10')[1]
        digits = sup.translate(str.maketrans('⁻⁰¹²³⁴⁵⁶⁷⁸⁹', '-0123456789'))
        exp = digits
        head = s.split('×10')[0]
    return float(head) * (10 ** int(exp) if exp else 1)


def load_csv(p):
    with open(p, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def col(rows, key):
    return [float(r[key]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docx', type=Path, default=DOCX_DEFAULT)
    ap.add_argument('--out', type=Path, default=RES / '论文数字一致性核对.txt')
    args = ap.parse_args()

    PT = paper_text(args.docx)
    S = json.loads((RES / '复算汇总.json').read_text(encoding='utf-8'))
    DEC = json.loads((RES / '表2_重建口径分解.json').read_text(encoding='utf-8'))
    t3 = load_csv(RES / '表3_进展模拟_per_case.csv')
    t4 = load_csv(RES / '表4_生长律消融_per_case.csv')
    t5 = load_csv(RES / '表5_命题核验_per_case.csv')
    t6 = load_csv(MAT / 'P0_生成模块隔离' / 'p0_variants_full.csv')
    t7 = load_csv(MAT / 'P0_生成模块隔离' / 'p0_switches_full.csv')
    resid = load_csv(MAT / 'P0_生成模块隔离' / 'p0_residual_full.csv')
    seeds = load_csv(MAT / 'P0_生成模块隔离' / 'p0_seedcheck_full.csv')
    psnr0 = json.loads((MAT / '指标口径' / 'psnr_origin.json').read_text(encoding='utf-8'))
    h25 = json.loads((MAT / '训练' / 'history_种子20260925.json').read_text(encoding='utf-8'))['history']['ldm']
    h26 = json.loads((MAT / '训练' / 'history_种子20260926.json').read_text(encoding='utf-8'))['history']['ldm']
    mask_cmp = load_csv(RES / '掩膜与专家标注对比_per_case.csv')
    slice_rng = json.loads((RES / '切片动态范围统计.json').read_text(encoding='utf-8'))
    PILOT = MAT / 'VAE重训试点'
    pil_rec = load_csv(PILOT / 'pilot_recon.csv')
    pil_aud = load_csv(PILOT / 'pilot_audit.csv')
    pil_loc = load_csv(PILOT / 'pilot_localization.csv')
    dedup = json.loads((RES / '切片去重与口径核验.json').read_text(encoding='utf-8'))
    timing = json.loads((RES / '推理耗时实测.json').read_text(encoding='utf-8'))

    P, F = [], []

    def rec(tag, paper_str, ok, src, note=''):
        (P if ok else F).append((tag, paper_str, src, note))

    def chk_str(tag, paper_str, src):
        """论文里必须出现（用于文本型结论）。"""
        rec(tag, paper_str, paper_str in PT, src)

    def chk(tag, paper_str, value, tol, src):
        """字符串出现 且 数值与复算值一致（tol 为论文显示精度允许的容差）。"""
        parsed = parse_num(paper_str)
        ok = paper_str in PT and parsed is not None and abs(parsed - value) <= tol
        rec(tag, paper_str, ok, src, f'代码 {value:.6g}')

    def chk_pct_sd(tag, vals, nd, src):
        """百分比形式的 均值±标准差（论文写法由复算值直接生成）。"""
        ps = pct_sd(vals, nd)
        rec(tag, ps, ps in PT, src, f'代码 {ps}')

    def chk_sd(tag, vals, nd, src):
        """均值±标准差：字符串由复算值直接生成，须与论文写法完全一致。"""
        ps = mean_sd(vals, nd)
        rec(tag, ps, ps in PT, src, f'代码 {ps}')

    # ------------------------------------------------------------- 表1 训练损失
    for i, v in enumerate(h25, 1):
        chk(f'表1 第{i}轮 DiT 损失（种子20260925）', f'{v:.4f}', v, 5e-5,
            '训练/history_种子20260925.json')
    chk('§3.3 种子20260926 起止损失', f'{h26[0]:.4f}', h26[0], 5e-5, '训练/history_种子20260926.json')
    chk('§3.3 种子20260926 末轮损失', f'{h26[-1]:.4f}', h26[-1], 5e-5, '训练/history_种子20260926.json')

    # ------------------------------------------------------------- 表2 重建
    m2 = S['表2']
    chk_sd('表2 整幅 MSE', None, 0, '') if False else None
    ps = f"{m2['mse']['mean']:.6f}±{m2['mse']['std']:.6f}"
    rec('表2 整幅 MSE', ps, ps in PT, '复算汇总.json 表2.mse', f"代码 {ps}")
    ps = f"{m2['mae']['mean']:.6f}±{m2['mae']['std']:.6f}"
    rec('表2 整幅 MAE', ps, ps in PT, '复算汇总.json 表2.mae', f"代码 {ps}")
    ps = f"{m2['psnr_db']['mean']:.3f}±{m2['psnr_db']['std']:.3f}"
    rec('表2 整幅 PSNR', ps, ps in PT, '复算汇总.json 表2.psnr_db', f"代码 {ps}")
    chk('§4.2 逐例 PSNR 下界', f"{m2['psnr_db']['min']:.2f}", m2['psnr_db']['min'], 5e-3, '复算汇总.json')
    chk('§4.2 逐例 PSNR 上界', f"{m2['psnr_db']['max']:.2f}", m2['psnr_db']['max'], 5e-3, '复算汇总.json')

    # ------------------------------------------------------------- §4.2 口径分解
    chk('§4.2 背景 MSE', sci(DEC['bg_mse']['mean'], 2), DEC['bg_mse']['mean'], 5e-8,
        '表2_重建口径分解.json bg_mse')
    chk('§4.2 脑区 MSE', f"{DEC['brain_mse']['mean']:.3f}", DEC['brain_mse']['mean'], 5e-4,
        '表2_重建口径分解.json brain_mse')
    chk('§4.2 脑区 PSNR', f"{DEC['brain_psnr_db_from_mean_mse']:.1f} dB",
        DEC['brain_psnr_db_from_mean_mse'], 0.05, '表2_重建口径分解.json')
    chk('§4.2 脑区体素占比', f"{100*DEC['brain_frac_of_volume']['mean']:.1f}%",
        100 * DEC['brain_frac_of_volume']['mean'], 0.05, '表2_重建口径分解.json')
    cp = [r['change_psnr_style'] for r in psnr0]
    cm = [r['change_mse'] for r in psnr0]
    chk_str('§4.2 代码打印 PSNR 区间', f'{min(cp):.1f}–{max(cp):.1f} dB', '指标口径/psnr_origin.json')
    chk_str('§4.2 末帧-源帧切片 MSE 区间', f'{min(cm)*1e4:.1f}–{max(cm)*1e4:.1f}×10⁻⁴',
            '指标口径/psnr_origin.json')

    # ------------------------------------------------------------- 表3 进展模拟
    chk_sd('表3 Soft IoU', col(t3, 'soft_contour_iou'), 4, '表3_进展模拟_per_case.csv')
    chk_sd('表3 Hard IoU', col(t3, 'hard_contour_iou'), 4, '表3_进展模拟_per_case.csv')
    chk_sd('表3 纹理相关性', col(t3, 'texture_corr'), 4, '表3_进展模拟_per_case.csv')
    chk_sd('表3 病灶区平均灰度变化', col(t3, 'lesion_mean_change'), 4, '表3_进展模拟_per_case.csv')
    chk_sd('表3 病灶内平均绝对变化', col(t3, 'abs_change_in_lesion'), 4, '表3_进展模拟_per_case.csv')
    chk_sd('表3 病灶外脑区平均绝对变化', col(t3, 'abs_change_in_brain_outside_lesion'), 4,
           '表3_进展模拟_per_case.csv')
    ratio = st.mean([float(r['abs_change_in_brain_outside_lesion']) / float(r['abs_change_in_lesion']) for r in t3])
    chk('表3 病灶内外变化比', f'{ratio:.3f}', ratio, 5e-4, '表3_进展模拟_per_case.csv')
    frac = st.mean([float(r['n_core']) / float(r['n_brain']) for r in resid])
    chk('表3 候选核占脑区比例', f'{100*frac:.1f}%', 100 * frac, 0.05, 'p0_residual_full.csv')

    # ------------------------------------------------------------- 表4 生长律消融
    def g(law, space):
        return [r for r in t4 if r['law'] == law and r['space'] == space]
    chk('表4 指数/体积 最大偏差', f"{st.mean(col(g('exponential','volume'),'dev_from_linear')):.4f}",
        st.mean(col(g('exponential', 'volume'), 'dev_from_linear')), 5e-5, '表4_生长律消融_per_case.csv')
    chk('表4 指数/半径 最大偏差', f"{st.mean(col(g('exponential','radius'),'dev_from_linear')):.4f}",
        st.mean(col(g('exponential', 'radius'), 'dev_from_linear')), 5e-5, '表4_生长律消融_per_case.csv')
    chk('表4 logistic/体积 最大偏差', f"{st.mean(col(g('logistic_fkpp','volume'),'dev_from_linear')):.4f}",
        st.mean(col(g('logistic_fkpp', 'volume'), 'dev_from_linear')), 5e-5, '表4_生长律消融_per_case.csv')
    chk('表4 体积比 V(100)/V(0)', f"{st.mean(col(g('exponential','volume'),'volume_ratio')):.3f}",
        st.mean(col(g('exponential', 'volume'), 'volume_ratio')), 5e-4, '表4_生长律消融_per_case.csv')
    chk('表4 logistic 体积比', f"{st.mean(col(g('logistic_fkpp','volume'),'volume_ratio')):.3f}",
        st.mean(col(g('logistic_fkpp', 'volume'), 'volume_ratio')), 5e-4, '表4_生长律消融_per_case.csv')
    chk('表4 等效倍增时间', f"{st.mean(col(g('exponential','volume'),'doubling_time_days')):.1f}",
        st.mean(col(g('exponential', 'volume'), 'doubling_time_days')), 0.05, '表4_生长律消融_per_case.csv')
    chk('表4 logistic 等效倍增时间', f"{st.mean(col(g('logistic_fkpp','volume'),'doubling_time_days')):.1f}",
        st.mean(col(g('logistic_fkpp', 'volume'), 'doubling_time_days')), 0.05, '表4_生长律消融_per_case.csv')

    # ------------------------------------------------------------- 表5 命题核验
    N, TD = 100.0, 45.0
    L = (2**(N/TD) - 2**((N-1)/TD)) / (2**(N/TD) - 1)
    lstar = TD * math.log2(TD * (2**(N/TD) - 1) / (N * math.log(2)))
    steps = np.arange(N + 1)
    w_of_t = lambda t: (2**(t/TD) - 1) / (2**(N/TD) - 1)
    w64 = w_of_t(steps)
    w32 = w64.astype(np.float32)
    dev64 = float(np.abs(w64 - steps / N).max())
    dev32 = float(np.abs(w32 - steps.astype(np.float32) / np.float32(N)).max())
    p1 = max(col(t5, 'prop1_affinity_residual'))
    p2 = max(col(t5, 'prop2_rel_dev'))
    p3 = max(col(t5, 'prop3_lipschitz_measured'))
    chk('命题1 最大相对残差', sci(p1, 3), p1, 5e-17, '表5_命题核验_per_case.csv')
    chk('命题2 最大相对偏差', sci(p2, 3), p2, 5e-9, '表5_命题核验_per_case.csv')
    chk('命题3 闭式 Lipschitz 常数', f'{L:.9f}', L, 5e-10, '解析式 L=(2^(N/Td)−2^((N−1)/Td))/(2^(N/Td)−1)')
    chk('命题3 实测 Lipschitz 常数', f'{p3:.9f}', p3, 5e-10, '表5_命题核验_per_case.csv')
    chk('命题3 相对偏差', sci((p3 - L) / L, 3), (p3 - L) / L, 5e-9, '表5_命题核验_per_case.csv')
    chk('命题4 整数刻度最大偏差（双精度）', f'{dev64:.9f}', dev64, 5e-10, '解析式 w(t)（float64）')
    chk('命题4 单精度实测最大偏差', f'{dev32:.9f}', dev32, 5e-10, '解析式 w(t)（float32）')
    chk('命题4 最大偏差位置 t*', f'{lstar:.2f}', lstar, 5e-3, '解析式 t*=Td·log₂[Td(2^(N/Td)−1)/(N·ln2)]')

    # ------------------------------------------------------------- 表5 三随机种子
    pairs = ['AB', 'AC', 'BC']
    chk_str('种子核验 指标逐位一致（51/51）', '51/51逐位一致', 'p0_seedcheck_full.csv')
    rec('种子核验 指标逐例最大差 0', '0.0',
        all(float(r[f'metric_maxdiff_{p}']) == 0.0 for r in seeds for p in pairs) and '最大绝对差0.0' in PT,
        'p0_seedcheck_full.csv')
    ps = mean_sd(col(seeds, 'latent2_mean_AB'), 4)
    rec('种子核验 潜变量平均绝对差', ps, ps in PT, 'p0_seedcheck_full.csv（AB 对）', f'代码 {ps}')
    ps = f"{st.mean(col(seeds,'latent2_max_AB')):.3f}±{st.stdev(col(seeds,'latent2_max_AB')):.3f}"
    rec('种子核验 潜变量最大绝对差', ps, ps in PT, 'p0_seedcheck_full.csv（AB 对）', f'代码 {ps}')
    vm, vs = st.mean(col(seeds, 'vol_mean_AB')), st.stdev(col(seeds, 'vol_mean_AB'))
    ps = f'{sci(vm, 5)}±{sci(vs, 3)}'
    rec('种子核验 解码体数据平均绝对差', ps, ps in PT, 'p0_seedcheck_full.csv（AB 对）', f'代码 {ps}')
    ps = f"{st.mean(col(seeds,'vol_max_AB')):.4f}±{st.stdev(col(seeds,'vol_max_AB')):.4f}"
    rec('种子核验 解码体数据最大绝对差', ps, ps in PT, 'p0_seedcheck_full.csv（AB 对）', f'代码 {ps}')
    rec('种子核验 裁剪后末帧逐位相同', '0（逐位相同）',
        all(float(r[f'target_max_{p}']) == 0.0 for r in seeds for p in pairs) and '0（逐位相同）' in PT,
        'p0_seedcheck_full.csv')

    # ------------------------------------------------------------- 表5/§4.5 残差口径
    ps = pct_sd(col(resid, 'fr_gt008_vol'))
    rec('残差饱和 整幅口径', ps, ps in PT, 'p0_residual_full.csv', f'代码 {ps}')
    ps = pct_sd(col(resid, 'fr_gt008_brain'))
    rec('残差饱和 脑掩膜口径', ps, ps in PT, 'p0_residual_full.csv', f'代码 {ps}')
    ps = pct_sd(col(resid, 'fr_gt008_support'))
    rec('残差饱和 候选支撑口径', ps, ps in PT, 'p0_residual_full.csv', f'代码 {ps}')
    ps = pct_sd(col(resid, 'fr_gt008_core'))
    rec('残差饱和 候选核口径', ps, ps in PT, 'p0_residual_full.csv', f'代码 {ps}')
    chk_sd('裁剪前残差平均绝对值 脑掩膜', col(resid, 'meanabs_brain'), 4, 'p0_residual_full.csv')
    chk_sd('裁剪前残差平均绝对值 候选支撑', col(resid, 'meanabs_support'), 4, 'p0_residual_full.csv')
    chk_sd('裁剪前残差平均绝对值 候选核', col(resid, 'meanabs_core'), 4, 'p0_residual_full.csv')

    # ------------------------------------------------------------- 表6 替身对照
    core = ['dit_seedA', 'dit_seedB', 'frozen', 'shuffled', 'mean_latent']
    for k in ['soft_iou', 'hard_iou', 'tex_corr', 'lesion_change']:
        vals = {v: col([r for r in t6 if r['variant'] == v], k) for v in core}
        same = all(np.array_equal(vals['dit_seedA'], vals[v]) for v in core[1:])
        ps = mean_sd(vals['dit_seedA'], 4)
        rec(f'表6 前五种情形逐位一致（{k}）', ps, bool(same and ps in PT),
            'p0_variants_full.csv', '逐位相同' if same else '存在差异')
    for v, name in [('identity', '不生成（源图副本）'), ('noise', '纯噪声输入')]:
        rows6 = [r for r in t6 if r['variant'] == v]
        for k in ['soft_iou', 'hard_iou', 'tex_corr', 'lesion_change']:
            chk_sd(f'表6 {name} {k}', col(rows6, k), 4, 'p0_variants_full.csv')

    # ------------------------------------------------------------- 表7 开关消融
    have = {r['switch'] for r in t7}
    for key, label in [('default', '默认（全部开启）'),
                       ('no_clip', '关闭残差裁剪'), ('clip_0.04', '裁剪阈值0.04'),
                       ('clip_0.15', '裁剪阈值0.15'), ('no_detail', '关闭细节回补'),
                       ('no_expo_reduce', '关闭曝光压制'), ('no_contour_lock', '关闭轮廓锁定'),
                       ('strength_0.10', '演化强度0.10'), ('strength_1.00', '演化强度1.00')]:
        if key not in have:
            continue
        rows7 = [r for r in t7 if r['switch'] == key]
        for k, nd in [('soft_iou', 4), ('hard_iou', 4), ('tex_corr', 4), ('lesion_change', 4),
                      ('psnr_brain_target', 2), ('ssim_brain_target', 4)]:
            chk_sd(f'表7 {label} {k}', col(rows7, k), nd, 'p0_switches_full.csv')
    # 表7 的“不生成（源图副本）”一行来自替身对照
    rows_id = [r for r in t6 if r['variant'] == 'identity']
    for k, nd in [('soft_iou', 4), ('hard_iou', 4), ('tex_corr', 4), ('lesion_change', 4),
                  ('psnr_brain_target', 2), ('ssim_brain_target', 4)]:
        chk_sd(f'表7 不生成（源图副本） {k}', col(rows_id, k), nd, 'p0_variants_full.csv')

    # ------------------------------------------------------------- 命题5 / M2 生成模块隔离审计
    pre = load_csv(MAT / 'P0_生成模块隔离' / 'p0_preclip_full.csv')
    sweep = load_csv(MAT / 'P0_生成模块隔离' / 'p0_sweep_full.csv')
    byp = load_csv(MAT / 'P0_生成模块隔离' / 'p0_bypass_check.csv')
    vae = load_csv(MAT / 'P0_生成模块隔离' / 'p0_vae_saturation.csv')
    indep = load_csv(MAT / 'P0_生成模块隔离' / 'p0_gen_independence.csv')

    def sweep_rows(s, c):
        key = 'inf' if c is None else str(c)
        return [r for r in sweep if abs(float(r['strength']) - s) < 1e-9 and r['clip'] == key]

    chk('§4.6 裁剪前残差均值(全幅)', '0.0614±0.0110', st.mean(col(pre, 'pre_mean_vol')), 5e-5, 'p0_preclip_full.csv')
    chk('图13(a) p99.9(全幅)', '0.7220', st.mean(col(pre, 'pre_p99.9_vol')), 5e-5, 'p0_preclip_full.csv')
    chk('图13(a) 上确界(全幅)', '0.8620', st.mean(col(pre, 'pre_max_vol')), 5e-5, 'p0_preclip_full.csv')
    chk('图13(a) 脑掩膜均值', '0.3627±0.0563', st.mean(col(pre, 'pre_mean_brain')), 5e-5, 'p0_preclip_full.csv')
    chk('图13(a) 候选核均值', '0.5231±0.0884', st.mean(col(pre, 'pre_mean_core')), 5e-5, 'p0_preclip_full.csv')
    chk('图13(b) 丢弃能量(c=0.08,脑)', '77.6%', 100 * (1 - st.mean(col(pre, 'kept_energy_frac_c0.08_brain'))), 0.05, 'p0_preclip_full.csv')
    chk('图13(c) 裁剪前PSNR(全幅)', '15.993±1.497', st.mean(col(pre, 'psnr_preclip_clamped_vol')), 5e-4, 'p0_preclip_full.csv')
    chk('图13(c) 裁剪前SSIM(全幅)', '0.725±0.021', st.mean(col(pre, 'ssim_preclip_vol')), 5e-4, 'p0_preclip_full.csv')
    chk('图13(c) 后处理PSNR(全幅)', '38.67±0.97', st.mean(col(pre, 'psnr_final_vol')), 5e-3, 'p0_preclip_full.csv')
    chk('图13(c) 后处理SSIM(全幅)', '0.9953±0.0006', st.mean(col(pre, 'ssim_final_vol')), 5e-5, 'p0_preclip_full.csv')
    chk('图13(c) 后处理PSNR(脑)', '30.93±0.90', st.mean(col(pre, 'psnr_final_brain')), 5e-3, 'p0_preclip_full.csv')
    chk('图13(c) 后处理SSIM(脑)', '0.9791±0.0022', st.mean(col(pre, 'ssim_final_brain')), 5e-5, 'p0_preclip_full.csv')
    chk('图13(c) 后处理PSNR(核)', '28.62±0.93', st.mean(col(pre, 'psnr_final_core')), 5e-3, 'p0_preclip_full.csv')
    chk('图13(c) 后处理SSIM(核)', '0.9723±0.0051', st.mean(col(pre, 'ssim_final_core')), 5e-5, 'p0_preclip_full.csv')
    chk('命题5 端点差均值', '0.1872±0.0195', st.mean(col(pre, 'delta_end_vol_max')), 5e-5, 'p0_preclip_full.csv')
    chk('命题5 端点差最大', '0.2452', max(col(pre, 'delta_end_vol_max')), 5e-5, 'p0_preclip_full.csv')
    chk('命题5 端点差最小', '0.1541', min(col(pre, 'delta_end_vol_max')), 5e-5, 'p0_preclip_full.csv')
    adani = load_csv(MAT / 'P0_生成模块隔离' / 'p0_adani_tstart_sweep.csv')
    adani_raw = [max(float(r['gen_maxdiff_050_020']), float(r['gen_maxdiff_080_020'])) for r in adani]
    adani_out = [max(float(r['out_maxdiff_050_020']), float(r['out_maxdiff_080_020'])) for r in adani]
    chk('§2.4 AdaNI档位 原始生成图最大差(均值)', '0.1134', st.mean(adani_raw), 5e-5, 'p0_adani_tstart_sweep.csv')
    chk('§2.4 AdaNI档位 原始生成图最大差(最大)', '0.1166', max(adani_raw), 5e-5, 'p0_adani_tstart_sweep.csv')
    chk('§2.4 AdaNI档位 最终输出最大差', '0.000', max(adani_out), 5e-5, 'p0_adani_tstart_sweep.csv')
    d0 = sweep_rows(0.35, 0.08)
    chk('图14 默认贡献(脑均值)', '0.0067', st.mean(col(d0, 'contrib_mean_brain')), 5e-5, 'p0_sweep_full.csv')
    chk('图14 默认上确界均值', '0.0215', st.mean(col(d0, 'contrib_max_vol')), 5e-5, 'p0_sweep_full.csv')
    chk('图14 默认上确界最大', '0.0250', max(col(d0, 'contrib_max_vol')), 5e-5, 'p0_sweep_full.csv')
    d16 = sweep_rows(0.35, 0.16)
    chk('图14 c=0.16 贡献', '0.0132', st.mean(col(d16, 'contrib_mean_brain')), 5e-5, 'p0_sweep_full.csv')
    chk('图14 c=0.16 上确界', '0.0430', st.mean(col(d16, 'contrib_max_vol')), 5e-5, 'p0_sweep_full.csv')
    dinf = sweep_rows(0.35, None)
    chk('图14 无裁剪贡献', '0.0363', st.mean(col(dinf, 'contrib_mean_brain')), 5e-5, 'p0_sweep_full.csv')
    chk('图14 无裁剪上确界', '0.1924', st.mean(col(dinf, 'contrib_max_vol')), 5e-5, 'p0_sweep_full.csv')
    chk('图14 无裁剪上确界最大', '0.2624', max(col(dinf, 'contrib_max_vol')), 5e-5, 'p0_sweep_full.csv')
    chk('图13(d) 候选核饱和率', '99.91%', 100 * st.mean(col(byp, 'sat_frac_mask')), 0.02, 'p0_bypass_check.csv')
    chk('图13(d) 脑掩膜饱和率', '95.99%', 100 * st.mean(col(byp, 'sat_frac_brain')), 0.02, 'p0_bypass_check.csv')
    for name, shown in [('dit_seedB', '0.183'), ('frozen', '0.182'), ('mean_latent', '0.243'), ('other_case', '0.004')]:
        chk(f'图13(d) 原始生成图最大差 {name}', shown, st.mean(col(byp, f'raw_maxdiff_{name}')), 5e-4, 'p0_bypass_check.csv')
    chk_str('图13(d) 裁剪后最大差为0', '裁剪后候选核内的最大差均为0.000', 'p0_bypass_check.csv')
    chk('§4.6 潜变量幅值均值', '11.43±1.39', st.mean(col(vae, 'latent_absmean')), 5e-3, 'p0_vae_saturation.csv')
    chk('§4.6 解码器logits均值', '231.88', st.mean(col(vae, 'logit_absmean')), 0.02, 'p0_vae_saturation.csv')
    chk('§4.6 Sigmoid饱和体素占比', '99.92%', 100 * st.mean(col(vae, 'logit_frac_gt6')), 0.02, 'p0_vae_saturation.csv')
    chk('§4.6 解码输出均值', '0.0000', st.mean(col(vae, 'gen_mean')), 5e-5, 'p0_vae_saturation.csv')
    chk('§4.6 解码输出标准差', '0.0012', st.mean(col(vae, 'gen_std')), 5e-5, 'p0_vae_saturation.csv')
    chk('§4.6 生成图两两平均差', '0.0000', st.mean(col(indep, 'gen_meanabs')), 5e-5, 'p0_gen_independence.csv')
    chk('§4.6 生成图两两最大差', '0.010', max(col(indep, 'gen_maxabs')), 5e-4, 'p0_gen_independence.csv')
    chk('§4.6 潜变量两两平均差', '2.11', st.mean(col(indep, 'latent_meanabs')), 5e-3, 'p0_gen_independence.csv')

    # ------------------------------------------- §4.3/§5 候选核 vs 专家标注
    core_frac = col(mask_cmp, 'core_brain_frac')
    seg_frac = col(mask_cmp, 'seg_brain_frac')
    vol_ratio = [float(r['core_voxels']) / max(1.0, float(r['seg_voxels_tri'])) for r in mask_cmp]
    chk_pct_sd('§4.3 候选核占脑区比例', core_frac, 2, '掩膜与专家标注对比_per_case.csv')
    chk_pct_sd('§4.3 专家病灶占脑区比例', seg_frac, 2, '掩膜与专家标注对比_per_case.csv')
    chk('§4.3 候选核占比范围下界', '19.31%', 100 * min(core_frac), 0.01, '掩膜与专家标注对比_per_case.csv')
    chk('§4.3 候选核占比范围上界', '29.96%', 100 * max(core_frac), 0.01, '掩膜与专家标注对比_per_case.csv')
    chk_sd('§4.3 Dice（候选核 vs 专家标注）', col(mask_cmp, 'dice_tri'), 4, '掩膜与专家标注对比_per_case.csv')
    chk_sd('§4.3 精确率（候选核 vs 专家标注）', col(mask_cmp, 'precision_tri'), 4, '掩膜与专家标注对比_per_case.csv')
    chk_sd('§4.3 召回（候选核 vs 专家标注）', col(mask_cmp, 'recall_tri'), 3, '掩膜与专家标注对比_per_case.csv')
    chk_sd('§4.3 Dice（最大池化最宽松参照）', col(mask_cmp, 'dice_max'), 4, '掩膜与专家标注对比_per_case.csv')
    chk_sd('§4.3 召回（最大池化最宽松参照）', col(mask_cmp, 'recall_max'), 3, '掩膜与专家标注对比_per_case.csv')
    chk('§4.3 候选核/专家病灶 体积倍数中位数', '619', float(np.median(vol_ratio)), 0.5, '掩膜与专家标注对比_per_case.csv')
    chk('§4.3 体积倍数范围下界', '7.5', min(vol_ratio), 0.05, '掩膜与专家标注对比_per_case.csv')
    chk('§4.3 体积倍数范围上界', '12887', max(vol_ratio), 0.5, '掩膜与专家标注对比_per_case.csv')

    # ------------------------------------------- §3.1 切片动态范围（逐片满量程）
    chk('§3.1 t1_gd 切片总数', '23400', slice_rng['total'], 0.5, '切片动态范围统计.json')
    chk('§3.1 满量程切片数', '20844', slice_rng['full_range'], 0.5, '切片动态范围统计.json')
    chk('§3.1 满量程占比', '89.1%', 100 * slice_rng['full_range_fraction'], 0.05, '切片动态范围统计.json')
    chk('§3.1 全零空白片数', '2556', slice_rng['blank'], 0.5, '切片动态范围统计.json')

    # ---------------------------------- §4.7 自编码器退化的根因诊断与修复验证（试点）
    pil_new = [r for r in pil_rec if r['model'] == 'pilot_after']
    chk_sd('§4.7 修复试点 整幅重建PSNR', col(pil_new, 'psnr_db'), 3, 'VAE重训试点/pilot_recon.csv')
    chk('§4.7 修复试点 解码输出标准差', '0.1447', st.mean(col(pil_new, 'rec_std')), 5e-5,
        'VAE重训试点/pilot_recon.csv')
    chk('§4.7 修复试点 候选核内裁剪饱和比例', '34.98%',
        100 * st.mean(col(pil_aud, 'sat_core_frozen')), 5e-3, 'VAE重训试点/pilot_audit.csv')
    chk('§4.7 后处理后候选核内替身最大差', '0.0433',
        st.mean(col(pil_aud, 'fin_diff_core_pairwise_max')), 5e-5, 'VAE重训试点/pilot_audit.csv')
    chk('§4.7 修复后完整DiT核内裁剪前残差', '0.1329', st.mean(col(pil_loc, 'raw_core_full')), 5e-5,
        'VAE重训试点/pilot_localization.csv')
    chk('§4.7 修复后完整DiT环带裁剪前残差', '0.1343', st.mean(col(pil_loc, 'raw_ring_full')), 5e-5,
        'VAE重训试点/pilot_localization.csv')
    chk('§4.7 修复后完整DiT核内/环带对比度', '0.990', st.mean(col(pil_loc, 'raw_contrast_full')), 5e-5,
        'VAE重训试点/pilot_localization.csv')
    chk('§4.7 自编码器往返核内/环带对比度', '1.4117', st.mean(col(pil_loc, 'raw_contrast_frozen')), 5e-5,
        'VAE重训试点/pilot_localization.csv')
    chk('§4.7 随机潜变量核内/环带对比度', '1.7064', st.mean(col(pil_loc, 'raw_contrast_noise')), 5e-5,
        'VAE重训试点/pilot_localization.csv')
    chk('§4.7 源图本身后处理核内/环带对比度', '1.6794', st.mean(col(pil_loc, 'pp_contrast_src')), 1e-4,
        'VAE重训试点/pilot_localization.csv')
    chk('§4.7 源图本身后处理核内平均变化', '0.0245', st.mean(col(pil_loc, 'pp_core_src')), 5e-5,
        'VAE重训试点/pilot_localization.csv')
    chk('§4.7 完整DiT后处理核内平均变化', '0.0315', st.mean(col(pil_loc, 'pp_core_full')), 5e-5,
        'VAE重训试点/pilot_localization.csv')

    # ------------------------------------- §3.1 切片级去重核验（内容哈希）
    dd, cal = dedup['dedup'], dedup['caliber']
    chk('§3.1 纳入哈希的t1_gd切片数', '23400', dd['n_slices_hashed'], 0.5, '切片去重与口径核验.json')
    chk('§3.1 跨集合同名对哈希交集上界', '1/150', dd['same_name_overlap_max'], 0.5, '切片去重与口径核验.json')
    chk('§3.1 集内训练相邻对哈希交集上界', '1/150', dd['within_train_overlap_max'], 0.5, '切片去重与口径核验.json')
    chk('§3.1 集内测试相邻对哈希交集上界', '1/150', dd['within_test_overlap_max'], 0.5, '切片去重与口径核验.json')
    chk('§3.1 参与比对的病例对数', '205', dd['total_pairs'], 0.5, '切片去重与口径核验.json')
    chk('§3.1 出现整体重复的病例对数', '0', dd['pairs_with_full_overlap'], 0.5, '切片去重与口径核验.json')

    # ------------------------------------- §3.2 推理耗时实测（与机器相关）
    chk('§3.2 CPU整幅重建耗时/s', '4.7', timing['cpu']['recon_median_s'], 0.06, '推理耗时实测.json')
    chk('§3.2 CPU进展模拟耗时/s', '26.1', timing['cpu']['sim_median_s'], 0.06, '推理耗时实测.json')
    chk('§3.2 MPS整幅重建耗时/s', '2.2', timing['mps']['recon_median_s'], 0.06, '推理耗时实测.json')
    chk('§3.2 MPS进展模拟耗时/s', '15.0', timing['mps']['sim_median_s'], 0.06, '推理耗时实测.json')
    chk('§3.2 MPS合计耗时/s', '17', timing['mps']['total_median_s'], 0.55, '推理耗时实测.json')

    # ------------------------------------- §3.3 重建指标口径（整幅/逐片/合并MSE）
    chk('§3.3 整幅与逐片MSE最大差', '4.9×10⁻⁹', cal['equal_check_max_abs_diff'], 5e-11,
        '切片去重与口径核验.json')
    chk('§3.3 内容抽取切片MSE/整幅', '1.99', cal['informative_over_volume_ratio'], 5e-3,
        '切片去重与口径核验.json')
    chk('§3.3 内容抽取切片PSNR', '13.13', cal['psnr_informative_slice_mean'], 5e-3,
        '切片去重与口径核验.json')
    chk('§3.3 合并MSE换算PSNR', '15.760', cal['psnr_pooled'], 5e-4, '切片去重与口径核验.json')
    chk('§3.3 两种PSNR口径之差', '0.233', cal['psnr_pooled_gap_db'], 5e-4, '切片去重与口径核验.json')

    # ------------------------------------- §4.6 逐例跨情形极差（口径统一为逐例）
    vv = load_csv(MAT / 'P0_生成模块隔离' / 'p0_variants_full.csv')
    byc = {}
    for r in vv: byc.setdefault(r['case'], {})[r['variant']] = r
    for mname, ps, tag in (('soft_iou', '0.0034', 'Soft IoU'), ('hard_iou', '0.0048', 'Hard IoU'),
                           ('tex_corr', '0.0188', '纹理相关性'), ('lesion_change', '0.0129', '病灶区灰度变化')):
        rg = [max(float(d[k][mname]) for k in d) - min(float(d[k][mname]) for k in d) for d in byc.values()]
        chk(f'§4.6 逐例跨7情形极差 {tag}', ps, float(np.mean(rg)), 5e-5, 'p0_variants_full.csv')

    # ------------------------------------- 文本型修正（图注 / 定义口径）
    chk_str('图4 题注 层位间距', '相差5个层位', 'BRAINMETASTASIS/train/Mets_030/t1_gd 文件名 z131 与 z136')
    chk_str('§4.4 FKPP等效倍增时间为回归估计', '由其模拟体积曲线的对数—线性回归估计得到',
            'dit_nb.py doubling_time_days = ln2/斜率')

    # ------------------------------------------------------------- 输出
    lines = [f'论文数字一致性核对（paper ↔ code）',
             f'论文：{args.docx}',
             f'结果：PASS {len(P)} 项，FAIL {len(F)} 项',
             '=' * 108]
    for tag, ps, src, note in P:
        lines.append(f'[PASS] {tag:34s} 论文“{ps}”  ← {src}' + (f'  [{note}]' if note else ''))
    for tag, ps, src, note in F:
        lines.append(f'[FAIL] {tag:34s} 论文“{ps}”  ← {src}' + (f'  [{note}]' if note else ''))
    report = '\n'.join(lines)
    args.out.write_text(report, encoding='utf-8')
    print(report)
    print(f'\n报告 -> {args.out}')
    return 1 if F else 0


if __name__ == '__main__':
    raise SystemExit(main())
