"""
重建流水线 Phase 1: 四井解析 -> 客观清洗 -> SHMAX/TOC/脆性 三目标训练表
============================================================================
特征: 只用原始测井测量曲线 (不用软件算的力学/储层解释成果, 避免"软件成果预测软件成果")
清洗: 哨兵(-9999/99990) + 物理边界 + 关键声波密度有效行
剔除标准全部客观、建模前冻结、可写进独立报告; 不看任何模型残差。
脆性列命名分裂(BRIT/BRITL/BRITS/BRITT), 本脚本查实四井覆盖与值域, 不武断合并。
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "clean"
OUT.mkdir(exist_ok=True)
WELLS = {
    'ZXX2': 'data/ZXX2/ZXX2井_曲线文本数据(4023-4292).txt',
    'ZXX3': 'data/ZXX3井数据/ZXX3_解释成果.txt',
    'ZXX7': 'data/ZXX7/ZXX7_解释成果.txt',
    'ZXX6': 'data/ZXX6/足216/ZXX6_解释成果.txt',
}
# 原始测井测量类特征候选 (仪器直接测量, 非软件解释成果)
# 去掉 CAL(ZXX3/7全缺) 与 KTH(覆盖差); 保留四井覆盖>93%的稳健特征
FEAT_CAND = ['AC','DTC','DTS','DEN','GR','CNL','LLD','LLS','K','TH','U','DEVI']
FEAT_BOUNDS = {  # 物理硬边界
    'AC':(20,500),'DTC':(20,250),'DTS':(40,600),'DEN':(1.5,3.3),'GR':(0,500),
    'CNL':(-15,90),'LLD':(0,5e4),'LLS':(0,5e4),'CAL':(4,20),'PEF':(0,20),'PE':(0,20),
    'K':(0,15),'KTH':(0,100),'TH':(0,60),'U':(0,60),'DEVI':(0,120),
}
TARGET_BOUNDS = {'SHMAX':(1,250),'TOC':(0,15),'PERM':(0,2000),'BRITL':(0,100)}
BRITTLE_VARS = ['BRIT','BRITL','BRITS','BRITT']  # 只查实, 不合并

def parse(path):
    lines = Path(path).read_text(encoding='latin-1').splitlines()
    cols, data = None, []
    for ln in lines:
        s = ln.strip()
        if cols is None and ('CURNAMES' in s or 'CURVENAME' in s):
            body = (s.split('CURNAMES')[-1] if 'CURNAMES' in s else s.split('CURVENAME')[-1]).lstrip('= ').strip()
            cols = [c.strip() for c in body.split(',') if c.strip()]
            continue
        if cols is None: continue
        parts = s.split()
        if len(parts) != len(cols)+1: continue
        try: data.append([float(x) for x in parts])
        except ValueError: continue
    df = pd.DataFrame(data, columns=['DEPTH']+cols)
    return df.mask((df <= -9999) | (df >= 99990))

parsed = {}
print("=== 各井解析 & 有效行 (关键声波密度 DTC/DTS/DEN 有效) ===")
for w, rel in WELLS.items():
    df = parse(ROOT/rel)
    n0 = len(df)
    for c in ['DTC','DTS','DEN']:
        if c in df: df = df[(df[c] > 0) | df[c].isna()]
    df = df.dropna(subset=[c for c in ['DTC','DTS','DEN'] if c in df]).reset_index(drop=True)
    parsed[w] = df
    print(f"  {w}: 原始 {n0} 行 -> 声波密度有效 {len(df)} 行")

# 公共特征
common_feat = [f for f in FEAT_CAND if all(f in parsed[w].columns for w in WELLS)]
# PE/PEF 别名归一
print(f"\n=== 四井公共原始测量特征 ({len(common_feat)}): {common_feat} ===")

print("\n=== 脆性列变体 四井覆盖 & 值域 (查实, 是否可当统一目标) ===")
print(f"{'井':<6}" + "".join(f"{v:>16}" for v in BRITTLE_VARS))
for w in WELLS:
    df = parsed[w]; row = f"{w:<6}"
    for v in BRITTLE_VARS:
        if v in df.columns and df[v].notna().sum() >= 50:
            y = df[v].dropna()
            row += f"{f'n={len(y)}[{y.min():.0f},{y.max():.0f}]':>16}"
        else:
            row += f"{'--':>16}"
    print(row)

print("\n=== 目标清洗后有效样本 & 剔除比例 (物理边界) ===")
print(f"{'井':<6}{'特征全有效行':>12}{'SHMAX':>18}{'TOC':>18}")
rows_out = {}
for w in WELLS:
    df = parsed[w].copy()
    feats = [f for f in common_feat]
    # 特征物理边界清洗
    for f in feats:
        lo, hi = FEAT_BOUNDS.get(f, (-np.inf, np.inf))
        df[f] = df[f].mask((df[f] < lo) | (df[f] > hi))
    feat_ok = df[feats].notna().all(axis=1)
    line = f"{w:<6}{int(feat_ok.sum()):>12}"
    keep_cols = ['DEPTH'] + feats
    for t,(lo,hi) in TARGET_BOUNDS.items():
        if t in df.columns:
            yt = df[t].mask((df[t] < lo) | (df[t] > hi))
            valid = feat_ok & yt.notna()
            df[t] = yt
            raw_n = df[t].notna().sum()
            line += f"{f'{int(valid.sum())} (剔{100*(1-valid.sum()/max(raw_n,1)):.0f}%)':>18}"
            if t not in keep_cols: keep_cols.append(t)
        else:
            line += f"{'缺该列':>18}"
    print(line)
    out = df[keep_cols].copy(); out.insert(0,'WELL',w)
    out.to_csv(OUT/f"{w}_clean.csv", index=False)
    rows_out[w] = out

print(f"\n干净训练表已写入: {OUT}/<井>_clean.csv (特征={len(common_feat)} + SHMAX/TOC)")
print("脆性目标待定列后再并入。")
