"""配对消融: 标准KAN(单尺度) vs FMS-KAN(多尺度 5->10->15)
均为 [12,10,4,1]、L-BFGS、相同总训练预算、相同数据划分 —— 唯一差别 = 单尺度 vs 多尺度refinement。
解决 issue#1 第4条(消融非配对)。
"""
import os, warnings, json, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from kan import KAN

SB = Path(__file__).resolve().parent
(SB / "_run").mkdir(exist_ok=True); os.chdir(SB / "_run")
D = SB / "clean"; WELLS = ['ZXX2', 'ZXX3', 'ZXX7', 'ZXX6']
FEATS = ['AC', 'DTC', 'DTS', 'DEN', 'GR', 'CNL', 'LLD', 'LLS', 'K', 'TH', 'U', 'DEVI']
WIDTH = [12, 10, 4, 1]; LOGT = {'PERM'}
big = pd.concat([pd.read_csv(D / f"{w}_clean.csv") for w in WELLS], ignore_index=True)
def r2(y, p): tt = ((y - y.mean())**2).sum(); return 1 - ((y - p)**2).sum() / tt

def std_kan(ds):   # 单尺度: 固定 grid=15, 不做 refinement; 总步数=100 与 FMS-KAN 对齐
    m = KAN(width=WIDTH, grid=15, k=3, seed=0, device='cpu')
    m.fit(ds, opt="LBFGS", steps=40, lamb=1e-4)
    m.fit(ds, opt="LBFGS", steps=60, lamb=1e-4)
    return m
def fms_kan(ds):   # 多尺度: 5 -> 10 -> 15 逐级细化; 总步数 40+30+30=100
    m = KAN(width=WIDTH, grid=5, k=3, seed=0, device='cpu')
    m.fit(ds, opt="LBFGS", steps=40, lamb=1e-4)
    for g in [10, 15]:
        m = m.refine(g); m.fit(ds, opt="LBFGS", steps=30, lamb=1e-4)
    return m

res = {}
for T in ['SHMAX', 'TOC', 'PERM']:
    sub = big.dropna(subset=FEATS + [T]); X = sub[FEATS].values; y0 = sub[T].values.astype(float)
    y = np.log1p(y0) if T in LOGT else y0
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, random_state=42)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=42)
    xs = StandardScaler().fit(Xtr); ys = StandardScaler().fit(ytr.reshape(-1, 1))
    Xtr_s, Xte_s = xs.transform(Xtr), xs.transform(Xte); ytr_z = ys.transform(ytr.reshape(-1, 1)).ravel()
    inv = lambda z: ys.inverse_transform(z.reshape(-1, 1)).ravel()
    back = lambda a: (np.expm1(a) if T in LOGT else a)
    ds = {'train_input': torch.tensor(Xtr_s, dtype=torch.float32),
          'train_label': torch.tensor(ytr_z.reshape(-1, 1), dtype=torch.float32),
          'test_input': torch.tensor(Xte_s, dtype=torch.float32),
          'test_label': torch.zeros(len(Xte_s), 1)}
    ms = std_kan(ds); mf = fms_kan(ds)
    with torch.no_grad():
        rs = r2(back(yte), back(inv(ms(ds['test_input']).numpy().ravel())))
        rf = r2(back(yte), back(inv(mf(ds['test_input']).numpy().ravel())))
    res[T] = {'std': round(float(rs), 4), 'fms': round(float(rf), 4), 'gain': round(float(rf - rs), 4)}
    print(f"{T}: 标准KAN(单尺度)={rs:.4f}  FMS-KAN(多尺度)={rf:.4f}  提升={rf-rs:+.4f}", flush=True)

res['mean'] = {'std': round(np.mean([res[t]['std'] for t in ['SHMAX','TOC','PERM']]), 4),
               'fms': round(np.mean([res[t]['fms'] for t in ['SHMAX','TOC','PERM']]), 4)}
res['mean']['gain'] = round(res['mean']['fms'] - res['mean']['std'], 4)
json.dump(res, open(SB / "ablation_paired.json", "w"), indent=2)
print(f"均值: 标准KAN={res['mean']['std']:.4f}  FMS-KAN={res['mean']['fms']:.4f}  提升={res['mean']['gain']:+.4f}", flush=True)
print("=== 配对消融完成 ===", flush=True)
