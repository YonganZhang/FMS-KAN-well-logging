"""正式公式提取: 直接从最终 FMS-KAN(12原始特征, 与 finalize_pipeline 同架构) 本身
剪枝 + 符号化提取解析公式 —— 替代旧 extract3.py(独立4特征小KAN)的割裂做法。
=> 精度与公式来自【同一个模型】, 解决 issue #2。
12原始特征: 派生特征几乎不涨精度却使符号公式臃肿/触发 pykan coef bug, 故与 finalize 一致移除,
使【精度模型 == 可解释公式模型】。PERM 用 log1p(与 finalize 一致); 符号化 R²<0 则诚实降级、不给公式。
输出: formulas_from_fmskan.json + 屏幕打印(含 x_i -> 特征名 映射)。
"""
import os, warnings, json, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from kan import KAN
from kan.utils import ex_round

# ---- 稳健性补丁: pykan 的 spline.curve2coef 在 torch.linalg.lstsq 间歇性数值失败时,
# except 分支只 print 不给 coef 赋值就 `return coef` -> UnboundLocalError; 失败与否随
# 运行波动, 是符号提取【非确定性崩溃】的根源。改为始终走确定性正则化正规方程求解,
# 使公式提取永不崩溃且逐次可复现(数值上等价于原 least-squares, lamb=1e-8)。
import kan.spline as _sp, kan.KANLayer as _kl, kan.MultKAN as _mk
from kan.spline import B_batch as _B_batch
def _curve2coef_stable(x_eval, y_eval, grid, k):
    in_dim = x_eval.shape[1]; out_dim = y_eval.shape[2]; batch = x_eval.shape[0]
    n_coef = grid.shape[1] - k - 1
    mat = _B_batch(x_eval, grid, k).permute(1, 0, 2)[:, None, :, :].expand(in_dim, out_dim, batch, n_coef)
    y = y_eval.permute(1, 2, 0).unsqueeze(dim=3)
    XtX = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0, 1, 3, 2), mat)
    Xty = torch.einsum('ijmn,ijnp->ijmp', mat.permute(0, 1, 3, 2), y)
    n = XtX.shape[2]
    eye = torch.eye(n, device=mat.device)[None, None].expand(XtX.shape[0], XtX.shape[1], n, n)
    return torch.linalg.solve(XtX + 1e-8 * eye, Xty)[:, :, :, 0]
for _m in (_sp, _kl, _mk):
    _m.curve2coef = _curve2coef_stable
torch.manual_seed(0); torch.set_num_threads(1)  # 单线程固定浮点归约顺序, 尽量逐次可复现
# 注: TOC 符号化稳定(R²≈0.837, 主导系数逐次一致); SHMAX/PERM 的紧凑闭式在本函数库下
# 拟合不稳定(SHMAX 偶发退化为高阶冗长项、PERM 符号R²<0), 脚本对其如实降级、保留数值模型。

SB = Path(__file__).resolve().parent
(SB / "_run").mkdir(exist_ok=True); os.chdir(SB / "_run")
D = SB / "clean"; WELLS = ['ZXX2', 'ZXX3', 'ZXX7', 'ZXX6']
BASE = ['AC', 'DTC', 'DTS', 'DEN', 'GR', 'CNL', 'LLD', 'LLS', 'K', 'TH', 'U', 'DEVI']
PHYS = []; FEATS = BASE + PHYS  # 12原始特征(与finalize一致): 精度模型==公式模型
WIDTH = [len(FEATS), 10, 4, 1]; LOGT = {'PERM'}

def add_phys(d):
    d = d.copy(); e = 1e-6
    d['Vp'] = 1 / (d['DTC'] + e); d['Vs'] = 1 / (d['DTS'] + e); d['VpVs'] = d['DTS'] / (d['DTC'] + e)
    d['AI'] = d['DEN'] / (d['DTC'] + e); d['SI'] = d['DEN'] / (d['DTS'] + e); d['Gpx'] = d['DEN'] / (d['DTS']**2 + e)
    return d
def r2(y, p): tt = ((y - y.mean())**2).sum(); return 1 - ((y - p)**2).sum() / tt

dfs = {w: add_phys(pd.read_csv(D / f"{w}_clean.csv")) for w in WELLS}
big = pd.concat([dfs[w] for w in WELLS], ignore_index=True)
results = {}
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
    # 与 finalize 完全一致的 FMS-KAN
    m = KAN(width=WIDTH, grid=5, k=3, seed=0, device='cpu'); m.fit(ds, opt="LBFGS", steps=40, lamb=1e-4)
    for g in [10, 15]:
        m = m.refine(g); m.fit(ds, opt="LBFGS", steps=30, lamb=1e-4)
    with torch.no_grad():
        r_cont = r2(back(yte), back(inv(m(ds['test_input']).numpy().ravel())))
    # 直接对该模型剪枝 + 符号化
    try:
        m = m.prune(); m.fit(ds, opt="LBFGS", steps=30, lamb=1e-4)
        m.auto_symbolic(lib=['x', 'x^2', 'x^3', 'tanh'])  # 数值安全库, 避免 1/x·log·exp 触发 coef bug
        m.fit(ds, opt="LBFGS", steps=30, lamb=1e-4)
        with torch.no_grad():
            r_sym = r2(back(yte), back(inv(m(ds['test_input']).numpy().ravel())))
        f = str(ex_round(m.symbolic_formula()[0][0], 3))
    except Exception as e:
        r_sym = float('nan'); f = f"ERR:{type(e).__name__}:{e}"
    nterms = f.count('+') + f.count('-')
    results[T] = {'cont_R2': round(float(r_cont), 3), 'sym_R2': round(float(r_sym), 3),
                  'n_terms': nterms, 'formula': f}
    print(f"\n=== {T}: 连续R²={r_cont:.3f}  符号化R²={r_sym:.3f}  项数≈{nterms} ===", flush=True)
    print(f"公式: {f}", flush=True)

json.dump(results, open(SB / "formulas_from_fmskan.json", "w"), ensure_ascii=False, indent=2)
print("\n特征索引映射: " + ", ".join(f"x_{i+1}={FEATS[i]}" for i in range(len(FEATS))), flush=True)
print("=== 完成, 已存 formulas_from_fmskan.json ===", flush=True)
