"""
KAN 公式提取 (概念验证: SHMAX)
==================================================
特征重要性选主控变量 -> 小KAN训练 -> 剪枝 -> 符号化 -> 解析公式。
报告: top特征 / 符号化前后 R² / 提取的公式(标准化空间, 附特征映射)。
"""
import os, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import torch
from kan import KAN
from kan.utils import ex_round

SB=Path(__file__).resolve().parent
(SB/"_run").mkdir(exist_ok=True); os.chdir(SB/"_run")
D=SB/"clean"; WELLS=['ZXX2','ZXX3','ZXX7','ZXX6']
FEATS=['AC','DTC','DTS','DEN','GR','CNL','LLD','LLS','K','TH','U','DEVI']
big=pd.concat([pd.read_csv(D/f"{w}_clean.csv") for w in WELLS], ignore_index=True)
def r2(y,p): tt=((y-y.mean())**2).sum(); return 1-((y-p)**2).sum()/tt

T='SHMAX'; K=5
sub=big.dropna(subset=FEATS+[T]); X=sub[FEATS].values; y=sub[T].values.astype(float)
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=0.30,random_state=42)
_,Xte,_,yte=train_test_split(Xtmp,ytmp,test_size=0.50,random_state=42)

# 特征重要性 -> top-K 主控变量
rf=RandomForestRegressor(n_estimators=300,n_jobs=-1,random_state=0).fit(Xtr,ytr)
imp=sorted(zip(FEATS,rf.feature_importances_),key=lambda z:-z[1])
print("特征重要性 top:", [(f,round(v,3)) for f,v in imp[:8]])
top=[f for f,_ in imp[:K]]; idx=[FEATS.index(f) for f in top]
print(f"选用主控变量 x1..x{K} =", top)

Xtr2,Xte2=Xtr[:,idx],Xte[:,idx]
xs=StandardScaler().fit(Xtr2); Xtr_s,Xte_s=xs.transform(Xtr2),xs.transform(Xte2)
ys=StandardScaler().fit(ytr.reshape(-1,1))
ytr_z=ys.transform(ytr.reshape(-1,1)).ravel(); yte_z=ys.transform(yte.reshape(-1,1)).ravel()
ds={'train_input':torch.tensor(Xtr_s,dtype=torch.float32),'train_label':torch.tensor(ytr_z.reshape(-1,1),dtype=torch.float32),
    'test_input':torch.tensor(Xte_s,dtype=torch.float32),'test_label':torch.tensor(yte_z.reshape(-1,1),dtype=torch.float32)}

m=KAN(width=[K,1],grid=6,k=3,seed=0,device='cpu')   # 纯加性: y=Σφ_i(x_i), 最可解释
m.fit(ds,opt="LBFGS",steps=60,lamb=1e-4)
pz=m(ds['test_input']).detach().numpy().ravel()
print(f"\nKAN[K,1]纯加性(符号化前) test R² = {r2(yte, ys.inverse_transform(pz.reshape(-1,1)).ravel()):.4f}")

m.auto_symbolic(lib=['x','x^2','x^3','1/x','sqrt','exp','log','tanh','sin'])
m.fit(ds,opt="LBFGS",steps=80,lamb=1e-4)   # 符号化后充分微调 affine 参数
pz2=m(ds['test_input']).detach().numpy().ravel()
print(f"KAN(符号化+微调后) test R² = {r2(yte, ys.inverse_transform(pz2.reshape(-1,1)).ravel()):.4f}")

try:
    formula=m.symbolic_formula()[0][0]
    print(f"\n提取公式 (标准化空间; x1..x{K} = {top}):")
    print(" ", ex_round(formula,3))
except Exception as e:
    print("公式提取:", type(e).__name__, e)
print(f"\n特征标准化参数 (还原用): mean={np.round(xs.mean_,3).tolist()}  std={np.round(xs.scale_,3).tolist()}")
print(f"目标标准化: mean={ys.mean_[0]:.3f} std={ys.scale_[0]:.3f}")
