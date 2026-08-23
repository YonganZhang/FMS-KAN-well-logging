"""④ 认真跑 3 目标 KAN 公式提取: top4 特征 -> [4,1] 纯加性 KAN -> 剪枝符号化 -> 公式+符号化前后R²。
输出如实; 若符号化后 R² 崩则论文降级、不 AI 编。"""
import os, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import torch
from kan import KAN
from kan.utils import ex_round
SB=Path(__file__).resolve().parent; (SB/"_run").mkdir(exist_ok=True); os.chdir(SB/"_run")
D=SB/"clean"; WELLS=['ZXX2','ZXX3','ZXX7','ZXX6']
BASE=['AC','DTC','DTS','DEN','GR','CNL','LLD','LLS','K','TH','U','DEVI']
big=pd.concat([pd.read_csv(D/f"{w}_clean.csv") for w in WELLS], ignore_index=True)
def r2(y,p): tt=((y-y.mean())**2).sum(); return 1-((y-p)**2).sum()/tt
for T,LOG in [('SHMAX',False),('TOC',False),('PERM',True)]:
    sub=big.dropna(subset=BASE+[T]); X=sub[BASE].values; y0=sub[T].values.astype(float)
    y=np.log1p(y0) if LOG else y0
    Xtr,Xt,ytr,yt=train_test_split(X,y,test_size=0.3,random_state=42); _,Xte,_,yte=train_test_split(Xt,yt,test_size=0.5,random_state=42)
    rf=RandomForestRegressor(200,n_jobs=-1,random_state=0).fit(Xtr,ytr)
    top=[f for f,_ in sorted(zip(BASE,rf.feature_importances_),key=lambda z:-z[1])[:4]]; idx=[BASE.index(f) for f in top]
    xs=StandardScaler().fit(Xtr[:,idx]); ys=StandardScaler().fit(ytr.reshape(-1,1))
    Xtr_s,Xte_s=xs.transform(Xtr[:,idx]),xs.transform(Xte[:,idx]); ytr_z=ys.transform(ytr.reshape(-1,1)).ravel()
    ds={'train_input':torch.tensor(Xtr_s,dtype=torch.float32),'train_label':torch.tensor(ytr_z.reshape(-1,1),dtype=torch.float32),
        'test_input':torch.tensor(Xte_s,dtype=torch.float32),'test_label':torch.zeros(len(Xte_s),1)}
    back=lambda a:(np.expm1(a) if LOG else a)
    m=KAN(width=[4,1],grid=5,k=3,seed=0,device='cpu'); m.fit(ds,opt="LBFGS",steps=60,lamb=1e-4)
    pz=ys.inverse_transform(m(ds['test_input']).detach().numpy().reshape(-1,1)).ravel(); r_pre=r2(back(yte),back(pz))
    m.auto_symbolic(lib=['x','x^2','x^3','1/x','sqrt','exp','log','tanh']); m.fit(ds,opt="LBFGS",steps=40,lamb=1e-4)
    pz2=ys.inverse_transform(m(ds['test_input']).detach().numpy().reshape(-1,1)).ravel(); r_sym=r2(back(yte),back(pz2))
    print(f"\n=== {T}  top4={top}  符号化前R²={r_pre:.3f}  符号化后R²={r_sym:.3f} ===",flush=True)
    try: print("  公式:", ex_round(m.symbolic_formula()[0][0],3))
    except Exception as e: print("  公式err:",type(e).__name__,e)
