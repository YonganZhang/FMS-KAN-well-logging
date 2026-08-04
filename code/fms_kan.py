"""
FMS-KAN 多尺度 (grid 5->10->20 多分辨率细化) vs 标准KAN — 同一 pooled split(可比)
=================================================================================
用 pykan 官方 grid refinement 实现多分辨率(粗->中->细),对应论文多尺度B-spline思想。
诚实标注: 用 grid extension 实现多尺度,非并行softmax融合(可后续升级)。
"""
import os, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from kan import KAN

SB=Path(__file__).resolve().parent
(SB/"_run").mkdir(exist_ok=True); os.chdir(SB/"_run")
D=SB/"clean"; WELLS=['ZXX2','ZXX3','ZXX7','ZXX6']
FEATS=['AC','DTC','DTS','DEN','GR','CNL','LLD','LLS','K','TH','U','DEVI']
big=pd.concat([pd.read_csv(D/f"{w}_clean.csv") for w in WELLS], ignore_index=True)
def r2(y,p): tt=((y-y.mean())**2).sum(); return 1-((y-p)**2).sum()/tt

def std_kan(Xtr,ytr,Xte,grid=10,steps=200,width=[12,10,6,1]):
    ds={'train_input':torch.tensor(Xtr,dtype=torch.float32),'train_label':torch.tensor(ytr.reshape(-1,1),dtype=torch.float32),
        'test_input':torch.tensor(Xte,dtype=torch.float32),'test_label':torch.zeros(len(Xte),1)}
    m=KAN(width=width,grid=grid,k=3,seed=0,device='cpu'); m.fit(ds,opt="Adam",steps=steps,lr=0.01,lamb=1e-3)
    with torch.no_grad(): return m(ds['test_input']).numpy().ravel()

def fms_kan(Xtr,ytr,Xte,grids=[5,10,20],steps=120,width=[12,10,6,1]):
    ds={'train_input':torch.tensor(Xtr,dtype=torch.float32),'train_label':torch.tensor(ytr.reshape(-1,1),dtype=torch.float32),
        'test_input':torch.tensor(Xte,dtype=torch.float32),'test_label':torch.zeros(len(Xte),1)}
    m=KAN(width=width,grid=grids[0],k=3,seed=0,device='cpu'); m.fit(ds,opt="Adam",steps=steps,lr=0.01,lamb=1e-3)
    for g in grids[1:]:
        m=m.refine(g); m.fit(ds,opt="Adam",steps=steps,lr=0.01,lamb=1e-3)
    with torch.no_grad(): return m(ds['test_input']).numpy().ravel()

print(f"{'目标':<7}{'标准KAN':>10}{'FMS-KAN':>10}   (同 split, test R²)")
for t in ['SHMAX','TOC','PERM']:
    sub=big.dropna(subset=FEATS+[t]); X=sub[FEATS].values; y=sub[t].values.astype(float)
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=0.30,random_state=42)
    Xval,Xte,yval,yte=train_test_split(Xtmp,ytmp,test_size=0.50,random_state=42)
    xs=StandardScaler().fit(Xtr); Xtr_s,Xte_s=xs.transform(Xtr),xs.transform(Xte)
    ys=StandardScaler().fit(ytr.reshape(-1,1)); ytr_z=ys.transform(ytr.reshape(-1,1)).ravel()
    ps=std_kan(Xtr_s,ytr_z,Xte_s); r_std=r2(yte,ys.inverse_transform(ps.reshape(-1,1)).ravel())
    pf=fms_kan(Xtr_s,ytr_z,Xte_s); r_fms=r2(yte,ys.inverse_transform(pf.reshape(-1,1)).ravel())
    print(f"{t:<7}{r_std:>10.4f}{r_fms:>10.4f}")
