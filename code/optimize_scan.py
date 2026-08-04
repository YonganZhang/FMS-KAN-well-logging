"""
优化思路扫描: 在 SHMAX 上找能把 KAN 从"标准水平"真正抬高的配方
==================================================================
试: Adam vs LBFGS / 正则 lamb / 网格 grid / 网络宽度 / grid细化 多尺度
基线参照: RF SHMAX=0.933, 标准KAN(Adam)=0.930
"""
import os, warnings, time, numpy as np, pandas as pd
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

def make_ds(Xtr,ytr,Xte):
    return {'train_input':torch.tensor(Xtr,dtype=torch.float32),'train_label':torch.tensor(ytr.reshape(-1,1),dtype=torch.float32),
            'test_input':torch.tensor(Xte,dtype=torch.float32),'test_label':torch.zeros(len(Xte),1)}

def train_cfg(ds, cfg):
    torch.manual_seed(0)
    m=KAN(width=cfg['width'],grid=cfg['grid'],k=cfg.get('k',3),seed=0,device='cpu')
    lr=cfg.get('lr', 0.01 if cfg['opt']=='Adam' else 1.0)
    m.fit(ds,opt=cfg['opt'],steps=cfg['steps'],lr=lr,lamb=cfg['lamb'])
    for g in cfg.get('refine',[]):          # 多尺度细化
        m=m.refine(g); m.fit(ds,opt=cfg['opt'],steps=cfg.get('refine_steps',cfg['steps']),lr=lr,lamb=cfg['lamb'])
    with torch.no_grad(): return m(ds['test_input']).numpy().ravel()

CONFIGS=[
 ('Adam-base',      dict(opt='Adam', grid=10, lamb=1e-3, steps=200, width=[12,10,6,1])),
 ('LBFGS',          dict(opt='LBFGS',grid=10, lamb=1e-3, steps=40,  width=[12,10,6,1])),
 ('LBFGS-lowreg',   dict(opt='LBFGS',grid=10, lamb=1e-4, steps=40,  width=[12,10,6,1])),
 ('LBFGS-grid20',   dict(opt='LBFGS',grid=20, lamb=1e-4, steps=40,  width=[12,10,6,1])),
 ('LBFGS-wide',     dict(opt='LBFGS',grid=15, lamb=1e-4, steps=40,  width=[12,16,8,1])),
 ('LBFGS-multiscale',dict(opt='LBFGS',grid=5, lamb=1e-4, steps=30, width=[12,12,6,1], refine=[10,20], refine_steps=25)),
]

for T in ['SHMAX','TOC']:
    sub=big.dropna(subset=FEATS+[T]); X=sub[FEATS].values; y=sub[T].values.astype(float)
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=0.30,random_state=42)
    _,Xte,_,yte=train_test_split(Xtmp,ytmp,test_size=0.50,random_state=42)
    xs=StandardScaler().fit(Xtr); Xtr_s,Xte_s=xs.transform(Xtr),xs.transform(Xte)
    ys=StandardScaler().fit(ytr.reshape(-1,1)); ytr_z=ys.transform(ytr.reshape(-1,1)).ravel()
    ds=make_ds(Xtr_s,ytr_z,Xte_s)
    print(f"\n===== {T} (RF基线: SHMAX0.933/TOC0.967, MLP TOC0.973) =====")
    for name,cfg in CONFIGS:
        try:
            t0=time.time(); pz=train_cfg(ds,cfg)
            p=ys.inverse_transform(pz.reshape(-1,1)).ravel()
            print(f"  {name:18} R2={r2(yte,p):.4f}   ({time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"  {name:18} ERR {type(e).__name__}: {e}")
