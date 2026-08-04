"""
Phase 2 (正式口径): pooled 全数据 -> train/val/test 标准划分, 七方对比
============================================================================
按用户口径: 四井合并, 随机 70/15/15 划分 (val 供 KAN 监控)。
所有模型同一 train 训练、同一 test 评估、同一特征标准化 —— 公平。
如实报告; 评估协议(pooled 随机划分,含相邻点相关)将写入方法与独立报告。
"""
import os, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import torch
from kan import KAN

SB=Path(__file__).resolve().parent
(SB/"_run").mkdir(exist_ok=True); os.chdir(SB/"_run")
D=SB/"clean"; WELLS=['ZXX2','ZXX3','ZXX7','ZXX6']
FEATS=['AC','DTC','DTS','DEN','GR','CNL','LLD','LLS','K','TH','U','DEVI']
big=pd.concat([pd.read_csv(D/f"{w}_clean.csv") for w in WELLS], ignore_index=True)
TARGETS=['SHMAX','TOC','PERM']

def r2(y,p):
    tt=((y-y.mean())**2).sum(); return 1-((y-p)**2).sum()/tt if tt>0 else float('nan')
def mae(y,p): return float(np.abs(y-p).mean())

def sk(name):
    return {'Linear':LinearRegression(),
            'Poly2':make_pipeline(PolynomialFeatures(2),Ridge(1.0)),
            'Poly3':make_pipeline(PolynomialFeatures(3),Ridge(1.0)),
            'RF':RandomForestRegressor(n_estimators=400,max_depth=16,n_jobs=-1,random_state=0),
            'GBDT':GradientBoostingRegressor(random_state=0),
            'MLP':MLPRegressor(hidden_layer_sizes=(128,64),max_iter=1000,random_state=0)}[name]

def kan_pred(Xtr,ytr,Xte,steps=200,grid=10,width=[12,10,6,1]):
    ds={'train_input':torch.tensor(Xtr,dtype=torch.float32),
        'train_label':torch.tensor(ytr.reshape(-1,1),dtype=torch.float32),
        'test_input':torch.tensor(Xte,dtype=torch.float32),
        'test_label':torch.zeros(len(Xte),1)}
    m=KAN(width=width,grid=grid,k=3,seed=0,device='cpu')
    m.fit(ds,opt="Adam",steps=steps,lr=0.01,lamb=1e-3)
    with torch.no_grad(): return m(ds['test_input']).numpy().ravel()

MODELS=['Linear','Poly2','Poly3','RF','GBDT','MLP','KAN']
rows=[]
for t in TARGETS:
    sub=big.dropna(subset=FEATS+[t])
    X=sub[FEATS].values; y=sub[t].values.astype(float)
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=0.30,random_state=42)
    Xval,Xte,yval,yte=train_test_split(Xtmp,ytmp,test_size=0.50,random_state=42)
    xs=StandardScaler().fit(Xtr); Xtr_s,Xte_s=xs.transform(Xtr),xs.transform(Xte)
    ys=StandardScaler().fit(ytr.reshape(-1,1)); ytr_z=ys.transform(ytr.reshape(-1,1)).ravel()
    print(f"\n== {t}: n={len(sub)} (train {len(Xtr)}/val {len(Xval)}/test {len(Xte)}), 值域[{y.min():.2f},{y.max():.2f}] ==")
    for mdl in MODELS:
        try:
            if mdl=='KAN':
                pz=kan_pred(Xtr_s,ytr_z,Xte_s); p=ys.inverse_transform(pz.reshape(-1,1)).ravel()
            else:
                m=sk(mdl); m.fit(Xtr_s,ytr); p=m.predict(Xte_s)
            rows.append((t,mdl,r2(yte,p),mae(yte,p)))
            print(f"  {mdl:8}: test R2={r2(yte,p):.4f}  MAE={mae(yte,p):.3f}")
        except Exception as e:
            rows.append((t,mdl,float('nan'),float('nan'))); print(f"  {mdl:8}: ERR {e}")

df=pd.DataFrame(rows,columns=['target','model','R2','MAE'])
piv=df.pivot(index='target',columns='model',values='R2')[MODELS]
print("\n================ 正式 test R² 误差表 (行=目标, 列=模型) ================")
print(piv.round(4).to_string())
df.to_csv(SB/"pooled_results.csv",index=False)
print(f"\n明细: {SB/'pooled_results.csv'}")
