"""
Phase 2: LOWO 留一井 泛化对比 — 七方公平对比
============================================================
所有模型: 同一份清洗数据 / 同一 LOWO 划分 / 同一特征标准化 —— 公平。
KAN(pykan) vs 线性/多项式/RF/GBDT/MLP。FMS-KAN(多尺度) 后续接入。
输出: LOWO 平均 R² 表 + 明细 csv。结果如实记录,不为"赢"削弱任何 baseline。
"""
import os, argparse, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import torch
from kan import KAN

SB = Path(__file__).resolve().parent
D = SB / "clean"
RUNDIR = SB / "_run"; RUNDIR.mkdir(exist_ok=True)
os.chdir(RUNDIR)  # pykan 的 ./model checkpoint 落此,不污染项目
WELLS = ['ZXX2','ZXX3','ZXX7','ZXX6']
FEATS = ['AC','DTC','DTS','DEN','GR','CNL','LLD','LLS','K','TH','U','DEVI']

def r2(y,p):
    tt=((y-y.mean())**2).sum(); return 1-((y-p)**2).sum()/tt if tt>0 else float('nan')
def mae(y,p): return float(np.abs(y-p).mean())

def load(): return {w: pd.read_csv(D/f"{w}_clean.csv") for w in WELLS}

def fit_sklearn(name,Xtr,ytr,Xte):
    m = {'Linear':LinearRegression(),
         'Poly2':make_pipeline(PolynomialFeatures(2),Ridge(1.0)),
         'Poly3':make_pipeline(PolynomialFeatures(3),Ridge(1.0)),
         'RF':RandomForestRegressor(n_estimators=300,max_depth=14,n_jobs=-1,random_state=0),
         'GBDT':GradientBoostingRegressor(random_state=0),
         'MLP':MLPRegressor(hidden_layer_sizes=(64,32),max_iter=800,random_state=0)}[name]
    m.fit(Xtr,ytr); return m.predict(Xte)

def fit_kan(Xtr,ytr,Xte,steps,grid,width):
    ds={'train_input':torch.tensor(Xtr,dtype=torch.float32),
        'train_label':torch.tensor(ytr.reshape(-1,1),dtype=torch.float32),
        'test_input':torch.tensor(Xte,dtype=torch.float32),
        'test_label':torch.zeros(len(Xte),1)}
    m=KAN(width=width,grid=grid,k=3,seed=0,device='cpu')
    m.fit(ds,opt="Adam",steps=steps,lr=0.01,lamb=1e-3)
    with torch.no_grad(): return m(ds['test_input']).numpy().ravel()

def run(targets,fold_wells,kan_steps,models,grid,width):
    data=load(); rows=[]
    for t in targets:
        for held in fold_wells:
            tr=pd.concat([data[w] for w in WELLS if w!=held],ignore_index=True).dropna(subset=FEATS+[t])
            te=data[held].dropna(subset=FEATS+[t])
            if len(te)<30 or len(tr)<100: continue
            xs=StandardScaler().fit(tr[FEATS].values)
            Xtr,Xte=xs.transform(tr[FEATS].values),xs.transform(te[FEATS].values)
            ytr,yte=tr[t].values.astype(float),te[t].values.astype(float)
            ys=StandardScaler().fit(ytr.reshape(-1,1)); ytr_s=ys.transform(ytr.reshape(-1,1)).ravel()
            for mdl in models:
                try:
                    if mdl in ('KAN','FMS-KAN'):
                        ps=fit_kan(Xtr,ytr_s,Xte,kan_steps,grid,width)
                        p=ys.inverse_transform(ps.reshape(-1,1)).ravel()
                    else:
                        p=fit_sklearn(mdl,Xtr,ytr,Xte)
                    rows.append((t,mdl,held,r2(yte,p),mae(yte,p)))
                    print(f"  {t:5} {mdl:8} held={held}: R2={r2(yte,p):.3f}")
                except Exception as e:
                    rows.append((t,mdl,held,float('nan'),float('nan')))
                    print(f"  [warn] {t}/{mdl}/{held}: {type(e).__name__}: {e}")
    df=pd.DataFrame(rows,columns=['target','model','well','R2','MAE'])
    if len(df):
        piv=df.groupby(['target','model'])['R2'].mean().unstack('model')
        print("\n=== LOWO 平均 R² (行=目标, 列=模型) ===")
        print(piv.round(3).to_string())
        df.to_csv(SB/"lowo_results.csv",index=False)
        print(f"\n明细已存 {SB/'lowo_results.csv'}")
    return df

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--targets',default='SHMAX,TOC,PERM')
    ap.add_argument('--wells',default='ZXX2,ZXX3,ZXX7,ZXX6')
    ap.add_argument('--kan_steps',type=int,default=80)
    ap.add_argument('--models',default='Linear,Poly2,Poly3,RF,GBDT,MLP,KAN')
    ap.add_argument('--grid',type=int,default=5)
    ap.add_argument('--width',default='12,8,4,1')
    a=ap.parse_args()
    run(a.targets.split(','),a.wells.split(','),a.kan_steps,a.models.split(','),a.grid,[int(x) for x in a.width.split(',')])
