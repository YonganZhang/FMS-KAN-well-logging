"""
补算对照模型逐井 R² (条形图/提升热力图要用): 标准KAN + Poly2, per井。
FMS-KAN 逐井直接从 data_desensitized_v2 的 measured/predicted 读取。
"""
import os, json, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import torch
from kan import KAN

SB=Path(__file__).resolve().parent
(SB/"_run").mkdir(exist_ok=True); os.chdir(SB/"_run")
D=SB/"clean"; PRED=SB/"data_desensitized_v2"
WELLS=['ZXX2','ZXX3','ZXX7','ZXX6']; WELLMAP={'ZXX2':'WellA','ZXX3':'WellB','ZXX7':'WellC','ZXX6':'WellD'}
BASE=['AC','DTC','DTS','DEN','GR','CNL','LLD','LLS','K','TH','U','DEVI']
TARGETS=['SHMAX','TOC','PERM']; LOGT={'PERM'}
def r2(y,p): tt=((y-y.mean())**2).sum(); return float(1-((y-p)**2).sum()/tt) if tt>0 else float('nan')
dfs={w:pd.read_csv(D/f"{w}_clean.csv") for w in WELLS}
big=pd.concat([dfs[w] for w in WELLS], ignore_index=True)
rows=[]
# FMS-KAN (from 已导出的预测)
for zw,well in WELLMAP.items():
    for T in TARGETS:
        f=PRED/well/T/"data.csv"
        if f.exists():
            d=pd.read_csv(f); rows.append(('FMS-KAN',well,T,r2(d['measured'].values,d['predicted'].values)))
# 标准KAN(单尺度G10 Adam, BASE12) + Poly2 per井
for T in TARGETS:
    sub=big.dropna(subset=BASE+[T]); X=sub[BASE].values; y0=sub[T].values.astype(float)
    y=np.log1p(y0) if T in LOGT else y0
    Xtr,Xt,ytr,yt=train_test_split(X,y,test_size=0.3,random_state=42)
    xs=StandardScaler().fit(Xtr); ys=StandardScaler().fit(ytr.reshape(-1,1))
    Xtr_s=xs.transform(Xtr); ytr_z=ys.transform(ytr.reshape(-1,1)).ravel()
    back=lambda a:(np.expm1(a) if T in LOGT else a)
    # 标准KAN
    ds={'train_input':torch.tensor(Xtr_s,dtype=torch.float32),'train_label':torch.tensor(ytr_z.reshape(-1,1),dtype=torch.float32),
        'test_input':torch.tensor(Xtr_s[:5],dtype=torch.float32),'test_label':torch.zeros(5,1)}
    mk=KAN(width=[12,10,6,1],grid=10,k=3,seed=0,device='cpu'); mk.fit(ds,opt="Adam",steps=150,lamb=1e-3)
    pf=make_pipeline(PolynomialFeatures(2),Ridge(1.0)).fit(Xtr_s,ytr)
    for zw,well in WELLMAP.items():
        dw=dfs[zw].dropna(subset=BASE+[T]); Xw=xs.transform(dw[BASE].values); yw=dw[T].values.astype(float)
        with torch.no_grad(): pk=back(ys.inverse_transform(mk(torch.tensor(Xw,dtype=torch.float32)).numpy().reshape(-1,1)).ravel())
        pp=back(pf.predict(Xw))
        rows.append(('Std.KAN',well,T,r2(yw,pk))); rows.append(('Poly.Reg',well,T,r2(yw,pp)))
    print(f"[done] {T}", flush=True)
pd.DataFrame(rows,columns=['model','well','target','R2']).to_csv(SB/"perwell_all.csv",index=False)
print("saved perwell_all.csv")
