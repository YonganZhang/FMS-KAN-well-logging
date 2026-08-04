"""
Phase A (可信重建): 固化 pipeline — 用已独立验证的逻辑(RF SHMAX≈0.933)重出
================================================================================
修正: (1) 用验证过的固定 concat 顺序+划分; (2) 内置 RF sanity 断言防虚高bug复发;
      (3) PERM 用 log1p 变换处理 ZXX2 量级问题, 预测后 expm1 还原。
配置: 特征=12原始+6物理派生; FMS-KAN=[18,10,4,1] 多尺度5->10->15 LBFGS; pooled 70/15/15(seed42)。
产物: data_desensitized_v2/<Well>/<T>/data.csv | models/<T>_fmskan.pt | final_error_table.csv | r2_per_well.json
"""
import os, json, warnings, numpy as np, pandas as pd
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
D=SB/"clean"
OUT_PRED=SB/"data_desensitized_v2"; OUT_MODEL=SB/"models"
OUT_PRED.mkdir(exist_ok=True); OUT_MODEL.mkdir(exist_ok=True)
WELLS=['ZXX2','ZXX3','ZXX7','ZXX6']                 # 验证过的顺序(RF SHMAX=0.9337)
WELLMAP={'ZXX2':'WellA','ZXX3':'WellB','ZXX7':'WellC','ZXX6':'WellD'}
BASE=['AC','DTC','DTS','DEN','GR','CNL','LLD','LLS','K','TH','U','DEVI']
PHYS=['Vp','Vs','VpVs','AI','SI','Gpx']; FEATS=BASE+PHYS
TARGETS=['SHMAX','TOC','PERM']; LOGT={'PERM'}       # 需 log 变换的目标
WIDTH=[len(FEATS),10,4,1]

def add_phys(d):
    d=d.copy(); e=1e-6
    d['Vp']=1/(d['DTC']+e); d['Vs']=1/(d['DTS']+e); d['VpVs']=d['DTS']/(d['DTC']+e)
    d['AI']=d['DEN']/(d['DTC']+e); d['SI']=d['DEN']/(d['DTS']+e); d['Gpx']=d['DEN']/(d['DTS']**2+e)
    return d
def r2(y,p): tt=((y-y.mean())**2).sum(); return float(1-((y-p)**2).sum()/tt) if tt>0 else float('nan')
def mae(y,p): return float(np.abs(y-p).mean())

dfs={w:add_phys(pd.read_csv(D/f"{w}_clean.csv")) for w in WELLS}
big=pd.concat([dfs[w] for w in WELLS], ignore_index=True)

def train_fmskan(ds):
    m=KAN(width=WIDTH,grid=5,k=3,seed=0,device='cpu'); m.fit(ds,opt="LBFGS",steps=40,lamb=1e-4)
    for g in [10,15]:
        m=m.refine(g); m.fit(ds,opt="LBFGS",steps=30,lamb=1e-4)
    return m
def sk(name):
    return {'Linear':LinearRegression(),'Poly2':make_pipeline(PolynomialFeatures(2),Ridge(1.0)),
            'Poly3':make_pipeline(PolynomialFeatures(3),Ridge(1.0)),
            'RF':RandomForestRegressor(n_estimators=400,max_depth=16,n_jobs=-1,random_state=0),
            'GBDT':GradientBoostingRegressor(random_state=0),
            'MLP':MLPRegressor(hidden_layer_sizes=(128,64),max_iter=1000,random_state=0)}[name]

err_rows=[]; r2_per_well={}
for T in TARGETS:
    sub=big.dropna(subset=FEATS+[T]); X=sub[FEATS].values; y0=sub[T].values.astype(float)
    y=np.log1p(y0) if T in LOGT else y0                # log 变换(仅PERM)
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=0.30,random_state=42)
    Xval,Xte,yval,yte=train_test_split(Xtmp,ytmp,test_size=0.50,random_state=42)
    xs=StandardScaler().fit(Xtr); ys=StandardScaler().fit(ytr.reshape(-1,1))
    Xtr_s,Xte_s=xs.transform(Xtr),xs.transform(Xte); ytr_z=ys.transform(ytr.reshape(-1,1)).ravel()
    inv=lambda z: ys.inverse_transform(z.reshape(-1,1)).ravel()
    back=lambda a: (np.expm1(a) if T in LOGT else a)   # 还原到原始量纲算R²
    # baselines (含 RF sanity)
    for name in ['Linear','Poly2','Poly3','RF','GBDT','MLP']:
        mm=sk(name); mm.fit(Xtr_s,ytr); p=mm.predict(Xte_s)
        rr=r2(back(yte),back(p)); err_rows.append((T,name,rr,mae(back(yte),back(p))))
        if T=='SHMAX' and name=='RF':
            assert 0.90<=rr<=0.96, f"RF SHMAX sanity 失败={rr:.4f} (应≈0.933,虚高bug复现!)"
    # FMS-KAN
    ds={'train_input':torch.tensor(Xtr_s,dtype=torch.float32),'train_label':torch.tensor(ytr_z.reshape(-1,1),dtype=torch.float32),
        'test_input':torch.tensor(Xte_s,dtype=torch.float32),'test_label':torch.zeros(len(Xte_s),1)}
    m=train_fmskan(ds)
    with torch.no_grad(): pk=inv(m(ds['test_input']).numpy().ravel())
    err_rows.append((T,'FMS-KAN',r2(back(yte),back(pk)),mae(back(yte),back(pk))))
    torch.save({'state_dict':m.state_dict(),'width':WIDTH,'grid':15,'k':3,'feats':FEATS,'log':T in LOGT,
                'x_mean':xs.mean_.tolist(),'x_std':xs.scale_.tolist(),'y_mean':float(ys.mean_[0]),'y_std':float(ys.scale_[0])},
               OUT_MODEL/f"{T}_fmskan.pt")
    # 逐井全点预测 -> 脱敏格式(供原绘图代码), measured/predicted 均为原始量纲
    for zw,well in WELLMAP.items():
        dfw=dfs[zw].dropna(subset=FEATS+[T,'DEPTH'])
        if len(dfw)<10: continue
        with torch.no_grad(): pw=back(inv(m(torch.tensor(xs.transform(dfw[FEATS].values),dtype=torch.float32)).numpy().ravel()))
        meas=dfw[T].values
        outdir=OUT_PRED/well/T; outdir.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'depth':dfw['DEPTH'].values,'measured':meas,'predicted':pw}).to_csv(outdir/"data.csv",index=False)
        r2_per_well[f"{well}|{T}"]=r2(meas,pw)
    print(f"[done] {T}", flush=True)

df=pd.DataFrame(err_rows,columns=['target','model','R2','MAE'])
df.to_csv(SB/"final_error_table.csv",index=False)
json.dump(r2_per_well, open(SB/"r2_per_well.json","w"), indent=2)
json.dump(WELLMAP, open(SB/"wellmap.json","w"), indent=2)
print("\n================ 最终 test R² 误差表(可信) ================")
print(df.pivot(index='target',columns='model',values='R2').round(4).to_string())
print("\nper井 R²:", json.dumps(r2_per_well,indent=0))
