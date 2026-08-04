"""
出图: 复用原 regenerate_desensitized.py 的绘图函数, 仅替换新数据/新R²/新目标(SHMAX/TOC/PERM)。
数据源: _rebuild_sandbox/data_desensitized_v2/ (FMS-KAN逐井预测) + perwell_all.csv + final_error_table.csv
输出:   paper/latex/figures/ (供论文) + paper/figures_v2/ (预览)
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

SB=Path(__file__).resolve().parent
PAPER=SB.parent/"paper"
DATA_DIR=SB/"data_desensitized_v2"
FIG_DIRS=[PAPER/"latex"/"figures", PAPER/"figures_v2"]
for d in FIG_DIRS: d.mkdir(parents=True, exist_ok=True)

WELLS=["WellA","WellB","WellC","WellD"]
TARGETS=["SHMAX","TOC","PERM"]
TARGET_LABELS={"SHMAX":"$\\sigma_{H,max}$ (MPa)","TOC":"TOC (wt%)","PERM":"Permeability (mD)"}
TARGET_SHORT={"SHMAX":"$\\sigma_H$","TOC":"TOC","PERM":"K"}
C_KAN="#7C3AED"; C_STD="#F59E0B"; C_POLY="#6B7280"; C_TRUE="#059669"; C_PRED="#DC2626"
CMAP_HEAT="PuBu"; CMAP_IMP="YlOrRd"
rcParams.update({'font.family':['sans-serif'],'font.sans-serif':['DejaVu Sans'],
 'mathtext.fontset':'dejavusans','axes.unicode_minus':False,'font.size':10,'axes.labelsize':11,
 'axes.titlesize':12,'xtick.labelsize':9,'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False})

# ---- R² 来源: perwell_all.csv(逐井三方) ----
pw=pd.read_csv(SB/"perwell_all.csv")
def r2d(model): return {(r.well,r.target):r.R2 for r in pw[pw.model==model].itertuples()}
R2_KAN=r2d('FMS-KAN'); R2_STD=r2d('Std.KAN'); R2_POLY=r2d('Poly.Reg')

def load_data(well,target):
    f=DATA_DIR/well/target/"data.csv"
    return pd.read_csv(f) if f.exists() else None
def save_fig(fig,name):
    for d in FIG_DIRS: fig.savefig(str(d/name),dpi=300,bbox_inches='tight',facecolor='white')
    print(f"  [OK] {name}")

def gen_r2_comparison():
    print("[1/5] R² 条形图...")
    fig,axes=plt.subplots(len(TARGETS),1,figsize=(9,9),sharex=True)
    y=np.arange(len(WELLS)); h=0.24
    for idx,t in enumerate(TARGETS):
        ax=axes[idx]
        vk=[R2_KAN.get((w,t),np.nan) for w in WELLS]
        vs=[R2_STD.get((w,t),np.nan) for w in WELLS]
        vp=[R2_POLY.get((w,t),np.nan) for w in WELLS]
        cl=lambda v:(float(np.clip(v,0.5,1.05)) if np.isfinite(v) else 0.5)
        ax.barh(y+h,[cl(x) for x in vk],h,label='FMS-KAN',color=C_KAN,edgecolor='white',linewidth=0.8)
        ax.barh(y,  [cl(x) for x in vs],h,label='Std. KAN',color=C_STD,edgecolor='white',linewidth=0.8)
        ax.barh(y-h,[cl(x) for x in vp],h,label='Poly. Reg.',color=C_POLY,edgecolor='white',linewidth=0.8)
        for i,(a,b,c) in enumerate(zip(vk,vs,vp)):
            for v,yy,col,fw in [(a,i+h,C_KAN,'bold'),(b,i,C_STD,'normal'),(c,i-h,C_POLY,'normal')]:
                if np.isfinite(v):
                    lab=f'{v:.3f}' if v>=0.5 else '<0.5'
                    ax.text(cl(v)+0.006,yy,lab,va='center',fontsize=7.5,color=col,fontweight=fw)
        ax.set_yticks(y); ax.set_yticklabels(WELLS)
        ax.set_xlim(0.5,1.08)
        ax.set_title(f'{TARGET_SHORT[t]}  ({TARGET_LABELS[t]})',loc='left',fontweight='bold')
        ax.axvline(0.95,color='#D1D5DB',ls=':',lw=0.8,alpha=0.7)
        if idx==0: ax.legend(fontsize=9,loc='lower right',framealpha=0.9)
    axes[-1].set_xlabel('$R^2$'); fig.tight_layout(h_pad=1.3)
    save_fig(fig,"fig_r2_comparison.png"); plt.close(fig)

def gen_scatter():
    print("[2/5] 密度散点...")
    combos=[("WellA","SHMAX"),("WellB","TOC"),("WellC","PERM"),("WellD","SHMAX")]
    fig,axes=plt.subplots(1,4,figsize=(18,4.5))
    from scipy.stats import gaussian_kde
    for idx,(w,t) in enumerate(combos):
        df=load_data(w,t); ax=axes[idx]
        if df is None: continue
        xy=np.vstack([df["measured"],df["predicted"]])
        try:
            dens=gaussian_kde(xy)(xy); o=dens.argsort()
            ax.scatter(df["measured"].values[o],df["predicted"].values[o],c=dens[o],s=6,cmap='plasma',alpha=0.7,edgecolors='none')
        except Exception:
            ax.scatter(df["measured"],df["predicted"],s=6,alpha=0.5,c=C_KAN,edgecolors='none')
        vmin=min(df["measured"].min(),df["predicted"].min()); vmax=max(df["measured"].max(),df["predicted"].max())
        m=(vmax-vmin)*0.05; ax.plot([vmin-m,vmax+m],[vmin-m,vmax+m],color='#374151',lw=1.5,alpha=0.6)
        ax.set_xlim(vmin-m,vmax+m); ax.set_ylim(vmin-m,vmax+m)
        rr=R2_KAN.get((w,t),np.nan)
        ax.text(0.95,0.05,f'$R^2$={rr:.3f}',transform=ax.transAxes,fontsize=11,fontweight='bold',ha='right',va='bottom',
                bbox=dict(boxstyle='square,pad=0.3',facecolor='white',edgecolor=C_KAN,alpha=0.9))
        ax.set_xlabel('Measured'); ax.set_ylabel('Predicted' if idx==0 else '')
        ax.set_title(f'{w}  {TARGET_SHORT[t]}',fontweight='bold'); ax.set_aspect('equal')
    fig.tight_layout(w_pad=2); save_fig(fig,"fig_scatter_pred_vs_true.png"); plt.close(fig)

def gen_depth():
    print("[3/5] 深度剖面...")
    for i,t in enumerate(TARGETS):
        suf={0:"a",1:"b",2:"c"}[i]
        fig,axes=plt.subplots(2,2,figsize=(14,8)); axf=axes.flatten()
        for wi,w in enumerate(WELLS):
            df=load_data(w,t); ax=axf[wi]
            if df is None: continue
            df=df.sort_values("depth")
            ax.plot(df["depth"],df["measured"],color=C_TRUE,lw=1.0,alpha=0.8,label="Measured")
            ax.plot(df["depth"],df["predicted"],color=C_PRED,lw=0.7,ls='-.',alpha=0.85,label="FMS-KAN")
            ax.fill_between(df["depth"],df["measured"],df["predicted"],alpha=0.08,color=C_KAN)
            ax.set_xlabel("Depth (m)"); ax.set_ylabel(TARGET_LABELS[t]); ax.set_title(w,fontweight='bold')
            ax.legend(fontsize=8,loc='upper right',framealpha=0.85); ax.grid(alpha=0.15)
        fig.suptitle(f'{TARGET_SHORT[t]}  Depth Profile Comparison',fontsize=13,fontweight='bold')
        fig.tight_layout(rect=[0,0,1,0.95]); save_fig(fig,f"fig_depth_{suf}_{t}.png"); plt.close(fig)

def gen_heatmap():
    print("[4/5] 提升热力图...")
    data=np.array([[R2_KAN.get((w,t),np.nan)-R2_STD.get((w,t),np.nan) for w in WELLS] for t in TARGETS])
    fig,ax=plt.subplots(figsize=(7,4.5))
    im=ax.imshow(data,cmap=CMAP_HEAT,aspect='auto',vmin=0,vmax=max(0.05,np.nanmax(data)))
    ax.set_xticks(range(len(WELLS))); ax.set_xticklabels(WELLS)
    ax.set_yticks(range(len(TARGETS))); ax.set_yticklabels([TARGET_SHORT[t] for t in TARGETS])
    for i in range(len(TARGETS)):
        for j in range(len(WELLS)):
            v=data[i,j]
            if np.isfinite(v): ax.text(j,i,f'{v:+.3f}',ha='center',va='center',fontsize=11,fontweight='bold',
                                       color="white" if v>np.nanmax(data)*0.6 else "#1F2937")
    fig.colorbar(im,ax=ax,shrink=0.85,label='$\\Delta R^2$ (FMS-KAN − Std. KAN)')
    ax.set_title('Improvement over Standard KAN',fontweight='bold'); fig.tight_layout()
    save_fig(fig,"fig_r2_improvement_heatmap.png"); plt.close(fig)

def gen_feature_heatmap():
    print("[5/5] 特征重要性热力图...")
    from sklearn.ensemble import RandomForestRegressor
    D=SB/"clean"; WZ=['ZXX2','ZXX3','ZXX7','ZXX6']
    BASE=['AC','DTC','DTS','DEN','GR','CNL','LLD','LLS','K','TH','U','DEVI']
    big=pd.concat([pd.read_csv(D/f"{w}_clean.csv") for w in WZ],ignore_index=True)
    imp={}
    for t in TARGETS:
        sub=big.dropna(subset=BASE+[t])
        rf=RandomForestRegressor(n_estimators=300,n_jobs=-1,random_state=0).fit(sub[BASE],sub[t])
        imp[t]=rf.feature_importances_
    M=np.array([imp[t] for t in TARGETS]).T
    fig,ax=plt.subplots(figsize=(6,7))
    im=ax.imshow(M,cmap=CMAP_IMP,aspect='auto')
    ax.set_xticks(range(len(TARGETS))); ax.set_xticklabels([TARGET_SHORT[t] for t in TARGETS])
    ax.set_yticks(range(len(BASE))); ax.set_yticklabels(BASE)
    for i in range(len(BASE)):
        for j in range(len(TARGETS)):
            ax.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',fontsize=8,
                    color='white' if M[i,j]>M.max()*0.55 else '#1F2937')
    fig.colorbar(im,ax=ax,shrink=0.85,label='RF Feature Importance')
    ax.set_title('Feature Importance',fontweight='bold'); fig.tight_layout()
    save_fig(fig,"fig_feature_heatmap.png"); plt.close(fig)

if __name__=="__main__":
    gen_r2_comparison(); gen_scatter(); gen_depth(); gen_heatmap(); gen_feature_heatmap()
    print("All figures done.")
