"""
出图 v2: 复用原绘图函数, 第二轮修正 —
 - 图三散点: 1×4 -> 2×2
 - 深度剖面: 竖版测井曲线风格(depth 纵轴向下、值横轴)
 - 图四热力图: 对提升值 clip(标准KAN井A PERM崩溃致极端, 正文说明)
 - 图二R²对比: 不画(改为论文表格)
数据源: data_desensitized_v2/(FMS-KAN预测) + perwell_all.csv
输出: paper/latex/figures/ + paper/figures_v2/
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
rcParams.update({'font.family':['serif'],'font.serif':['Times New Roman','DejaVu Serif'],
 'mathtext.fontset':'stix','axes.unicode_minus':False,'font.size':10,'axes.labelsize':11,
 'axes.titlesize':12,'xtick.labelsize':9,'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False})

pw=pd.read_csv(SB/"perwell_all.csv")
def r2d(model): return {(r.well,r.target):r.R2 for r in pw[pw.model==model].itertuples()}
R2_KAN=r2d('FMS-KAN'); R2_STD=r2d('Std.KAN'); R2_POLY=r2d('Poly.Reg')

def load_data(well,target):
    f=DATA_DIR/well/target/"data.csv"
    return pd.read_csv(f) if f.exists() else None
def save_fig(fig,name):
    for d in FIG_DIRS: fig.savefig(str(d/name),dpi=300,bbox_inches='tight',facecolor='white')
    print(f"  [OK] {name}")

# ---- 图三: 密度散点 2×2 ----
def gen_scatter():
    print("[1/4] 密度散点 2x2...")
    from scipy.stats import gaussian_kde
    combos=[("WellA","SHMAX"),("WellB","TOC"),("WellC","PERM"),("WellD","SHMAX")]
    fig,axes=plt.subplots(2,2,figsize=(10.5,10)); axf=axes.flatten()
    for idx,(w,t) in enumerate(combos):
        df=load_data(w,t); ax=axf[idx]
        if df is None: continue
        xy=np.vstack([df["measured"],df["predicted"]])
        try:
            dens=gaussian_kde(xy)(xy); o=dens.argsort()
            ax.scatter(df["measured"].values[o],df["predicted"].values[o],c=dens[o],s=7,cmap='plasma',alpha=0.75,edgecolors='none')
        except Exception:
            ax.scatter(df["measured"],df["predicted"],s=7,alpha=0.5,c=C_KAN,edgecolors='none')
        vmin=min(df["measured"].min(),df["predicted"].min()); vmax=max(df["measured"].max(),df["predicted"].max())
        m=(vmax-vmin)*0.05; ax.plot([vmin-m,vmax+m],[vmin-m,vmax+m],color='#374151',lw=1.5,alpha=0.6)
        ax.set_xlim(vmin-m,vmax+m); ax.set_ylim(vmin-m,vmax+m)
        rr=R2_KAN.get((w,t),np.nan)
        ax.text(0.95,0.06,f'$R^2$={rr:.3f}',transform=ax.transAxes,fontsize=12,fontweight='bold',ha='right',va='bottom',
                bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor=C_KAN,alpha=0.9))
        ax.text(-0.12,1.04,chr(97+idx),transform=ax.transAxes,fontsize=15,fontweight='bold',va='bottom',ha='left')
        ax.set_xlabel(f'Measured {TARGET_SHORT[t]}'); ax.set_ylabel(f'Predicted {TARGET_SHORT[t]}')
        ax.set_aspect('equal')
    fig.tight_layout(w_pad=2.5,h_pad=2.5); save_fig(fig,"fig_scatter_pred_vs_true.png"); plt.close(fig)

# ---- 深度剖面: 竖版测井曲线风格(1×4井, x=值 y=depth向下) ----
def gen_depth():
    print("[2/4] 深度剖面(竖版测井风格)...")
    for i,t in enumerate(TARGETS):
        suf={0:"a",1:"b",2:"c"}[i]
        fig,axes=plt.subplots(1,4,figsize=(13,9))
        for wi,w in enumerate(WELLS):
            df=load_data(w,t); ax=axes[wi]
            if df is None: continue
            df=df.sort_values("depth")
            ax.plot(df["measured"],df["depth"],color=C_TRUE,lw=1.0,alpha=0.85,label="Measured")
            ax.plot(df["predicted"],df["depth"],color=C_PRED,lw=0.8,ls='-',alpha=0.8,label="FMS-KAN")
            ax.fill_betweenx(df["depth"],df["measured"],df["predicted"],alpha=0.10,color=C_KAN)
            ax.invert_yaxis()                      # 深度向下增加(测井惯例)
            ax.set_xlabel(TARGET_LABELS[t]); ax.set_ylabel("Depth (m)" if wi==0 else "")
            ax.text(0.5,1.02,w,transform=ax.transAxes,fontsize=11,fontweight='bold',ha='center')
            ax.grid(alpha=0.15)
            if wi==0: ax.legend(fontsize=8,loc='lower left',framealpha=0.85)
        fig.tight_layout(w_pad=1.5); save_fig(fig,f"fig_depth_{suf}_{t}.png"); plt.close(fig)

# ---- 图四: 提升热力图(clip 防标准KAN崩溃致极端) ----
def gen_heatmap():
    print("[3/4] 提升热力图(clip)...")
    raw=np.array([[R2_KAN.get((w,t),np.nan)-R2_STD.get((w,t),np.nan) for w in WELLS] for t in TARGETS])
    data=np.clip(raw,0,0.1)                        # clip: 只显示 0~0.1 合理提升区间
    fig,ax=plt.subplots(figsize=(7,4.5))
    im=ax.imshow(data,cmap=CMAP_HEAT,aspect='auto',vmin=0,vmax=0.1)
    ax.set_xticks(range(len(WELLS))); ax.set_xticklabels(WELLS)
    ax.set_yticks(range(len(TARGETS))); ax.set_yticklabels([TARGET_SHORT[t] for t in TARGETS])
    for i in range(len(TARGETS)):
        for j in range(len(WELLS)):
            v=raw[i,j]
            lab=f'{v:+.3f}' if abs(v)<1 else ('>>0.1' if v>0 else '<<0')
            ax.text(j,i,lab,ha='center',va='center',fontsize=10,fontweight='bold',
                    color="white" if data[i,j]>0.06 else "#1F2937")
    fig.colorbar(im,ax=ax,shrink=0.85,label='$\\Delta R^2$ (FMS-KAN − Std. KAN), clipped to [0,0.1]')
    fig.tight_layout(); save_fig(fig,"fig_r2_improvement_heatmap.png"); plt.close(fig)

# ---- 特征重要性热力图 ----
def gen_feature_heatmap():
    print("[4/4] 特征重要性热力图...")
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
    fig.tight_layout(); save_fig(fig,"fig_feature_heatmap.png"); plt.close(fig)

if __name__=="__main__":
    gen_scatter(); gen_depth(); gen_heatmap(); gen_feature_heatmap()
    print("Figures done (图二R²对比改为论文表格,不出图).")
