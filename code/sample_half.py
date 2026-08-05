"""
甲方数据红线: 逐样本随机抽取 50%, 脱敏井名(WellA-D), 仅保留特征+三目标, 供 GitHub 公开。
绝不上传全量真实数据。seed 固定可复现。
"""
import numpy as np, pandas as pd
from pathlib import Path
SB=Path(__file__).resolve().parent
CLEAN=SB/"clean"; OUT=SB/"github_repo"/"data"; OUT.mkdir(parents=True,exist_ok=True)
MAP={'ZXX2':'WellA','ZXX3':'WellB','ZXX7':'WellC','ZXX6':'WellD'}
FEATS=['DEPTH','AC','DTC','DTS','DEN','GR','CNL','LLD','LLS','K','TH','U','DEVI']
TARGETS=['SHMAX','TOC','PERM']
tot_full=tot_half=0
for zw,well in MAP.items():
    df=pd.read_csv(CLEAN/f"{zw}_clean.csv")
    cols=[c for c in FEATS+TARGETS if c in df.columns]
    df=df[cols]
    half=df.sample(frac=0.5,random_state=42).sort_values('DEPTH').reset_index(drop=True)
    half.to_csv(OUT/f"{well}.csv",index=False)
    tot_full+=len(df); tot_half+=len(half)
    print(f"{well}: 全量 {len(df)} -> 上传 {len(half)} (随机50%)")
print(f"总计: 全量 {tot_full} -> 公开 {tot_half} 行 ({100*tot_half/tot_full:.0f}%); 井名已脱敏为 WellA-D")
