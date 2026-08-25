"""
复现性桥接脚本 (reproducibility bridge)
=======================================
从公开脱敏数据 data/Well{A-D}.csv 还原出各训练脚本所需的 clean/{内部井名}_clean.csv,
使公开副本无需原始 LAS 即可直接运行 train_pooled.py / finalize_pipeline.py / fms_kan.py / extract3.py。

数据说明(甲方红线):
- data/Well{A-D}.csv 为甲方授权、逐样本随机 50% 抽取并井名脱敏的子集(仅 12 原始测井特征 + 3 目标 + DEPTH)。
- 原始 LAS 与全量数据不公开; build_dataset.py 仅供持有受控原始数据者复现完整清洗流程使用。
- 因此公开副本复现的是"从已发布脱敏数据出发"的可运行路径, 数值会因 50% 抽样而与论文全量结果存在差异,
  但方法、管线与相对结论可完整复现。

用法:
    python code/prepare_clean.py      # 生成 code/clean/*.csv
    python code/train_pooled.py       # 之后即可正常运行其余脚本
"""
import pandas as pd
from pathlib import Path

SB = Path(__file__).resolve().parent
DATA = SB.parent / "data"
OUT = SB / "clean"
OUT.mkdir(exist_ok=True)

# 公开脱敏井名 <-> 训练脚本内部井名
MAP = {'WellA': 'ZXX2', 'WellB': 'ZXX3', 'WellC': 'ZXX7', 'WellD': 'ZXX6'}

total = 0
for well, zw in MAP.items():
    src = DATA / f"{well}.csv"
    if not src.exists():
        raise FileNotFoundError(f"缺少公开数据 {src}; 请确认 data/Well{{A-D}}.csv 存在")
    df = pd.read_csv(src)
    df.to_csv(OUT / f"{zw}_clean.csv", index=False)
    total += len(df)
    print(f"{well}.csv -> clean/{zw}_clean.csv  ({len(df)} rows)")

print(f"\n完成: 共 {total} 行写入 code/clean/; 公开副本现可直接运行训练与提取脚本。")
print("列: " + ", ".join(pd.read_csv(DATA / 'WellA.csv').columns))
