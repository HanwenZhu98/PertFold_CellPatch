"""summarize_bench.py 的主表（不做 bootstrap，快）"""
import re, sys
import pandas as pd
import summarize_bench as B
from summarize_v10 import composite, mean_reps, REF
pd.set_option("display.width", 250)
for title, arms in B.TABLES.items():
    rows = []
    refs = {k: v for k, v in REF.items() if (k in ("identity", "训练平均") if title.startswith("全部") else True)}
    for k, v in refs.items():
        u = mean_reps([v.format(r=r) for r in range(5)]); rows.append({"配置": k, **{a: round(b, 3) for a, b in u.items()}, "综合分": round(composite(u)[0], 3)})
    for name, d, stem in arms:
        ps = B.fs_paths(d, stem)
        if not ps:
            rows.append({"配置": name + "（未完成）"}); continue
        u = mean_reps(ps); c, s = composite(u)
        rows.append({"配置": f"{name}（s={re.search(r'_fs(.*)_per_pert', ps[0]).group(1)}）", **{a: round(b, 3) for a, b in u.items()}, "综合分": round(c, 3)})
    print(f"\n===== {title}"); print(pd.DataFrame(rows).to_string(index=False)); sys.stdout.flush()
