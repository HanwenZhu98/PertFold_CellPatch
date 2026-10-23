"""两组 5 折结果的逐扰动 bootstrap（综合分 / 六项之差）：python quick_boot.py <目录A> <stemA> <目录B> <stemB> [次数]"""
import sys
from pathlib import Path
import numpy as np
import summarize_bench as B
from summarize_v10 import SIX, composite, six, wide
da, sa, db, sb = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
pa, pb = B.fs_paths(da, sa), B.fs_paths(db, sb)
wa, wb = [wide(p) for p in pa], [wide(p) for p in pb]
perts = sorted(set.intersection(*[set(w.index) for w in wa + wb]))
def score(ws, idx):
    u = {k: np.mean([six(w.loc[idx])[k] for w in ws]) for k in SIX}
    return composite(u)[0], u
c0a, ua = score(wa, perts); c0b, ub = score(wb, perts)
print(f"{sa}: 综合分 {c0a:.3f}；{sb}: {c0b:.3f}；差 {c0a - c0b:+.3f}")
print("六项之差：", {k: round(ua[k] - ub[k], 3) for k in SIX})
rng = np.random.default_rng(0); d = []
for _ in range(n):
    idx = rng.choice(perts, len(perts))
    d.append(score(wa, idx)[0] - score(wb, idx)[0])
print(f"逐扰动 bootstrap {n} 次：差 {np.mean(d):+.3f}，95% 区间 [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]，P(差 > 0) = {np.mean(np.array(d) > 0):.3f}")
