# PertFold_CellPatch

在 VCC 2025 H1 数据上预测 CRISPRi 基因扰动的表达反应，面向 VCC 2026。

**当前版本：`r4-v2`（2026-10-22）**　·　各版本的结果与改动见 [HISTORY.md](HISTORY.md)

## 模型
PertFold3 · R4 配置：根据**已知扰动的邻居**，预测**未见扰动**的表达变化 δ。
- **候选邻居：** 从 STRING、K562 Perturb-seq、DepMap 三个来源各取 5 个最相关的训练扰动，合并后最多 10 个。
- **encoder（CellPatch）：** 基因 embedding（由 ESM2 初始化）⊙ 表达值分箱，再做交叉注意力，得到 128 个 patch。
- **cell / pert MSA：** 各 2 层，分别读入邻居扰动的细胞 bulk 和邻居的靶基因。
- **邻居权重（nbw）：** 学习每个邻居与查询扰动的相关程度，用 softmax 加权得到邻居 δ。
- **decoder：** 基因查询对 z 做交叉注意力；nbw 证据经零初始化 adapter 注入；三个头分别预测 DE、LFC、δ 分箱，损失采用 PIE 的四项。
- **训练：**
  - 阶段 1：扰动 JEPA + EP-IDM + SIGReg，3000 步；
  - 阶段 2：全部参数微调 5000 步，按 val 选权重。

## 当前结果
5 折交叉验证（全部 147 个扰动）。指标是 VCC 2026 的六项（cell-eval，5 次重复），↑ 越高越好，↓ 越低越好。各版本的结果见 [HISTORY.md](HISTORY.md)。

| 方法 | PDS↑ | MSE↓ | fidelity↑ | reach↑ | sig_jacc↑ | lfc_nmae↓ |
|---|---|---|---|---|---|---|
| PIE（Arc，官方配方） | 待测 | | | | | |
| STRING kNN（不训练） | 0.605 | 1.064 | 0.205 | 0.239 | **0.089** | 0.946 |
| **r4-v2 完整模型** | 0.617 | **1.041** | 0.233 | **0.256** | 0.088 | **0.931** |
| r4-v2 nbw 读出 | **0.629** | 1.094 | **0.313** | 0.249 | 0.059 | 0.936 |

- 相对 kNN 的 PDS：完整模型 +0.012，不显著；nbw 读出 +0.024，显著。

## 目录
| 路径 | 内容 |
|---|---|
| `pertfold3/model_v5.py` | 模型 |
| `scripts/` | 训练、预测、cell-eval 评估、汇总 |
| `run_r4.sh` / `verify_r4.sh` | 从头训练 / 复现性检查 |
| [HISTORY.md](HISTORY.md) | 每个版本的测试结果与改动 |
| [docs/NOTES.md](docs/NOTES.md) | 内部记录：代码来源、完整超参、运行方法、版本管理 |
