# PertFold3 · R4 版本（2026-10-22 整理）

R4 是 2026-10-20 v6 实验中在 H1 上选出的配置：N1u + mul_std + 阶段 1 "JEPA + EP-IDM + SIGReg"。本目录是它的完整代码与复现入口。

## 一、代码来源与验证
- 代码**原样复制**自 `pertfold/`（2026-10-22）。模型文件里的其他开关（v3x / v5 / v6 消融 / v7）都保留，R4 只用到其中一部分（见第三节）。之所以不删减：删减后就不再是经过验证的这份代码。
- **验证（`verify_r4.sh`）：** 用本目录的代码对原 R4 权重（种子 0）重新预测，与 2026-10-20 当时的预测逐元素比对。结果：yhat、de_prob 的最大差都是 0，**逐位相同**。
- 本目录只改了两处路径：
  - `sys.path` 指向 `pertfold_r4`；
  - `train_h1_v5.py` 的默认输出改为 `pertfold_r4/runs`。
- 数据仍读取 `pertfold/data`（只读）。

## 二、目录
| 路径 | 内容 |
|---|---|
| `pertfold3/model_v5.py` | 模型（Config3 / PertFold3 / CellPatch / MSA / NbScorer / Decoder / pie_loss） |
| `scripts/train_h1_v5.py` | 训练（数据 H1Data、两阶段训练、val 选权重） |
| `scripts/predict_h1_v5.py` | test 预测：`--readout exp`（完整模型，δ 取分箱期望值）/ `nbw`（只用学到的邻居权重） |
| `scripts/eval_model_h1.py`、`benchmark_h1.py` | cell-eval2（VCC 2026 六项）打分，用 `.venv-eval` 环境 |
| `scripts/compare_split.py` | 汇总 split_811 test 上的六项 |
| `scripts/diag_v6.py` | encoder 诊断（扰动检索、样本间余弦、RankMe） |
| `checkpoints/R4_s0`、`R4_s1` | 原 R4 两个种子的权重与配置（软链接到 `pertfold/runs3/h1_v6/...`）；`ref_pred_exp_rep0.npz` 是原预测，验证用 |
| `run_r4.sh` | 从头重训 R4 → 预测 → cell-eval → 汇总 |
| `verify_r4.sh` | 复现性检查（见第一节） |

## 三、R4 的完整设置（与原始命令行逐项相同）
**数据：**
- H1（VCC 2025），147 个扰动，2125 个基因（2000 个 HVG + 靶基因）；
- 样本为 32 个细胞的 bulk；
- 划分 `pertfold/data/h1/split_811.json`：train 118 / val 15 / test 14。

**模型：**
- **候选邻居（并集池）：** STRING 直接分 ≥ 0.7、K562 Perturb-seq、DepMap 三个来源，各取前 5 个训练扰动，合并后最多 10 个（`--relation string+perturbseq_k562+depmap`）。
- **encoder（CellPatch）：**
  - 基因 token = 基因 embedding（ESM2 逐维标准化后初始化，可训练）⊙ LN(AutoBin((x − μ) / σ))，μ、σ 取 bulk 的统计量（`--value-enc mul --value-norm --norm-bulk`）；
  - 128 个 latent 做交叉注意力，得到 128 个 patch。
- **cell_MSA / pert_MSA：** 各 2 层，分别读邻居扰动的细胞 bulk 和邻居的靶基因；扰动 token = Linear(ESM2) 拆成 8 个。
- **扰动预测：** 交叉注意力注入扰动 → 4 层 TransformerBlock → z。
- **邻居学习权重（N1，`--nb-learn`）：**
  - 打分 MLP 的输入是 [e_q, e_k, e_q ⊙ e_k, 关联特征]，用 MSE 回归 c_k（邻居与查询扰动真实 δ 的相关，只在训练扰动上算）；
  - a_k = softmax(s / 0.1)。
- **decoder：**
  - 基因查询对 z 做交叉注意力；
  - PIE 式证据（对照均值、ctx、str、nbw）经零初始化 adapter 注入，每类以 0.25 的概率整类丢弃；
  - 三个头：DE、LFC、δ 分箱（15 个箱，宽 ln 1.1）；损失为 PIE 的四项。

**训练：**
- **阶段 1：** 3000 步，lr 3e-4，batch 32。
  - 扰动 JEPA：目标 = 在线 CellPatch(扰动后 bulk y) 停梯度，y 的 147 个靶基因列先换成对照均值（`--mask-targets`），MSE（`--jepa-mode sg`）；
  - EP-IDM：由池化后的 [z_x, z_y] 分类出是哪个扰动，CE，权重 1（`--idm 1`）；
  - SIGReg 0.1：加在池化后的 x、y latent 上（`--jepa-sigreg 0.1`）；
  - 另加邻居打分的辅助 loss。
- **阶段 2：** 全部微调 5000 步，lr 1e-4（`--stage2 ft:1e-4`）；每 25 步在 val 上评估，按 val pearson_exp 保存最好的权重。
- **其他：** AdamW（weight decay 0.01），bf16 autocast，梯度裁剪 1，种子 0 / 1。

## 四、结果（split_811 test 14 个扰动，cell-eval 5 次重复）
| | PDS↑ | MSE↓ | fidelity↑ | reach↑ | sig_jacc↑ | lfc_nmae↓ |
|---|---|---|---|---|---|---|
| R4 原跑（种子 0 / 1 平均），完整模型 | 0.736 | 0.916 | 0.247 | 0.207 | 0.089 | 0.892 |
| R4 原跑，nbw 读出 | 0.750 | 0.927 | 0.335 | 0.191 | 0.038 | 0.884 |
| R4 同代码重跑（2026-10-21），完整模型 | 0.711 | 0.940 | 0.216 | 0.196 | 0.090 | 0.908 |
| STRING kNN（不训练） | 0.712 | 0.875 | 0.236 | 0.201 | 0.083 | 0.865 |

- **重跑噪声：** 同配置、同种子重跑，PDS 从 0.736 / 0.735 变为 0.705 / 0.716。两次的预测彼此相关 0.93，与真实 δ 的相关 0.369 对 0.366，几乎相同。差距主要来自 3 个扰动的 PDS 排名挪了一两位（14 个扰动时 PDS 的步长是 1/13）。
- **含义：** split_811 上 0.03 以内的差别都属于噪声，结论要用 5 折（147 个扰动）。
- **复现：** 原 R4 权重可以随时复现 0.736（verify_r4.sh 通过）；重新训练不是逐位确定的。
- **5 折（147 个扰动，2026-10-22）：** R4 完整模型 PDS 0.612，STRING kNN 0.605，差 +0.007，不显著；R4 nbw 读出 0.628，比 kNN 高 +0.023，显著。5 折的 PDS 尺度与 split_811 不同。
- 详细实验记录见 `pertfold/ABLATION_summary_1022.md`。

## 五、运行
```bash
# 复现性检查（约 1 分钟）
GPU=0 ./verify_r4.sh
# 从头重训 R4 两个种子 + 预测 + cell-eval（约 3 小时；每个训练约 21 GB 显存）
GPU=0 ./run_r4.sh 0 1
```
**环境：**
- 训练与预测：`/cluster2/home/hanwen/miniconda3/envs/vcc26`；
- cell-eval：`/cluster2/home/hanwen/vcc_26/backbone/.venv-eval`。

**数据依赖（只读）：**
- `pertfold/data/h1`（X.npy、pert.npy、labels.npz、meta.json、split_811.json）；
- `pertfold/data/esm`、`pertfold/data/string`、`pertfold/data/relations`；
- `splithalf/data/H1.h5ad`（评估用的原始计数）；
- `backbone/cell_eval_vcc2026_base.yaml`；
- `pertfold/runs3/split_811_ref`（STRING kNN 基线）。

## 六、版本管理（GitHub）
- **仓库：** https://github.com/HanwenZhu98/PertFold_CellPatch（private），分支 `main`。
- **认证：** HTTPS + classic token（`repo` 权限），已存在 `~/.git-credentials`，所有节点通用；token 过期后重新生成，下次 push 时输入新的即可。
- **只管代码：** `checkpoints/`、`runs/`、`*.npz`、`*.pt`、`*.log` 等都在 `.gitignore` 里，不上传。结果和权重留在集群上，记录里写明产生它们的 commit。

**日常保存：**
```bash
cd /cluster2/home/hanwen/vcc_26/pertfold_r4
git add -A
git commit -m "改了什么、为什么"
git push
```

**效果好的版本打标签：**
```bash
git tag -a r4-v1 -m "一句话说明：配置 + 关键结果（如 5 折 PDS）"
git push --tags
```
- 标签命名：`r4-v<序号>`，按顺序递增；说明里写结果，方便以后对照。
- 跑重要实验前先 commit，在日志里记下 `git rev-parse --short HEAD`。

**查看 / 回退：**
```bash
git log --oneline --decorate     # 历史与标签
git tag -n                       # 所有标签及说明
git diff r4-v1                   # 当前代码与某个标签的差别
git switch --detach r4-v1        # 临时切到某个版本（看完 git switch main 回来）
```

**版本记录：**
| 标签 | 日期 | 说明 |
|---|---|---|
| — | 2026-10-22 | 初始提交：R4 代码原样整理（`878978c`） |
