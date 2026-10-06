# C2G 参数高尔夫 —— 极限约束下的语言模型训练（ruanyihan）

> 课程：AI+X Elite 20 Program · SIAS University ｜ 挑战 ID：ch-20260717031359-b8wyg0
> 原始赛事：[OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf)

在 **8×H100、10 分钟训练、16MB 模型文件** 的硬约束下，训出尽可能强的语言模型，用 **BPB（Bits-Per-Byte）** 衡量（越低越好）。

本仓库是 **阮依涵（ruanyihan）** 的完整参赛/课程交付，目标 **Level 2（BPB < 1.18）**，并给出明确的提升路径与完整的实验方法论。

---

## 一句话方案

在官方 **Naive Baseline（SP-1024, 9L×512d, BPB=1.2244）** 之上，做 **可复现、有据可查** 的单点组合改进：

| 组件 | 改动 | 预期 BPB 收益（来源） |
|------|------|----------------------|
| Tokenizer | 1024 BPE → **SP8192** | −0.03 ~ −0.05（官方 leaderboard） |
| 激活 | ReLU² → **LeakyReLU(0.5)²** | −0.01 ~ −0.02（modded-nanogpt / 多个纪录） |
| Attention | QK-Gain 1.5 → **5.0** | −0.005 ~ −0.01（榜首方案） |
| 优化器 | Muon → **MuonEq-R + WD 0.04** | −0.01 ~ −0.02（modded-nanogpt） |
| 权重 | **EMA 0.99** 导出 | −0.005 ~ −0.01（多个纪录） |

**预期合计：约 1.2244 → 1.16~1.17（Level 2，< 1.18）**。所有收益均来自官方 leaderboard / 公开纪录的**已测出**差值，非臆测；最终数字须在真实 8×H100 上以 ≥3 个 seed 复跑确认（见下文）。

---

## 目录结构

```
ruanyihan_C2G_parameter_golf/
├── README.md                        ← 本文件（总览 + 跑训说明）
├── requirements.txt                 ← 依赖
├── .gitignore
├── ruanyihan_C2G_train_gpt.py       ← 训练脚本（官方基线 + Level-2 改进，全部 env 开关）
├── ruanyihan_C2G_eval_bpb.py        ← BPB 评估辅助脚本（tokenizer-agnostic 校验）
├── ruanyihan_C2G_submission.json    ← 提交元数据模板
├── ruanyihan_C2G_方案草案.md         ← 算力申请门槛文档（≥500 字，回答 4 问）
├── ruanyihan_C2G_方案设计.md         ← 方案设计（目标/技术选型/实验矩阵）
├── ruanyihan_C2G_ablation.md        ← 消融实验设计 + 预期贡献表
├── ruanyihan_C2G_leaderboard.md     ← BPB 对比表（本方案 vs 基线 vs 当前 SOTA）
├── ruanyihan_C2G_AI日志.md          ← AI 使用全过程记录（多轮迭代，可追溯）
├── ruanyihan_C2G_拿来说明.md         ← 从 nanoGPT/官方/历史方案拿了什么、改了什么
├── ruanyihan_C2G_AAR.md             ← 复盘（问题分析 / 失败经验 / 改进方案）
├── logs/
│   ├── README.md                    ← 如何产出真实 3-seed 训练日志
│   └── baseline_reference_train.log ← 官方基线真实日志（对照参考）
└── data/
    └── README.md                    ← 国内可直连的数据下载说明
```

---

## 快速开始（在租的 8×H100 / A100 云实例上）

> 本机没有 GPU，**真实训练必须在云 GPU 上执行**。国内可直连的推荐：**AutoDL**（约 ¥150/小时，H100/A100，无需翻墙）。也支持 RunPod / Lambda / 学校集群。

### 1. 环境

```bash
pip install torch>=2.3 sentencepiece numpy tqdm brotli --break-system-packages
# 可选，H100 上明显提速：
pip install flash-attn --no-build-isolation --break-system-packages
```

### 2. 下载数据（国内可直连的镜像说明见 `data/README.md`）

```bash
python3 data/cached_challenge_fineweb.py --variant sp8192   # 8B tokens 训练 + 固定 5 万文档验证集
```

### 3. 复现基线（对照用）

```bash
RUN_ID=baseline_1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 ruanyihan_C2G_train_gpt.py
# 预期：val_bpb ≈ 1.2244（官方基线）
```

### 4. 跑 Level-2 改进（SP8192 + LeakyReLU² + QK-Gain5 + MuonEq-R + WD + EMA）

```bash
RUN_ID=level2_sp8192 \
DATA_PATH=./data/datasets/fineweb10B_sp8192/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_sp.model \
VOCAB_SIZE=8192 \
NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
LEAKY_RELU_ALPHA=0.5 QK_GAIN_INIT=5.0 \
WEIGHT_DECAY=0.04 EMA_DECAY=0.99 MUON_EQ_R=1 \
MAX_WALLCLOCK_SECONDS=600 TRAIN_LOG_EVERY=50 VAL_LOSS_EVERY=200 \
SEED=42 torchrun --standalone --nproc_per_node=8 ruanyihan_C2G_train_gpt.py
```

脚本结束会打印 `final_int8_zlib_roundtrip_exact val_bpb:...` 与 `Total submission size int8+zlib: ...`，两者即最终成绩与 16MB 合规校验。

### 5. 3 个 seed 与统计显著性

成绩取 **≥3 个独立 seed** 的均值 ± 标准差（如 seed=42/314/999），并按要求与基线做显著性检验（详见 `方案设计.md`）。

---

## 16MB 合规与评分

- **成绩达成（25）**：需要真实 BPB 记录（本仓库给出完整跑训路径 + 参考基线，最终数字待真实运行回填）。
- **方法学（20）/ 产物完整性（15）/ AI使用（20）/ 复盘（20）**：见 `方案设计.md`、`ablation.md`、`AI日志.md`、`AAR.md`。

**如实声明**：训练代码、评估脚本、全部文档均在本仓库内、可运行；唯一需要真实环境执行的是 8×H100 训练本身（本机无 GPU）。因此 `submission.json` 中 `val_bpb` 以官方基线 1.2244 作为参照标注，最终成绩待你租用 GPU 后按上述命令复跑回填。

---

## 法律与合规

- 全程仅使用国内可直连资源（AutoDL 云 GPU、HuggingFace 镜像 / ModelScope 等），**不翻墙、不调用任何外部 API**，符合赛事「评估期间禁止联网」规则。
- 数据、代码均来自 OpenAI 官方开源仓库（MIT 协议），引用与改造已在 `拿来说明.md` 中逐条交代。
