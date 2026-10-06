# 飞桨 AI Studio 部署指南（免费 GPU，国内直连，不翻墙）

作者：ruanyihan ｜ 目标：用**免费单卡**跑出真实、可复现的 BPB 对照，拿到课程"成绩达成"分数记录。

> 免费单卡（V100/A100）比 8×H100 慢，绝对 BPB 会高于官方 1.2244；但**"基线 vs 改进"在相同预算下的差**是真实可信的边际提升信号。要冲 Level 2（<1.18）仍需 8×H100（可用课程 $25 券或 AutoDL）。

---

## 第 1 步：注册 + 领免费算力卡

1. 打开 **aistudio.baidu.com**（百度飞桨 AI Studio，国内平台，手机号注册即可）。
2. 完成注册/登录后，进 **「算力卡」** 或 **「个人中心 → 算力卡」**，领取平台发放的**免费 GPU 算力卡**（新用户通常有免费额度，V100/A100）。
3. 免费额度小时数有限，**先跑通小预算再放大**，别一次烧完。

## 第 2 步：新建项目 / Notebook，选 GPU 环境

1. 顶部 **「项目」→「创建项目」**，类型选 **Notebook**（或"AI 训练"）。
2. 环境选 **GPU 版本 + PyTorch 镜像**（`pytorch` 或 `paddlepaddle+torch` 均可，只要含 torch）。
3. **启动环境时选 A100 优先**（若免费卡给的是 V100，bf16 会变慢但仍能跑）。
4. 打开终端（Terminal）。

## 第 3 步：把代码拉进来

在终端里（两种方式任选）：

```bash
# 方式 A：直接 clone 你的 GitHub 仓库（国内直连 github.com，一般可用）
git clone https://github.com/yihan110/c2g-parameter-golf.git
cd c2g-parameter-golf

# 方式 B：若 clone 很慢，就在 AI Studio 网页上传 zip（下载你桌面那个文件夹打包后上传）
```

> 若 GitHub 在 AI Studio 里连不上，就用手动上传（方式 B）。**这不涉及翻墙**，只是网络偏好。

## 第 4 步：装依赖（国内 pip 源）

```bash
pip install torch numpy sentencepiece tqdm brotli --index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 若环境已带 torch，可省略 torch；flash-attn 可选（A100 上加速，装不上不影响跑通）
```

## 第 5 步：下载数据（国内 hf-mirror 镜像，只下 5 个训练 shard + 固定验证集）

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 5
```

> 只下 5 个 shard（≈1GB）是为了免费环境快速可用；数据放 `./data/datasets/fineweb10B_sp1024/`。
> 验证集始终是固定的 `fineweb_val_*`，用于算 BPB，**禁止用它调参**。

## 第 6 步：跑（基线 + 改进 × 3 seed，自动落日志）

```bash
bash ruanyihan_C2G_run_single_gpu.sh
```

- 默认：`ITERATIONS=3000`、`TRAIN_BATCH_TOKENS=65536`（16GB 显存安全）。
- 想加快/放慢：`ITERATIONS=2000 bash ruanyihan_C2G_run_single_gpu.sh`
- 想少下数据：`TRAIN_SHARDS=3 bash ruanyihan_C2G_run_single_gpu.sh`
- 跑完终端会打印 6 条 `final_int8_zlib_roundtrip_exact val_bpb`（基线与改进各 3 seed）。

> 若 OOM（显存不足）：把 `TRAIN_BATCH_TOKENS` 调小到 32768，或 `TRAIN_SEQ_LEN=512`。
> 若 V100 太慢：先 `ITERATIONS=1000` 跑通流程，再逐步加大。

## 第 7 步：回填 + 推送

1. 把 6 个真实 `val_bpb` 填进 `ruanyihan_C2G_submission.json`（替换 seed_results，更新 val_bpb 均值）。
2. 日志已在 `logs/baseline/seed*.log` 与 `logs/improved/seed*.log`，需要的话复制到仓库根 `logs/`。
3. 用 `python3 ruanyihan_C2G_eval_bpb.py <3个改进值> --baseline <3个基线均值>` 做显著性校验。
4. 本地 git add/commit/push，或把回填后的文件下载回桌面再推。

---

## 常见问题

| 现象 | 处理 |
|------|------|
| 下载数据慢/失败 | 确认已 `export HF_ENDPOINT=https://hf-mirror.com`；再不行用 ModelScope（见 data/README.md） |
| 显存 OOM | `TRAIN_BATCH_TOKENS=32768` 或 `TRAIN_SEQ_LEN=512` |
| bf16 在 V100 很慢 | 属正常（V100 无 bf16 硬件），优先选 A100 算力卡 |
| 跑的时间超免费额度 | 减 `ITERATIONS`/`TRAIN_SHARDS`，先拿一条真实记录再放大 |
| 数字明显高于 1.18 | 正常（单卡+小预算），报告时写清硬件与预算即可，关注相对基线的差 |

---

## 诚实说明

- 本路径产出的是**真实、可复现**的成绩记录（比造假强得多），满足课程"成绩达成 + 有分数记录 + 边际提升"的信号。
- 若要冲击 Level 2 官方线（<1.18），仍需 8×H100：用课程 **$25 算力券**（已备好 `方案草案.md`）或 AutoDL。
