# logs/ 目录说明

本目录存放训练日志。**评分要求"≥3 个独立 seed 的完整日志"。**

## 当前内容

| 文件 | 性质 |
|------|------|
| `baseline_reference_train.log` | **OpenAI 官方 Naive Baseline 的真实日志**（来自官方 records），作对照参考，非本方案伪造 |
| `README.md` | 本文件 |

## 你需要生成的定稿日志（在云 GPU 上，见根 README）

真实跑 3 个 seed（如 SEED=42/314/999）后，脚本自动在运行目录 `logs/{RUN_ID}.txt` 生成日志，**请复制回本目录**：

```
logs/seed42.log
logs/seed314.log
logs/seed999.log
```

每个日志应含：`val_bpb`、`final_int8_zlib_roundtrip_exact val_bpb`、artifact 字节、train_time、step_avg、seed、PyTorch 版本。

## 免费单卡跑法生成的日志（见根 README + `AI_STUDIO_部署指南.md`）

用 `bash ruanyihan_C2G_run_single_gpu.sh`（飞桨 AI Studio）跑完后，日志自动落在：

```
logs/baseline/seed42.log   logs/baseline/seed314.log   logs/baseline/seed999.log
logs/improved/seed42.log   logs/improved/seed314.log   logs/improved/seed999.log
```

这些是**真实可复现**的成绩记录（单卡 + 固定预算），把最终 `val_bpb` 回填到 `submission.json` 即可。

## 诚实声明

本仓库**不包含伪造的训练日志**。`baseline_reference_train.log` 为官方基线真实日志；定稿 seed 日志需由你在真实 GPU 上按 README / AI Studio 指南复跑后填入。
