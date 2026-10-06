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

## 诚实声明

本仓库**不包含伪造的训练日志**。`baseline_reference_train.log` 为官方基线真实日志；3 个定稿 seed 日志需由你在真实 8×H100 上按 README 命令复跑后填入，回填 `submission.json` 的真实 `val_bpb`。
