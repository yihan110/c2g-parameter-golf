#!/usr/bin/env bash
# =============================================================================
# C2G 单卡一键跑训（飞桨 AI Studio / 任何 Linux 单 GPU）
# 作者：ruanyihan
#
# 目标：在没有 8xH100 的情况下，用 1 张免费 GPU 跑出【真实、可复现】的
#       BPB 对照（基线 vs 改进，各 3 个 seed），证明"边际提升"，拿到
#       课程"成绩达成"维度的分数记录。全程国内直连，不翻墙。
#
# 用法（在 AI Studio 终端里，先按 AI_STUDIO_部署指南.md 装好环境、下好数据）：
#   bash ruanyihan_C2G_run_single_gpu.sh
# 或自定义预算后运行：
#   ITERATIONS=3000 bash ruanyihan_C2G_run_single_gpu.sh
#
# 注意：免费单卡比 8xH100 慢很多，这里刻意用【固定预算 + 同一数据集】跑基线与
#       改进，得到的绝对 BPB 会比官方 1.2244 高，但【改进相对基线的差】是真实
#       可信的边际提升信号。绝对成绩若要冲 Level 2(<1.18)，仍需 8xH100（见 README）。
# =============================================================================
set -euo pipefail

# ---------- 可调参数（默认适合 16GB 显存单卡） ----------
ITERATIONS="${ITERATIONS:-3000}"
TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-65536}"   # 单卡微批 65536/8=8192 token，16GB 安全
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-65536}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-1024}"
TRAIN_SHARDS="${TRAIN_SHARDS:-5}"                    # 下载多少个训练 shard（1 shard≈100M token）
NPROC="${NPROC:-1}"                                  # 单卡就是 1

DATA_PATH="./data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
SEEDS=(42 314 999)

echo "==== C2G single-GPU runner ===="
echo "ITERATIONS=$ITERATIONS  TRAIN_BATCH_TOKENS=$TRAIN_BATCH_TOKENS  shards=$TRAIN_SHARDS"

# ---------- 1) 环境检查 ----------
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import sentencepiece, numpy, tqdm; print('deps ok')"

# ---------- 2) 下载数据（若还没有） ----------
if [ ! -d "$DATA_PATH" ] || [ -z "$(ls -A "$DATA_PATH" 2>/dev/null)" ]; then
  echo "---- downloading sp1024 data ($TRAIN_SHARDS train shards) via hf-mirror ----"
  export HF_ENDPOINT=https://hf-mirror.com
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$TRAIN_SHARDS"
else
  echo "---- data already present, skip download ----"
fi

# ---------- 3) 函数：跑一个 seed 并落日志 ----------
run_seed() {
  local name="$1"; shift
  local seed="$1"; shift
  local logdir="logs/${name}"
  mkdir -p "$logdir"
  echo ""
  echo "======== RUN $name seed=$seed ========"
  RUN_ID="${name}_seed${seed}" \
  SEED="$seed" \
  DATA_PATH="$DATA_PATH" \
  TOKENIZER_PATH="$TOKENIZER_PATH" \
  TRAIN_BATCH_TOKENS="$TRAIN_BATCH_TOKENS" \
  VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
  TRAIN_SEQ_LEN="$TRAIN_SEQ_LEN" \
  ITERATIONS="$ITERATIONS" \
  MAX_WALLCLOCK_SECONDS=0 \
  TRAIN_LOG_EVERY=200 \
  VAL_LOSS_EVERY=1000 \
  "$@" \
  torchrun --standalone --nproc_per_node="$NPROC" ruanyihan_C2G_train_gpt.py 2>&1 | tee "$logdir/seed${seed}.log"
}

# ---------- 4) 基线（9L，关闭改进，匹配 naive baseline 配置） ----------
BASE_ARGS="VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 \
LEAKY_RELU_ALPHA=0 QK_GAIN_INIT=1.5 WEIGHT_DECAY=0 EMA_DECAY=0 MUON_EQ_R=1"

# ---------- 5) 改进（9L，同架构，只开 Level-2 改进：LeakyReLU2+QK5+WD+EMA） ----------
IMP_ARGS="VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 \
LEAKY_RELU_ALPHA=0.5 QK_GAIN_INIT=5.0 WEIGHT_DECAY=0.04 EMA_DECAY=0.99 MUON_EQ_R=1"

echo ""
echo "==== Starting baseline runs (3 seeds) ===="
for s in "${SEEDS[@]}"; do
  run_seed baseline "$s" env $BASE_ARGS
done

echo ""
echo "==== Starting improved runs (3 seeds) ===="
for s in "${SEEDS[@]}"; do
  run_seed improved "$s" env $IMP_ARGS
done

# ---------- 6) 汇总最终 val_bpb ----------
echo ""
echo "================ SUMMARY (final_int8_zlib_roundtrip_exact val_bpb) ================"
for s in "${SEEDS[@]}"; do
  echo -n "baseline seed$s : "
  grep "final_int8_zlib_roundtrip_exact" "logs/baseline/seed${s}.log" | tail -1
done
for s in "${SEEDS[@]}"; do
  echo -n "improved seed$s : "
  grep "final_int8_zlib_roundtrip_exact" "logs/improved/seed${s}.log" | tail -1
done
echo ""
echo "跑完把上面对应数字填进 ruanyihan_C2G_submission.json 和 AAR，再 git push 即可。"
