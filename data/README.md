# data/ 目录说明（国内可直连下载）

## 需要的数据

1. **tokenizer**：`fineweb_1024_bpe.model`（基线）与 `fineweb_8192_sp.model`（Level-2，SP8192）。
2. **数据集**：FineWeb 固定 5 万文档验证集（`fineweb_val_*`）+ 训练 shard（`fineweb_train_*`）。

官方用 HuggingFace 托管。**你网络不能翻墙**，因此用国内可直连的镜像/通道：

| 通道 | 地址 | 说明 |
|------|------|------|
| **ModelScope（魔搭）** | `modelscope.cn` | 国内可直连，可用 `modelscope download` 拉取镜像数据 |
| **HuggingFace 镜像站** | `hf-mirror.com` | 国内可直连的 HF 镜像，`export HF_ENDPOINT=https://hf-mirror.com` 后即可用官方脚本下载 |
| AutoDL 内置 | 实例内通常已预置 | 若实例有缓存可直接用 |

### 推荐做法（hf-mirror）

```bash
export HF_ENDPOINT=https://hf-mirror.com
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80
```

### 备选（ModelScope 手动）

在 ModelScope 搜索 `fineweb10B` 或 `fineweb` 镜像仓库，用 `modelscope download --model <repo> --local_dir ./data/datasets/` 下载后，把 shard 放入 `./data/datasets/fineweb10B_sp8192/`、tokenizer 放入 `./data/tokenizers/`。

## 说明

- 验证集是**固定**的"前 5 万文档"，用于算 BPB；**禁止**用验证集调参（会过拟合，官方复核会翻车）。
- 评估期间禁止联网，因此下载务必在训练前完成并本地缓存。
