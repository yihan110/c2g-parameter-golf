# C2G Leaderboard 对比表

**作者：** 阮依涵（ruanyihan） ｜ 数据来源：OpenAI Parameter Golf 官方公开 leaderboard（`README.md`）与官方基线纪录。

> BPB 越低越好。`*` 标注为**预期/参考值**，最终以真实 8×H100 复跑回填为准。

## 1. 位置对比

| 方案 | val_bpb | 说明 |
|------|--------:|------|
| **本方案（目标）** | **1.16 ~ 1.17** * | SP8192+LeakyReLU²+QK-Gain5+MuonEq-R+WD+EMA（Level 2） |
| 官方 Naive Baseline | 1.2244 | SP1024, 9L×512d, KV4, tied embeddings |
| 当前榜首（SOTA） | 1.0810 | SP8192+3-Layer Recurrence+Parallel Residuals+Legal TTT |
| Elite 20 录取门槛建议 | < 1.15 | 官方指南 |
| Level 3（全球前 50） | < 1.12 | 课程分级 |
| **Level 2（本目标）** | **< 1.18** | 课程分级 |

## 2. 逐级差距（相对基线 1.2244）

| 里程碑 | 需要的降幅 | 本方案覆盖 |
|--------|-----------|-----------|
| Level 2（<1.18） | −0.0444 | 目标（A..E 组合 ≈ −0.05~−0.06） |
| Level 3（<1.12） | −0.1044 | 预留：需再加深度递归/并行残差/TTT 等（后续） |
| SOTA（1.0810） | −0.1434 | 长期：架构级创新 |

## 3. 单项参考（公开纪录，用于拆解差值）

| 纪录 | val_bpb | 关键组件 |
|------|--------:|---------|
| SP8192+3LayerRecur+ParResid+TTT | 1.0810 | 词表+深度递归+并行残差+TTT+QK-Gain5.25 |
| SP8192+ParallelResid+ScoreFirstTTT | 1.0822 | 词表+并行残差+TTT |
| SP4096+DepthRecur+ParResid+MuonEqR | 1.0897 | 词表4096+深度递归+MuonEq-R |
| MuonEqR+DepthRecur+WD090+AllInt6 | 1.0912 | MuonEq-R+高WD+int6 |
| 4096Vocab+MLP4x+WD085 | 1.0979 | 词表4096+MLP4x+高WD |
| 11L EMA+GPTQ-lite+warmdown | 1.1228 | EMA |
| 11L XSA4+EMA+Int6MLP3x | 1.1271 | EMA+int6 |
| LeakyReLU²+TTT+ParallelMuon | 1.1194 | LeakyReLU² |
| **官方 Naive Baseline** | **1.2244** | 起步点 |

## 4. 结论

本方案以 **Level 2（< 1.18）** 为当前达成目标，组件全部有公开纪录支撑；`方案设计.md` 预留 Level 3 通道（深度递归 / 并行残差 / 合法 TTT / int6 量化），可作为真实跑通后的下一跳。
