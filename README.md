# STARPath

基于官方 [MMP](https://github.com/mahmoodlab/MMP) 整理的生存预测实验框架，包含 STARPath、MMP、DIMAF、SlotSPE、SurvPath 和单模态基线。STARPath 主干集中在 [modal_starpath.py](src/mil_models/modal_starpath.py)，保留完整的 coarse、route atlas 与动态 TITAN memory 交互。

支持 **BRCA、BLCA、STAD、HNSC、LUAD、LUSC、CRC、KIRC**。CRC 对应 RNA 和 split 中的 COADREAD；启动脚本统一使用 CRC。

## 默认协议

| 设置 | 默认值 |
|---|---|
| 生存终点 / 划分 | DSS / `train_test`，读取 `SPILT_DSS_test` |
| 主损失 | NLL；所有模型也可选 Cox |
| NLL batch size | 1；**DIMAF 默认 64** |
| Cox batch size | 默认目标 64 位患者；末尾风险集按患者总数调整 |
| Checkpoint | `last`，实际训练结束的最后一个 epoch |
| Early stopping | 关闭；与 checkpoint 选择独立 |
| 训练轮数 / 学习率 / 随机种子 | 10 / `1e-4` / 1 |
| 交叉验证 | 5 折，编号 0–4 |
| STARPath 注入层 / 可训练层 | `2,4` / `2,3,4,5`，从 0 开始编号 |
| STARPath 每张切片 patch 上限 | 512；完整 morphology 和 atlas 在抽样前计算 |

`--batch-size` 可覆盖模型支持的默认值，但 **STARPath、ABMIL、TransMIL、MCAT、SurvPath、SlotSPE 的 NLL 必须使用 batch=1**；可通过 `accum_steps` 做梯度累积增大有效批量。DIMAF 的 NLL 默认是真正的 batch=64，Cox 也默认 64。本工程统一 NLL 使用 mean，并保留 DIMAF 独立的 distance-correlation 辅助项；与官方纯 NLL 协议的区别见 [模型来源](docs/model_provenance.md)。

所有模型的 Cox 训练都使用 `EventAwareRiskSetBatchSampler`，为风险集安排可比较的事件患者，并将单患者尾批与前一批重组；每位患者每轮恰好出现一次，不丢弃或重复患者。上述单患者 WSI 模型使用逻辑风险集：先逐患者 forward 汇总风险并计算 Cox 梯度，再恢复 RNG 逐患者第二次 forward/backward。普通 NLL 梯度累积不代替 Cox 风险集。

## 数据和原型

根目录集中在 [src/configs/data_paths.json](src/configs/data_paths.json)。RNA 默认使用 `/data2/lama/self_unify_RNA`，模型保持既有 recipe：

| 模型 | RNA |
|---|---|
| MMP-trans、MMP-OT、SurvPath、DIMAF | `mmp_set/hallmarks` |
| STARPath | 默认 `mmp_set/hallmarks`；可显式选择 `--rna-set surv_set` 使用其 hallmarks |
| MCAT、MLP、SNN、S-MLP | `surv_set/raw_rna_data/combine` |
| SlotSPE | `slotspe` |
| ABMIL、TransMIL、TITAN | 不使用 RNA |

| `--endpoint` | `--split-mode` | 数据根目录 |
|---|---|---|
| `dss` | `train_test` | `/data2/lama/SPILT_DSS_test` |
| `dss` | `train_val_test` | `/data2/lama/SPILT_DSS_val_test` |
| `os` | `train_test` | `/data2/lama/SPILT_OS_test` |
| `os` | `train_val_test` | `/data2/lama/SPILT_OS_val_test` |

实际读取各根下的 `src/splits/survival/TCGA_<COHORT>_overall_survival_k=<fold>/`。终点由配置决定；OS 同时切换 split 根和标签列。

MMP-trans、MMP-OT、DIMAF、STARPath 需要形态原型。按照官方 MMP 聚类协议，**只使用当前 endpoint、当前划分、当前折的 train.csv**；val 不参加拟合，也不自动复用同名旧 split 的原型。原型保存在 `artifacts/prototypes/<endpoint>/<split-mode>/<cancer>/fold_<k>/<fingerprint>/`，使用时校验训练名单和文件指纹。

## 启动实验

以下命令从仓库根执行。当前机器的 shell 脚本默认使用 `/data2/lama/miniconda3/envs/MIL/bin/python`；其他环境可设置 `PYTHON_BIN`。环境定义见 [env.yaml](env.yaml)。`--dry-run` 展示最终配置和命令，不开始训练或聚类。

```bash
cd /data2/lama/STARPath
bash src/scripts/prototype/cancer.sh BRCA --dry-run
bash src/scripts/survival/BRCA/starpath.sh --dry-run
```

准备默认 DSS train/test 原型，然后运行 STARPath 五折：

```bash
bash src/scripts/prototype/cancer.sh BRCA
bash src/scripts/survival/BRCA/starpath.sh
```

一次构建 DSS/OS × train-test/train-val-test × 八癌种 × 五折的全部 160 份原型：

```bash
python tools/build_prototype_matrix.py --gpu 5 --jobs 4
```

`--gpu` 指定可用 GPU，`--jobs` 是并发构建数；合格原型自动复用。每折日志和批量进度保存在 `artifacts/prototypes/build_runs/`，全部核验通过后生成 `index.csv` 和 `index.json`。当前生存流程不使用本仓库的 `src/splits`，该目录仅供保留的原 MMP embedding 示例使用。

每个癌种都有以下 13 个模型脚本：`starpath.sh`、`mmp_trans.sh`、`mmp_ot.sh`、`dimaf.sh`、`slotspe.sh`、`survpath.sh`、`abmil.sh`、`transmil.sh`、`mcat.sh`、`mlp.sh`、`snn.sh`、`s_mlp.sh`、`titan.sh`。例如：

```bash
bash src/scripts/prototype/cancer.sh LUSC
bash src/scripts/survival/LUSC/dimaf.sh --loss nll
bash src/scripts/survival/CRC/slotspe.sh --loss cox --batch-size 64
```

若原型构建使用 `--mode kmeans`，训练时须匹配 `--prototype-mode kmeans`；其他原型参数也应一致。

使用 OS、验证集、early stopping，并选择验证集 C-index 最好的 checkpoint：

```bash
bash src/scripts/prototype/cancer.sh BRCA --endpoint os --split-mode train_val_test
bash src/scripts/survival/BRCA/starpath.sh \
  --endpoint os --early-stopping 1 --checkpoint best \
  --checkpoint-metric c_index --es-metric loss --es-patience 5 --max-epochs 30
```

开启 early stopping 或选择 `best` 都会强制使用对应的 `train_val_test` 划分。**early stopping 决定何时停止，不自动改变 checkpoint 选择**：`--early-stopping 1 --checkpoint last` 评估实际停止时的最后一轮；`--early-stopping 0 --checkpoint best` 训练到设定轮数，再加载验证集最优轮。也可显式指定 `--split-mode train_val_test`，保留 `last` 并关闭 early stopping。test 不参与选轮或早停。

修改 STARPath 注入层和可训练层：

```bash
bash src/scripts/survival/BRCA/starpath.sh \
  --inject-layers 1,2,4 --trainable-layers 1,2,3,4,5
bash src/scripts/survival/BRCA/starpath.sh \
  --inject-layers 2 --trainable-layers none
```

动态 memory 在**最后一次注入的前一个 block**结束后写回一次：默认 `2,4` 在 block 3 写回；`2` 在 block 1 写回；`1,2,4` 仍在 block 3 写回。单独指定 `0` 没有前置 block，会报错。可训练层独立选择，`none` 表示冻结全部 TITAN 权重。

统一 Python 入口是 [tools/run_survival.py](tools/run_survival.py)，例如在已激活环境中执行 `python tools/run_survival.py --cancer BRCA --model starpath --dry-run`。模型专用参数可通过 `--` 传入底层训练入口；例如关闭 DIMAF 的解耦辅助项：

```bash
bash src/scripts/survival/BRCA/dimaf.sh --loss nll -- --dimaf_disentanglement_weight 0
```

STARPath 的 NLL 保持每次一位患者，累积 8 次再更新参数：

```bash
bash src/scripts/survival/BRCA/starpath.sh --loss nll --batch-size 1 -- --accum_steps 8
```

脚本环境变量也可直接调整，如 `LOSS_FN`、`BATCH_SIZE`、`SURVIVAL_ENDPOINT`、`CHECKPOINT_SELECTION`、`EARLY_STOPPING`、`MAX_EPOCHS`、`FOLDS`；显式 CLI 选项优先。`--folds 0` 只运行一折；`--data-config <json>` 选择另一份完整路径配置。

## 结果与目录

结果按协议、癌种、模型、配置指纹和本次 run 分隔，例如：

```text
results/DSS_standard_test/BRCA/STARPath/nll_bs1_last_<config-id>/<run-id>/
  config.json
  run_manifest.json
  fold_0/summary.csv
  fold_0/history.jsonl
  fold_0/last_checkpoint.pth
  fold_0/checkpoint_selection.json
  ...
  cv_summary.csv
  cv_summary.json
```

有验证集时另存 `best_checkpoint.pth`。`cv_summary.csv` 包含本次各折 C-index、均值和**样本标准差（ddof=1）**；单折标准差留空。仅汇总本次明确请求且全部成功的折，`cv_summary.json` 标明是否完成五折，历史结果不会混入新汇总。

| 位置 | 用途 |
|---|---|
| `src/mil_models/modal_starpath.py` | STARPath 主干、共享形态统计、患者聚合 |
| `src/mil_models/TITAN/` | 普通 TITAN 编码器、独立本地资源与预计算 embedding |
| `src/mil_models/TITAN_STARPath/` | STARPath 专用 callback 编码器及独立资源 |
| `src/mil_models/model_factory.py`、`survival_adapter.py` | 模型创建与公共生存损失接口 |
| `src/wsi_datasets/`、`src/training/` | 数据对齐、归一化、采样和训练 |
| `src/scripts/`、`tools/` | 八癌种入口、原型构建与审计 |
| `tests/models/`、`tests/data/`、`tests/training/`、`tests/scripts/` | 行为与协议测试 |

全套 unittest 从根目录运行；`-t .` 防止 `tests/training` 遮蔽训练主包：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /data2/lama/miniconda3/envs/MIL/bin/python -m unittest discover -s tests -t . -v
```

STARPath 与旧默认 C 的 671 个 `state_dict` 键一致；受控同权重 CPU 对照中，关键前向张量和参数梯度最大绝对差均为 0。真实 TITAN 权重也已分别从两个目录加载并验证。这些结构与数值检查不代表已完成全部癌种的五折训练。

更多说明：[STARPath 架构](docs/starpath_architecture.md)、[模型来源](docs/model_provenance.md)、[数据与划分](docs/data_and_splits.md)、[原型协议](docs/prototypes.md)、[工具用途](docs/tools.md)。

本仓库保留 MMP 作者归属及原始 [LICENSE.md](LICENSE.md)（CC BY-NC-SA 4.0）。第三方模型和资源按各自上游条款提供，详见模型来源文档。使用 MMP 方法或相关代码时请引用：

```bibtex
@inproceedings{song2024multimodal,
  title={Multimodal Prototyping for cancer survival prediction},
  author={Song, Andrew H and Chen, Richard J and Jaume, Guillaume and Vaidya, Anurag Jayant and Baras, Alexander and Mahmood, Faisal},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```
