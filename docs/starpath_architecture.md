# STARPath 架构与迁移边界

主实现为 [`modal_starpath.py`](../src/mil_models/modal_starpath.py)，合并旧实验的 `STARPath.py`、`starpath_coarse_utils.py`、`STARPathPatientAdapter` 和三个 slide aggregator。TITAN 源码与资源保留在独立的 `TITAN_STARPath/`。

当前只保留完整的 coarse + route atlas + dynamic TITAN 方案，即旧 C。A/B 路径、variant 参数、旧 unfreeze bool 及静态 post-block residual 接口已移除；注入层和可训练层仍可独立调整。模型构造时才导入 TITAN，dataset 导入共享形态统计函数不会触发 Transformers 或权重加载。

## 数据流

1. **完整切片统计。** 当前折 train-only 聚类得到 16 个固定、768 维形态原型。全部 CONCH patch 按最近欧氏距离分配，计算形态均值、occupancy、valid mask，以及各形态锚点的完整 pseudo-ST 均值和 counts。空 slot 保留原 ID，以精确零和无效 mask 表示。
2. **同步抽样。** 完整 morphology/ST atlas 统计结束后，WSI、坐标和 ST 才使用同一组 patch 索引抽样。默认每张切片最多 512 patches，评估抽样固定；完整 atlas 不随 patch cap 改变。
3. **Coarse conditioning。** 病人 bulk RNA 经 50 个独立 Hallmark SNN 编码，再与完整形态 tokens 做 KL-UOT；传输后的分子 context 以有界残差调制 TITAN 输入。coarse RNA 每位病人只编码一次，各切片共享结果，包括训练时 AlphaDropout 的同一次采样。
4. **Fine regions 与 atlas。** 区域构建只读取原始 patch features 和坐标，默认 12 个区域，coarse 残差不改变边界。抽样 ST 按区域聚合并编码为通路 tokens；完整 atlas 提供形态锚定的 routing cost 调整。
5. **Regional KL-UOT。** bulk RNA 通路、区域 ST 通路、形态和 occupancy 确定区域传输，生成分子 memory values 和 support；区域形态与坐标形成 lookup keys。
6. **一次 TITAN 编码。** TITAN 接收 coarse-conditioned patches，pre-block callback 读取区域 memory，post-block observer 在指定位置更新 lookup keys。分子 values 在全部读取中保持不变。
7. **病人输出。** 各切片独立编码，再按患者聚合，默认算术均值；gated attention 与 set transformer 可显式选择。分类头输出 NLL 的离散时间 logits 或 Cox 的单个 log-risk，公共 adapter 处理主损失以及 region、alignment 辅助项。

RNA 保持旧 recipe 语义。默认 `mmp_set` 使用训练集逐基因标准化的同一视图供 coarse/fine 使用；显式选择 `surv_set` 时，保留 coarse 标准化与 fine 全局范围转换两视图。归一化器和基因轴来自训练集，val/test 只应用已确定的转换。

STARPath 的 NLL 按单患者 forward，要求 batch=1；`accum_steps` 可增大有效更新批量。Cox 使用默认 64 位患者的逻辑风险集，由 `EventAwareRiskSetBatchSampler` 分组，单患者尾批与前一批重组，每轮不丢弃或重复患者。训练先逐患者 forward 得到整组 Cox 风险梯度，再恢复 RNG 逐患者重算并 backward，避免同时保留整组切片的计算图。这与普通 NLL 梯度累积的损失定义不同。

## 注入和解冻

层编号从 0 开始，当前 TITAN 有 6 个 vision blocks。动态 memory 在最后一次注入前一个 block 结束后写回一次。

| 注入层 | 写回位置 | 默认可训练层 |
|---|---|---|
| `2,4` | block 3 结束后 | `2,3,4,5` |
| `2` | block 1 结束后 | 同上，除非显式修改 |
| `1,2,4` | block 3 结束后 | 同上，除非显式修改 |

默认配置与旧 C 的计算顺序相同。列表不能为空或越界；单独的 `0` 没有前置 block，明确拒绝。可训练层不必等于注入层；`--trainable-layers none` 冻结全部 TITAN 参数。text encoder、投影等未选中参数保持冻结，STARPath 新模块继续训练。

从仓库根运行：

```bash
bash src/scripts/survival/BRCA/starpath.sh --inject-layers 1,2,4 --trainable-layers 1,2,3,4,5 --dry-run
bash src/scripts/survival/BRCA/starpath.sh --inject-layers 2 --trainable-layers none --dry-run
bash src/scripts/survival/BRCA/starpath.sh --dry-run -- --starpath_slide_agg gated
```

## 数值与数据约束

保留 UOT 的 FP32/log-domain、eps/clamp、有效 mask、零质量和低 support 处理。这些是数值与模型机制，并非历史接口 fallback。

保留 WSI/ST patch 数、source coordinates、基因顺序、正整数 `patch_size_lv0`、原型维度、finite 值及 atlas counts/valid 的校验。TITAN callback 辅助数据和 patch 使用相同 sparse grid、CLS 和 background-mask 映射；pre-block 残差不写入 CLS，post-block observer 不得替换 hidden tensor。

普通 TITAN 从 `TITAN/` 读取官方本地资源和预计算 embedding；STARPath 从 `TITAN_STARPath/` 读取私有源码、config、safetensors 和 tokenizer。两个目录独立保存资源，没有跨目录符号链接或缺文件后转用另一分支的机制。

## 验证记录

旧参考为 `Path2Space_stage_exp/Survival_Exp_Path2Space_v5_layer_ablation_2345/Survival_Exp_Path2Space_v5_layer_ablation_2345`；原项目未修改。

- 默认配置的新旧 **671 个 `state_dict` 键**一致，包含参数和 buffers。
- 同随机种子、同权重、相同输入的 CPU 对照：初始化张量、关键前向输出/中间量、所有已计算参数梯度最大绝对差均为 **0**。全模型对照使用相同的六层 TITAN 测试替身，检查 STARPath 数学与 callback 顺序。
- 另用真实本地权重独立加载官方和私有 TITAN：无 callback 的 4-patch 输出均为 `[1,768]`，最大绝对差为 **0**；私有 `[2,4]` callback、6 层 observer 和辅助 token 对齐验证通过。
- 持久化测试覆盖 C 梯度、独立解冻、可变注入层、UOT 解析解/KKT/半精度、患者聚合、完整 atlas、同步抽样、坐标与基因轴约束。

这些验证针对结构和受控数值行为，不等价于完成真实队列的训练性能复现。
