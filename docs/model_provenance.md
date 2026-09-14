# 模型来源、修改范围与核验

以官方 MMP 为基础，迁入旧 Path2Space 实验的 STARPath 与基线接入层。本次固定核验的上游版本如下；未固定独立 commit 的来源不表述为已完成逐版本复现。

| 模型 / 组件 | 上游与参考版本 |
|---|---|
| MMP、PANTHER、OT、MMP 接入版 SurvPath | [mahmoodlab/MMP](https://github.com/mahmoodlab/MMP/tree/1fd75e37592f7ab47b787f737733ba3c5a7e714c)，`1fd75e37592f7ab47b787f737733ba3c5a7e714c` |
| DIMAF | [Trustworthy-AI-UU-NKI/DIMAF](https://github.com/Trustworthy-AI-UU-NKI/DIMAF/tree/286dae63fcdc65de38224981cf345845d25c57be)，`286dae63fcdc65de38224981cf345845d25c57be` |
| SlotSPE | [zylvemvet/SlotSPE](https://github.com/zylvemvet/SlotSPE/tree/02051a2083add7b427727e0200e4263516903561)，`02051a2083add7b427727e0200e4263516903561` |
| SurvPath 方法及既有 MIL/RNA 基线 | [mahmoodlab/SurvPath](https://github.com/mahmoodlab/SurvPath)；沿用旧实验接入文件，未逐个对照独立 SurvPath 最新版本 |
| TITAN | [mahmoodlab/TITAN](https://github.com/mahmoodlab/TITAN)、[模型卡](https://huggingface.co/MahmoodLab/TITAN)；迁入旧实验本地快照 |
| STARPath | 旧 `Survival_Exp_Path2Space_v5_layer_ablation_2345` 完整 C 模型；合入 `modal_starpath.py` |

## MMP 与公共接口

核验时，旧实验的 `model_multimodal.py`、`model_OT.py`、`model_PANTHER.py`、`model_configs.py`、`model_h2t.py`、`model_protocount.py`、`tokenizer.py`、`PANTHER/`、`OT/` 与目标官方 MMP clone 对应文件字节一致。

整理后保留主干；`model_multimodal.py` 的 OT uniform weights 从硬编码 `.cuda()` 改为输入 device，数值定义不变。主要改动位于 factory、公共 `process_surv`、adapter、数据和训练入口。

旧 factory 忽略脚本传入的 `tau=1`、`ot_eps=1`，实际使用 `PANTHER_default` 中的 `tau=0.001`、`ot_eps=0.1`。新 factory 使显式参数生效，同时入口默认保留原来实际执行的 `0.001` 和 `0.1`，避免在接口修复时无意改变默认原型表示。

单患者 WSI 模型进入 ABMIL、TransMIL、MCAT、SurvPath、SlotSPE 前，按 mask 移除 padding；RNA recipe 不因此改换。统一 NLL 风险为 `-sum(cumprod(1-sigmoid(logits)))`，Cox 使用单个 log-risk。

NLL 数据协议另有必要修正：旧 `WSIOmicsSurvivalDataset` 构造时丢失传入的 `label_bins`，使 val/test 各自重新 qcut。当前所有模型统一使用 `UnifiedSurvivalDataset`，严格复用训练集拟合的 bins；旧 `wsi_survival.py` 已移除，这里的类名仅描述历史问题。主干数值对齐因此不等于旧实验 NLL 结果完全复现；新旧结果比较须同时考虑这项标签修正与新的冻结划分。历史 dataset 来源仍可在上述 MMP commit 中查阅。

所有模型的 Cox 训练使用 `EventAwareRiskSetBatchSampler`：尾部单患者批与前一批重组，保证每位患者每轮恰好出现一次。STARPath、ABMIL、TransMIL、MCAT、SurvPath、SlotSPE 按逻辑风险集做两次逐患者 forward，第二次恢复 RNG 后计算参数梯度；这些模型的 NLL 仍要求 batch=1，可使用 `accum_steps`。DIMAF NLL 默认直接使用 batch=64。这里是统一训练协议，不能以单次 backbone 对照代替完整训练协议对照。

## DIMAF

目标为 [`model_dimaf.py`](../src/mil_models/model_dimaf.py)。解除旧版单输出限制，按主 loss 创建 NLL 的离散时间头或 Cox 单输出头，保留融合计算。

同时修正原输入接口：官方 [embeddings.py](https://github.com/Trustworthy-AI-UU-NKI/DIMAF/blob/286dae63fcdc65de38224981cf345845d25c57be/src/embeddings/embeddings.py#L33) 只拼接原型 occupancy 与 mean；旧管线误用 MMP 的 occupancy + mean + covariance。本工程 DIMAF 输入现为 `[prob, mean]`，768 维 patch 对应每原型 769 维输入；MMP 仍保留 covariance。

官方 `survival/train.py` 已支持 NLL，纯 NLL 风险公式与上面的统一公式一致；Cox 官方用 `exp(logits)` 评价，本工程使用排序等价的 logits。官方复合 `nll_distcor` 的 CLI bins 和风险分支存在不一致，因此主 loss 与解耦辅助项明确分开。

当前协议：

- **NLL 和 Cox 都默认 batch=64**，遵从当前实验要求，也对应官方 main 的 batch 设置。
- 主 NLL 为统一的 **mean reduction**；官方纯 NLL 是 sum，且不包含 distance correlation。
- 两项 distance correlation 各乘 0.5，再用独立权重 `dimaf_disentanglement_weight=7`。本工程默认 NLL 是主 NLL + 解耦辅助项，不能称为官方纯 NLL 超参数的完全复刻。
- `--dimaf_disentanglement_weight 0` 关闭辅助项。显式 batch=1 时没有可估计的患者间关系，辅助项返回可微零，并记录 `distance_correlation_estimable=False`。

同权重、3 位患者、16 个 WSI 原型、50 个 RNA pathways 的构造输入核验中，官方与迁移前 DIMAF 的 logits、四种单/跨模态表征最大绝对差均为 **0**。这验证融合计算；不意味着旧输入管线或全部训练超参数与官方一致。

## SlotSPE

目标为 [`model_slotspe.py`](../src/mil_models/model_slotspe.py)。修复旧接入版 static K/V cross-attention 中多余的内部 LayerNorm；按官方方式从进入块的 x1/x2 直接计算静态 K/V。

迁移前构件对照中，self Transformer、dynamic K/V 的最大绝对差为 0，slot attention 约 `2.38e-7`；默认 static K/V 差达 `1.74068`。修复后的持久化测试使用官方生成的 40 个输出值验证。

删除 legacy 路径、人造 signature gene chunks、基因轴静默裁剪/补零、非 finite 均值填充。官方评估时缺 RNA 重建功能保留；完整模态实验仍校验冻结 cohort 的 RNA 覆盖。

Cox 模式保留官方辅助学习：**主头 1 维，两辅助 NLL decoder 使用独立 bins（默认 4）**；dataset 同时提供连续时间与离散标签，adapter 为辅助头使用独立 NLL。不会因切换 Cox 丢掉辅助 NLL 或重建项。

官方按验证集 C-index 选择 best，并另存 last，不要求 early stopping。本工程因此把两者独立控制；best 默认按验证 C-index，指标并列保存较晚轮。

## STARPath 与 TITAN

[`modal_starpath.py`](../src/mil_models/modal_starpath.py) 保留完整 C，删除 A/B 和兼容入口，合并形态统计、患者 adapter、slide 聚合。默认注入 `2,4`、可训练 `2,3,4,5`；可变注入层数采用“最后注入前一 block 结束后写回一次”，见 [架构说明](starpath_architecture.md)。

默认新旧 `state_dict` **671 个键**一致。受控 CPU、相同 TITAN 测试替身的对照中，初始化张量、关键前向张量及已计算参数梯度最大绝对差均为 **0**。

普通 TITAN 保留冻结预计算 embedding + 线性 survival head。`TITAN/` 与 `TITAN_STARPath/` 独立保存源码、配置、权重和 tokenizer；私有分支删除旧静态注入，保留 pre-block callback、post-block observer 及相同 grid/CLS/background 对齐。真实权重分别加载后的 4-patch 无 callback 输出差为 0，私有 callback 前向也已验证。

## 验证范围与归属

基线持久化测试包含 11 个非 STARPath、非 TITAN 模型 × NLL/Cox 的 22 个 forward/backward 子用例，以及 SlotSPE Cox 辅助梯度、padding 不变性和静态 K/V 官方输出。STARPath 另有主干、TITAN callback 与数据对齐测试。受控验证不代表八癌种完整训练已完成，也不保证跨环境随机训练轨迹逐位一致。

MMP 作者归属和原始 [`LICENSE.md`](../LICENSE.md) 保留，内容为 CC BY-NC-SA 4.0。第三方组件和资源按各自上游条款提供，根目录 license 不替代全部上游条款。TITAN 原始模型卡保留在 [`TITAN/README.md`](../src/mil_models/TITAN/README.md) 和 [`TITAN_STARPath/README.md`](../src/mil_models/TITAN_STARPath/README.md)，注明 CC BY-NC-ND 4.0 及资源条件。vision/text 源码原有 open_clip、iBOT、timm 来源说明也保留。DIMAF、SlotSPE、SurvPath 来源见上表。
