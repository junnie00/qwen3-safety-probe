# ModernBERT 冻结表示＋双视图分类头 PoC

最新状态（2026-11-06）：S3 实验及核验已完成，用户选择以 S3 为当前主要研究方案，S2 保留为低误报对照。先读[面向主分支读者的阶段报告](modernbert-experiment-summary.md)，再按需查看本文的完整演进记录；第1—7节为初始三分类协议，第8—12节记录后续二分类、预测证据及发布前审查实验，不应把初始约束当成当前方案。

最近实验已有119项回归测试通过，S3的606个epoch检查点重放通过。分支为 `cx`；早期工作已本地提交 `272f7a1`，后续实验及本次报告尚未追加提交，未推送、未合并 `main`。

## 1. 目标与已确认范围

验证：在相同误报预算下，攻击证据视图能否补充通用视图的 A/B/C 分类能力。

用户已确认：

- 所有检测组件先以 ModernBERT-base 为骨干；本轮共享一份冻结骨干，只训练线性分类头，不更新骨干、不训练 LoRA。
- 复用现有冻结 Qwen 草稿/主模型轨迹，不重新采集，不替换生成侧模型。
- 第一阶段只做离线检测，不宣称已经验证在线并行、实际拦截或旧 Qwen Probe 的成绩复现。
- 通用视图：用户任务＋当前草稿＋最开始三条/最近三条聊天消息，按原位置去重，保留来源角色。
- 不使用 `history_pairs`。专家视图只读用户任务＋当前草稿＋已知 `attack_str`，本轮不额外增加引用语境处理。
- 先做一个覆盖已有攻击的“攻击证据 Probe”，暂不拆多个类型专家。
- 使用已知攻击原文的 oracle 路由；不训练攻击分类/定位器，不调用 Jev API。
- 已知 `attack_str` 非空时才启用证据分支。含攻击但行为为 C 的样本必须保留；不得按 A/B/C 真值决定是否启用。
- 保留额外通用视图分类头的对照实验。
- 本实验进程树合计 CPU 与内存不得超过有效总资源的30%。后台其他任务不属于此限额对象，也不得被修改、清理或停止。

## 2. 数据与输入

### 数据源

- `artifacts/text_probe_v2/frozen/records.jsonl`：块级输入、监督与来源。
- `artifacts/text_probe_v2/frozen/resolved_cases.jsonl`：按 case_id 关联原始 question、attack_str、消息和来源组。
- 保留已有 train/val/test，不重新随机拆块，不覆盖冻结记录。

现有数据：1184 case，16196条块记录；717 case有3条聊天消息，467 case有6条。首三条＋尾三条去重对全部case均保留完整消息，因此当前实际的信息删减来自移除模型行为历史，而不是聊天消息数量限制。

已知类别且不属于 after_violation 的块级样本：

| split | A | B | C | 合计 |
| --- | ---: | ---: | ---: | ---: |
| train | 234 | 83 | 3793 | 4110 |
| val | 62 | 20 | 996 | 1078 |
| test | 102 | 16 | 1765 | 1883 |

这些是长度处理之前的标签过滤数，实施时重新验证。训练中的 oracle-攻击子集为2071点：A=234、B=83、C=1754。

### 两种确定性视图

任务字段取关联case的 `question`，不额外调用模型提取可信任务。序列化规则、字段顺序与版本固定，角色来自已有元数据。

```text
通用视图 G：
[TASK] 用户任务
[DRAFT] 当前草稿
[MESSAGES] 首三条＋尾三条消息，按位置去重并保持顺序与role

攻击证据视图 E：
[TASK] 用户任务
[DRAFT] 当前草稿
[ATTACK] 数据集已有的攻击原文
```

`[TASK]` 等是普通文本分隔符，不通过扩词表增加未经预训练的新token。

严禁输入：当前target、参考答案、行为标签、task_label、case/group编号、split、标注器说明。已知attack_str进入E是显式声明的oracle特权，不是可部署输入。

### 长度与有损输入审计

- 默认总预算4096 token，两种视图使用同一tokenizer。
- 先只做token长度统计，不把“最多六条消息”当作“肯定不超长”。
- TASK和DRAFT必须完整；E中的攻击证据必须完整。
- G超过预算时，只裁剪MESSAGES部分的中间token，按剩余预算保留前后两端；保存原消息角色信息、原始长度、保留长度及裁剪位置。这不使用当前target或攻击真值决定截取位置。
- 若TASK＋DRAFT或E的必需字段自身超过预算，暂停该批并报告，不默默裁掉关键字段、不仅从某一个实验组排除不利样本。
- 报告裁剪样本比例及按领域/类别的覆盖。任何因数据错误而排除的样本在全部对照中一致处理。
- 对移除history后的输入做哈希，统计“相同输入却有不同A/B/C标签”的碰撞。保留真实标签并报告信息不足，不把碰撞自动修成同类。

## 3. 冻结编码与分类头

骨干：`answerdotai/ModernBERT-base`，实施时固定下载revision和tokenizer版本。仅此模型，不下载Qwen-8B，不强制安装额外FlashAttention或源码编译包。

本轮默认实现参数（非历史实验配置）：

- 骨干 `eval()`、关闭参数梯度，使用 `inference_mode`。正式运行改用FP32：开跑前BF16短输入跨batch相对L2漂移约1%–2%，FP32诊断约百万分之一量级；这是数值正确性修正，不根据分类测试成绩选择精度。
- 固定采用最后层hidden states的非padding均值池化；不根据测试结果更换池化。
- pooled向量以FP32缓存，维度读取模型config。
- 三个独立线性头，均为 `Linear(hidden_size, 3)`：G、G_aux、E。
- 只保存池化后的向量，不缓存每个token的hidden states或attention。
- 特征提取按长度桶试batch=1/2/4/8；每桶使用最长输入，预热一次、计时三次，选择距最高吞吐5%以内的最小batch。OOM时减半，不提高资源限额；G和E顺序编码，不要求实际同时占用GPU。
- 缓存键包含模型/tokenizer revision、序列化版本、输入token哈希、池化和裁剪策略。

由于骨干冻结，G与G_aux可复用同一份通用特征；E需要额外编码一次。三组对照不需要三份ModernBERT骨干。

这是“冻结表示中的线性可读性”实验。效果差不能直接推导为骨干微调或类型路由一定无效。

## 4. 训练与对照

### 训练样本

- G：全部有效train记录。
- G_aux与E：完全相同的oracle-攻击train子集，包括攻击被抵抗的C类；仅特征视图不同。
- G_aux使用通用特征，E使用证据特征。
- 所有头使用相同优化流程。训练类别权重从各自train子集计算，val/test保持自然分布，不重采样。

### 三组对照

| 配置 | 所有检查点 | attack_str非空时增加 | 目的 |
| --- | --- | --- | --- |
| M0 | G | 无 | 基线 |
| M1 | G | G_aux | 控制附加分类头/攻击子集训练的收益 |
| M2 | G | E | 测量证据视图的增量价值 |

M1与M2使用完全一致的oracle启用掩码、辅助头训练子集和聚合方法。比较M2对M1的增益，比仅比较M2对M0更能说明视图价值。

局限：冻结骨干下，两通用头看到相同表示，M1不是两个独立微调模型的强集成；M1还可以复用编码结果，因此与M2不是等推理成本。分别报告头参数量、独特编码次数和缓存前成本，不声称完全等算力对照。

### 默认分类头训练参数

- 分类头在CPU训练，使用缓存向量；不为小线性头额外占用GPU。
- AdamW，lr=1e-3、weight_decay=1e-4，batch=128。
- 最多100 epoch，patience=10；按各自验证范围的未加权交叉熵选择检查点。
- 训练用按train频数计算的逆频率类权重，记录准确权重；不把重复采样算新增独立数据。
- seeds=42、43、44，串行运行。M1与M2的辅助头使用配对种子。
- 所有参数在看test结果前固定。本轮不展开大规模超参搜索。

## 5. 聚合、校准与指标

本轮离线任务的主工作点明确为：**验证集上A类块级误报率不超过5%，比较测试集A类召回**。A类误报的负例为真实B/C点。这是块级指标，不等同任务级误报率或真实攻击成功率。

对当前启用的头：

```text
score_A = max(各头P(A))
score_B = max(各头P(B))
若 score_A >= tau_A，输出A；
否则若 score_B >= tau_B，输出B；
否则输出C候选。
```

- 每个配置独立用validation选择tau_A，在满足5%约束的候选点中最大化A召回；平手优先更低误报/更高阈值。
- 固定tau_A后，在validation选择tau_B以最大化最终B类F1。
- 最大值分数不是一组归一化的三分类概率，不把它们直接当最终校准概率。
- 未启用分支不投票，不将其伪装成“预测C”。C输出也不是安全证明。
- validation同时用于早停和阈值选择，需披露小样本选择偏差；test不参与选择。

主要交付指标：

1. 测试A召回、实际A误报率，及M2相对M0/M1的变化。
2. B召回、C被误判A/B的比例、macro-F1、三分类混淆矩阵。
3. 证据分支新增检出的A点与新增误报，按来源组/领域分层。
4. 三个种子的结果；不确定性按source group重采样，不把相邻块当独立样本。
5. 单独提特征耗时、分类头训练耗时、编码调用次数、CPU和内存峰值。

本轮仅在有效标签检查点上报告离线分类。不会把不完整标签轨迹的聚合结果宣传为完整任务防御率，不把缓存后的头推理时间宣传为在线端到端延迟。

在总体A误报约束下，如果M2相对M1没有稳定收益，结论是当前表示/输入/线性头没有体现增量收益，而不是整个专家路线无效。若有增益，再讨论实际攻击识别器、类型划分、骨干微调和在线部署。

## 6. CPU与内存硬限制

### 本机只读核验

当前有效逻辑CPU为28，MemTotal约23.466 GiB；cgroup v2。已实现 `scripts/run_probe_limited.py`，每次在负载子进程内部读取实际cgroup设置，确认 `cpu.max=200000 100000`、`memory.max=6442450944`、`memory.high=5368709120`、`memory.swap.max=0`、affinity为两个允许核后才执行。互斥锁防止本PoC同时运行多个scope叠加资源。正式任务限额证明与连续资源采样保存在运行目录的 `run-resources/`。

本轮采用比30%更保守的默认限额：

| 项目 | 计划设置 | 当前机器占比/作用 |
| --- | --- | --- |
| CPU硬配额 | CPUQuota=200% | 合计2个逻辑核，约7.14% |
| CPU亲和性 | 继承允许CPU中的两个核 | 防止大量线程短时扩散到全机；不是独占核 |
| 内存硬限 | MemoryMax=6G | 6 GiB，约25.57% |
| 内存软阈值 | MemoryHigh=5G | 提前触发回收/压力信号 |
| 交换空间 | MemorySwapMax=0 | 不为本实验新增swap占用 |
| 线程 | OMP/MKL/OPENBLAS等为2，torch线程为2 | 与硬限一致 |
| 数据加载 | num_workers=0，pin_memory=false | 避免worker复制和额外锁页内存 |
| tokenizer | TOKENIZERS_PARALLELISM=false | 不创建并行分词线程池 |
| 调度 | nice=15，ionice idle | 让位于其他任务，不代替硬限 |

启动时重新计算有效CPU、上层cgroup限制和内存总量。若环境变小，自动把2核/6GiB下调到30%以内；不可因沿用旧数值而越限。可用内存不足时暂停启动/下一批，不把“有硬限”当作系统一定有余量。

### 执行方法与验证

- 正式任务及所有数据处理、分词、特征提取、头训练子进程必须位于**同一个用户级systemd scope/cgroup**，限额按整个进程树合计，不为每个头各给一份30%。
- 用systemd-run的CPUQuota/MemoryMax/MemoryHigh/MemorySwapMax设置硬限，并用taskset绑定两个允许的CPU。
- 第一次只启动轻量自检子进程，从自身 `/proc/self/cgroup` 定位真实路径，再读取 `cpu.max`、`memory.max`、`memory.high`、`memory.swap.max` 和affinity，确认限制已落地后才进入下载/提特征阶段。
- 自检只检查配置和小规模功能，不做多核忙循环、不分配数GiB内存来故意触发OOM。
- 若用户级限制不能落地，停止并报告，不退化成只用nice/线程变量而宣称满足硬上限。
- 分片写缓存、保存进度、记录退出状态。超过软阈值时优先减小当前batch/释放临时对象，触及硬限中止本实验，不自动提限。
- 不清理历史失败的systemd单元，不停止其他进程，不修改全局配置或驱动。

CPUQuota按每核100%计，200%不是整机200%；此处是28核中的2核。内存按cgroup计量，包含归属该组的匿名页和文件缓存等，不只看Python数组大小。

## 7. 实施顺序与产物

1. 实现并验证资源限额包装器。
2. 流式核对输入记录、case关联、split/group隔离及有损输入碰撞；固定数据与配置哈希。
3. 获取固定版本的ModernBERT-base/tokenizer，先核验token长度和一个小批次，无额外模型下载。
4. 顺序提取G和E池化特征，分片缓存，支持续跑。
5. CPU串行训练G、G_aux、E及三个种子；保存训练曲线和验证选择。
6. validation选择阈值；一次性执行预先固定的test对照并汇总，不根据test表现修改模型。

正式产物目录：`artifacts/modernbert_oracle_poc/run-20261104-v2/`，包括配置、数据清单、视图/长度审计、特征缓存、分类头、阈值、预测、资源日志及结果报告。执行不需要Jev API、不需要8B重新采集。

### 执行记录与复核

- 模型及tokenizer revision固定为 `8949b909ec900327062f0ebf497f51aef5e6f0c8`，下载完成，运行使用离线缓存。
- `baseline/modernbert_data.py` 实现输入白名单、分组隔离、首尾消息去重、长度审计与输入冲突报告；`baseline/modernbert_poc.py` 实现长度分桶测速、冻结编码及可恢复特征缓存；`baseline/modernbert_heads.py` 实现训练、校准、对照与来源组bootstrap。
- 数据审计核实train/val/test为4110/1078/1883条；E启用点合计3670条。无输入需要截断，无来源组跨split。G/E分别有5组输入对应多个标签，各影响10条记录，保留原标签。
- BF16起步检查未通过跨batch数值一致性，停止后独立对照FP32；正式提取用FP32，不放宽一致性容差。
- FP32首轮产物核验发现通用Fast tokenizer的 `build_inputs_with_special_tokens` 不执行backend模板。已标记 `run-20261104-fp32/INVALID.json`，不采用旧特征和成绩。v2显式核验单序列模板并加入原有CLS/SEP；不新增token、不改词表。首轮调试成绩已经读取，修正依据是tokenizer协议，不按成绩挑选版本。
- 修复后87项回归测试通过（不含需外部基准依赖的 `test_official_benchmarks.py`）。
- 正式成绩已由 `scripts/verify_modernbert_run.py` 复核：九个检查点重放的预测/混淆矩阵吻合，数据与代码哈希、缓存完整性、资源限额与退出状态均通过。详见正式运行的 `verification.json`。
- 正式pipeline耗时151.53秒；进程树内存峰值约2.19GiB，平均0.873逻辑核，memory.high/max/OOM事件均为0。长度≤512 token用batch8，更长输入用batch4。
- 三seed测试均值：M0/M1/M2的A召回分别为4.58%/7.19%/16.67%，A误报率1.59%/2.32%/1.93%；M2的B召回均值只有6.25%。证据视图有A增益信号，但绝对召回低、B退化、一个seed的A增益bootstrap区间包含零，不可据此声称已具备可部署防御能力。

重新执行时使用新的run目录；对同一代码/配置的中断续跑可复用原目录：

```bash
python3 scripts/run_probe_limited.py \
  --log-dir artifacts/modernbert_oracle_poc/<run_id>/run-resources \
  -- env HF_HUB_OFFLINE=1 .venv/bin/python -m baseline.modernbert_poc \
  --run-dir artifacts/modernbert_oracle_poc/<run_id> --stage all
```

复核（使用独立的资源日志目录，避免覆盖正式运行资源记录）：

```bash
python3 scripts/run_probe_limited.py \
  --log-dir artifacts/modernbert_oracle_poc/<run_id>/verification-resources \
  -- .venv/bin/python scripts/verify_modernbert_run.py \
  artifacts/modernbert_oracle_poc/<run_id>
```

## 8. A／非A二分类探索性对照（已执行）

用户确认后，第一步仅复用正式v2的特征和完全相同的样本划分，不扩充标签不完整样本。保留M0/M1/M2和seeds42/43/44；输出改为单logit，A=1、非A=0，使用BCEWithLogitsLoss，pos_weight按各头训练负/正比计算。其余优化器、学习率、batch128、100epoch上限、patience10保持；按未加权验证BCE早停，验证集A误报≤5%时最大化A召回。

实现：`baseline/modernbert_binary.py`；独立检查点重放：`scripts/verify_modernbert_binary.py`。二分类训练/评估全程CPU-only，原三分类代码与特征不变。94项回归测试和九个二分类检查点核验通过。

结果：`artifacts/modernbert_oracle_poc/run-20261105-binary/report.md`。M2的A召回均值从16.67%升至20.26%，但误报率从1.93%升至3.28%，精确率下降；一个seed召回下降、两个上升，不能断言稳定优于三分类。测试集已看过，本轮是探索性对照，不作为新的盲测。实际峰值内存约1.02GiB、平均0.24逻辑核，无GPU计算。

重跑请使用新的输出目录：

```bash
python3 scripts/run_probe_limited.py \
  --log-dir artifacts/modernbert_oracle_poc/<binary_run_id>/run-resources \
  -- env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 .venv/bin/python -m baseline.modernbert_binary \
  --source-run artifacts/modernbert_oracle_poc/run-20261104-v2 \
  --run-dir artifacts/modernbert_oracle_poc/<binary_run_id>
```

## 9. ModernBERT-large与完整收敛轨迹（已完成并核验）

用户要求换large并保存训练轨迹。对照方案在运行前固定于 `artifacts/modernbert_oracle_poc/large-vs-base-20261105/protocol.json`：

- 继续A/非A二分类、冻结骨干、相同v2输入、数据与划分。large固定revision为 `45bb4654a4d5aaff24dd11d4781fa46d39bf8c13`；仍用FP32，原base特征直接复用。
- 同时重跑base/large的小头，每模型3个头×3个seed，全部跑满100epoch；不实际早停，但记录patience10的模拟停止位置。所有其他优化参数与前轮二分类一致。
- 每轮保存train/validation未加权及加权BCE、梯度范数、学习率、验证集5%FPR工作点、验证预测。保存epoch000到epoch100的权重、优化器与随机数状态，并单独保存best和final。
- 最佳checkpoint只取全100轮中未加权validation BCE最低者。test只在选择完checkpoint后评估，不逐轮看test调模型。结论基于本轮同训练规则的base对large，不把训练长度收益归因于模型变大。
- large部分batch未通过原有FP32容差（atol/rtol=1e-4）；`baseline/modernbert_large_features.py`跳过不一致候选并回退，而非放宽标准，还检查长度桶内padding一致性。
- 代码：`baseline/modernbert_trajectory.py`保存完整轨迹；`baseline/modernbert_capacity.py`做配对对照；`scripts/verify_modernbert_capacity.py`重放全部epoch验证预测及最终测试决策。
- 98项回归测试通过。正式large特征目录为 `artifacts/modernbert_oracle_poc/large-features-20261105-v2/`；对照产物目录为 `artifacts/modernbert_oracle_poc/large-vs-base-20261105/`。数值检查中止的旧large目录不作为结果。
- CPU与内存硬限保持2核/6GiB（5GiB软限）、swap=0，各阶段串行，不叠加领取资源。

已完成并重放核验全部1818个epoch检查点及18组测试配置。正式报告：`artifacts/modernbert_oracle_poc/large-vs-base-20261105/report.md`。

- 同样全100轮、按最低验证BCE选点：base/large M2 A召回均值为9.48%/48.04%，A误报0.99%/1.57%，A精确率34.70%/63.89%；三个seed的large M2召回均更高。自然语言攻击子集召回35.14%、误报6.76%，不能把总体表现当作所有攻击来源的表现。
- large E三seed最佳验证轮次为96/83/99，尚不能判断100轮饱和；验证曲线波动大，末轮并非最优。旧patience10会在37/25/53轮停止，错过后续更低验证损失。
- 更低验证BCE不保证低误报工作点召回更高：base本轮比旧早停结果的召回低。没有事后按test更换选点标准，后续需单独研究优化稳定性与选点指标。
- 正式特征提取＋全轨迹头训练评估进程合计约6分33秒；峰值内存约2.91GiB，平均CPU各阶段不足1逻辑核，限额与资源事件均已核验。未提交、未推送、未修改主分支。
- 每个头的 `trajectory.csv/jsonl/svg`、`validation_scores.npy`、`checkpoints/epoch-000..100.pt`、`best.pt`、`final.pt`全部保留。训练loss下降不等于泛化继续提升；下一步可另设固定学习率衰减/延长训练窗口的对照，本轮未自动追加。

## 10. 预测攻击证据、漏检模式与独立意图告警（已完成）

上述已验证代码与仓库配置已按用户要求提交至本地cx，commit `272f7a1`；未推送，main未动。以下新增工作在该commit之后完成，尚未再次提交。

### 检测头定义

用户确认的目标是：输入原用户任务＋工具材料，判断是否包含试图越权改变原任务的指令。攻击构造span为正例，常规工具材料为负例；攻击最终被抵抗的case仍是攻击存在正例。当前以数据构造标签为监督代理，不声称完成全面人工语义复核。

`baseline/evidence_detector.py`使用冻结ModernBERT-base表示＋线性头。工具材料只按空行及192token窗口/96stride产生候选，gold不参与候选边界或推理。训练100轮、三个seed，按验证case BCE选点；在验证clean case误报≤5%约束下校准存在性阈值。只输出预测最高分原文段落，不进行攻击类型路由。

`baseline/evidence_pipeline.py`将预测片段接入既有冻结large E头，G保持不变。不按gold启用分支，不回填缺失证据；clean误触发分支和错误片段均计入评估。精确token哈希相同的输入可复用旧确定性特征缓存，不改变预测文本。

报告：`artifacts/modernbert_oracle_poc/predicted-evidence-20261105/report.md`。检测器攻击存在召回87.60%、clean case误报6.15%；接入预测证据的A召回47.39%、A误报3.14%，对照oracle为48.04%/1.57%。105项回归测试、300个检测器epoch检查点与全部下游预测核验通过；每轮轨迹与权重保留。检测器最佳轮次99/100/93，尚不能确认已饱和。

### 哪些A容易漏检

`scripts/analyze_attack_recall.py`分析既有large oracle M2的102个真实A（102个case、56个来源组）。明确LLM处理指令/marker召回64.8%，夹带额外内容64.7%，改做无关任务18.2%，业务流程伪装插入凭证13.9%，改变语言/emoji的9个A全部漏检。代码领域17个A平均召回5.9%。部分漏检中草稿仍正常、主模型才突然执行注入，属于观察信息不足，不能全归因于不认识攻击意图。

完整分组与示例：`artifacts/modernbert_oracle_poc/attack-recall-analysis-20261105/report.md`。机制归并是事后分析，不作为新增训练标签；小组样本量与领域/模板混杂限制因果解释。所有945个攻击case都在tool材料末尾追加攻击，当前定位任务偏容易，不能外推任意位置或正常引用攻击的困难负例。

### 是否把意图检测单独OR到A分支

`scripts/evaluate_intent_parallel.py`进行了额外CPU诊断。直接OR使A召回47.39%→95.75%，但相对A标签误报3.14%→50.72%；保持已有行为工作点、再以验证整体A误报≤5%校准第三分支时，三个seed均选择关闭额外OR，无增量召回。

因此应保留`attack_present`和`behavior_risk`两个独立信号。意图阳性可以触发隔离/复核，但不自动等同于主模型中招。若业务目标是所有真实攻击都拦截，应按攻击存在标签而非A标签评估策略；上述50.72%不是正常工具材料的检测误报率。

## 11. 主模型文本输入消融（已完成并核验）

用户授权同时尝试三组：S0保持当前结构；S1两个行为头都增加最近256token的历史主模型输出；S2在S1上再加入当前待发布主模型输出块。

**时间边界变更：**第2节对当前target的禁用继续适用于S0/S1。S2是用户明确授权的“生成后、发布前审查”实验，允许输入当前`supervision.target_output`，不允许输入标签、判定说明、完整未来输出或参考答案。不能把S2增益解释为提前预测提升。线上仍需实现发布/工具执行前缓冲，本轮不修改在线解码与外部副作用控制。

协议预先固定于 `artifacts/modernbert_oracle_poc/target-text-ablation-20261106/protocol.json`：

- 主结果使用冻结意图检测器预测的证据及启用mask；两层模型、source group划分、7071条有效块不变。
- 历史仅取`history_pairs`中按顺序拼接的target文本，再取最近256个large tokenizer token，不读取历史draft或接受统计。历史快照的16196条记录已经核实等于初始历史加严格此前各块target。
- 视图实现 `baseline/target_text_features.py`。S1在序列化函数级不访问supervision；S2只额外读取target_output。保留v2 CLS/SEP和mean pooling，用TARGET_HISTORY/TARGET_CURRENT普通文本分隔符，总预算4096。
- S0复用已有large全100轮的G/E头及预测证据结果。S1/S2各训练G/E×三个seed，共12个新线性头；骨干冻结FP32，训练参数与S0完全一致，全部100epoch并保留每轮轨迹及checkpoint。
- 为不引入第二个训练变量，E仍按原方案使用oracle攻击子集训练与选点，实际主评估时统一使用预测证据。另报告oracle诊断，不当作可部署结果。
- 检查点只按验证BCE选择；阈值仅在val整体A误报≤5%时校准，test不选epoch。S0/S1/S2使用配对来源组bootstrap。
- 训练/评估实现 `baseline/target_text_ablation.py`，复核 `scripts/verify_target_text_ablation.py`；输入时间隔离、gold-free预测证据、映射与bootstrap测试已通过，含原回归共113项通过。
- 产物根目录 `artifacts/modernbert_oracle_poc/target-text-ablation-20261106/`，新特征位于features/，头和完整轨迹位于S1/、S2/。资源继续按整个进程树2核/6GiB、swap0限制。

结果已核验：S0/S1/S2的A召回均值分别47.39%/36.27%/52.94%，实际A误报3.14%/2.68%/1.80%，精确率46.13%/43.55%/63.27%。S1三个seed召回均下降；S2相对S0增益主要来自一个seed，另外两个seed的召回差bootstrap区间包含零，不能夸大稳健性。

按攻击方式，S2的marker处理指令召回64.8%→76.7%，业务理由插入凭证13.9%→33.3%，语言/emoji0%→14.8%；但答案夹带类64.7%→39.2%，任务替换仍12.1%。这不是所有类型同时改善。S2必须解释为待发布文本审查，不能当作提前预测能力提升。

S1证据头最佳epoch96/83/99，S2为86/83/99，仍不能确认100轮已充分收敛。已保存并重放1212个epoch检查点，重新构建所有输入验证token哈希/映射，S0与意图检测输出不变。12份有效曲线保存为 `trajectory.checked.svg`：原SVG标题未转义小于号，已由 `scripts/render_checked_trajectories.py` 仅重绘展示层，不改变权重/特征/数值轨迹；请使用checked曲线。

提特征与训练评估进程合计约14分钟，内存峰值分别约3.38GiB和0.56GiB，平均CPU0.73核和1.34核，未触发内存soft/hard/OOM事件。完整结论及限制见该运行的report.md与verification.json；未追加Git提交。

## 12. S3：仅当前待发布输出，不增加模型历史（已完成并核验）

用户要求继续隔离历史字段的影响。S3在原G/E输入后仅追加TARGET_CURRENT与当前target_output，不访问history_pairs、不增加TARGET_HISTORY。原先G聊天上下文里的assistant消息保持不动，避免同时改变原有聊天输入。S2/S3都属于发布前审查，不能解释为提前预测。

协议预先固定于 `artifacts/modernbert_oracle_poc/current-target-only-20261106/protocol.json`。冻结相同large骨干、FP32、mean pooling、意图检测与证据/启用mask、原样本与划分。训练规则保持100epoch、3seed、AdamW与原类别权重、最低验证BCE选点；E仍按原oracle子集训练、预测证据评估。S0/S2原权重、结果与阈值直接复用，主比较为S3对S2。

实现 `baseline/current_target_only.py`；重放核验 `scripts/verify_current_target_only.py`。119项回归测试通过，包含禁止读取history_pairs、修改历史不影响输入、当前输出白名单、预测证据不回填以及有效SVG转义。产物目录 `artifacts/modernbert_oracle_poc/current-target-only-20261106/`，features为新特征，S3为六个头的完整轨迹/检查点；曲线使用trajectory.checked.svg。资源保持整个进程树2核/6GiB、swap0，各阶段串行。

已完成606个epoch检查点及9组新增测试配置重放核验，S0/S2原结果保持不变。S3预测证据主结果：A召回61.11%、A误报3.35%、A精确率51.26%；S2为52.94%/1.80%/63.27%。去掉额外历史提高召回，但不是全面占优，误报也上升。S3三seed召回59.80%/61.76%/61.76%，相对S2的第一个seed召回增益区间跨零。

攻击分组：S3的任务替换召回39.4%、业务凭证52.8%、答案夹带60.8%，高于S2；语言/emoji仍14.8%，代码领域17个A全部漏检。不能用总体增益掩盖领域退化。证据头最佳epoch96/89/99，仍不能确认100轮已饱和。

正式特征提取与训练评估约6分04秒，内存峰值约3.56GiB，平均CPU各阶段不足1核，未触发内存soft/hard/OOM。详见该运行report.md和verification.json；未追加Git提交。
