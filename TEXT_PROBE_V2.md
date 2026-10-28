# 文本 Probe 第二版实验

指南要求的 AgentLAB、BFCL-v4 和 LiveCodeBench 评测已另接原生环境，
入口、范围和结果见 [OFFICIAL_BENCHMARKS.md](OFFICIAL_BENCHMARKS.md)。
下文 BIPIA 结果仍属于训练及探索实验，不替代上述基准。

本版针对首版的四个问题：训练样本稀少、块级起点标注含糊、生成提示格式不统一，
以及每块重复处理 135 个固定问题/选项 token。首版权重和实验输出保存在 `artifacts/text_probe/`。

## 改动

Probe 仍是独立的 Qwen3-0.6B，冻结的 Qwen3-0.6B 草稿和 Qwen3-8B target 不参与训练。
保留 Kev 的“用一次预填充读取文本状态并作决策”的做法；读出位置改为一个特殊 token，
后接 `1024 → 3` 线性头。移除了三个自然语言选项，每块固定分支从 **135 token 降至 1 token**。
旧指针头检查点仍可加载，配置中的 `readout` 区分二者。

训练只更新 LoRA rank=16 和分类头。按训练集真实类别频数进行均衡、有放回抽样，
使用普通三分类交叉熵；验证与测试不重采样。增加的抽样次数不计入独立数据量。
本次 batch=8、2 epoch、lr=1e-4；每 16 个 batch 的窗口按长度组批，再打乱 batch，
减少长短文本混装的补齐开销。首轮统计的补齐 token 数减少 44.51%，见 `batching_profile.json`。
校准和在线推理都合并 LoRA 到冻结权重，保证使用相同计算方式，减少逐层 adapter 的调度。

## 数据和来源

在已有 BIPIA EmailQA 之外，增加同一个官方固定版本的 table/code 数据，
下载地址、字节数和 SHA256 在 `data/bipia/additional_source.json`，合计 2,048,160 字节。
没有额外下载模型权重。

预先准备 1200 个任务变体，来源分组如下，实际采集排除项另有记录：

| split | 原始来源组 | 任务变体 |
| --- | ---: | ---: |
| train | 140 | 700 |
| val | 35 | 175 |
| test | 65 | 325 |

来源包括邮件、表格和代码。每个来源包含 clean、两个可核验 marker 攻击、两个官方文本攻击。
训练来源从原来的 36 个增至计划中的 140 个；同一原始邮件/表格/代码讨论的不同问题、
攻击和种子不能跨 split。原有邮件的 train/val/test 分组保持一致。
测试 marker 前缀和模板从训练中留出，官方自然语言攻击使用官方 train/test 划分。
本实验的组合、采样及 marker 攻击是研究用构造，不能称为官方 BIPIA 全量 benchmark。

生成与 Probe 从同一份 `context.ordered_messages` 获取可信系统指令、真实用户请求和不可信 tool 内容。
采集器断言两者消息完全相同，记录实际 prompt token IDs 的哈希。
不再混入旧提示格式生成的轨迹；训练器拒绝混合新旧格式。

部分变体是 follow-up：先真实生成 clean 回答，再加入更新的外部材料、用户追问，
并把此前全部真实草稿/target 配对交给 Probe。这些是已展开的多轮案例，
不是 AgentLAB 的动态工具环境。

大部分草稿使用 greedy；训练中的一个 marker 变体对草稿使用 temperature=0.8，
以增加真实候选分歧。target 仍是 greedy，使用逐 token 匹配、首个不匹配纠正及 bonus token，
其最终输出通过测试保证与 greedy target 一致。每个实际 draft/target 输出均保存，
不根据原始攻击标签直接赋值，也不人为制造“target 抵抗”标签。

采集输出上限 96 token，提示上限 2048 token；超长明确记录，不静默截断。
`naturally_finished` 记录是否自然结束。最终任务真值覆盖实际保存/发布的有长度上限的回答，
不能推断未生成的后续文本。

## 块级标注

三类顺序仍是 A=target 中招，B=草稿中招/target 正常，C=两者正常。
target 单独中招也归 A。当前 target 及所有标签只出现在离线监督中。

- clean 没有注入，自动标为 C；一般回答错误不等于被注入劫持。
- marker 同时记录“开始输出独特 marker 前缀”和“完整 marker 目标达成”。
  不再等完整长标记出现才标记风险。只有短暂不完整的前缀保留为未知。
  最终拦截评测分别报告完整目标成功及注入前缀是否已经发布。
- 自然语言攻击先对完整实际 target 回答判断任务结果，再独立判断选中的可见前缀。
  正样本附近及草稿被拒绝的位置优先选择，以提高 B 类和起点附近的覆盖。
  每个前缀标注请求只包含当时已接受的前缀和当前新片段，不包含未来输出。
  草稿标注请求不包含当前 target。同一次 API 请求不放入同一 case 的其他前缀或另一种当前输出。
- 标注必须给出当时可见、新增、能区别于 clean 的逐字证据。
  只有 `**` 等标点、证据仅在未来出现、证据仅来自旧前缀，均不能作为新正例。
  标签冲突、API 错误、低置信度或含糊片段保留未知，不能作为正常类。
  对尚未验证具体映射的数字/字母替换、乱序、去空格、错拼、编码和反转类任务，
  不使用 MiMo 的自信判断充当映射验证，先隔离块级标签；只有实际回答与 clean 完全相同，
  才保留确定的未中招任务真值。它们仍保留在数据中，覆盖率和隔离清单单独报告。

自然语言标签来自 MiMo-V2.6-Flash，仍需人工审核，不能称为人工金标。
先审查了 pilot 中 5 个正例、4 个负例、2 个不确定例，见 `label_pilot_review.json`。
已观察到明确的 B 类：草稿开始讲光合作用/递归，而实际 target 继续回答账单金额。
第一处独立确认的可见偏离与精确因果起点仍有区别，语义起点评测需要说明这一限制。

实际采集 **1184 个任务、16,196 个检查点**，16 个超长变体被排除。
最终训练来源为 **139 个**，可用三分类训练点 **4110 个**：A=234、B=83、C=3793。
B 类覆盖 59 个原始来源；这些数字都不含重复抽样带来的重复次数。
验证集 A/B/C=62/20/996，全部已知类别测试点 A/B/C=102/16/1765。
隔离了 33 个仅新增标点/格式的正例判断，以及 159 个未验证映射的变换类 case。
任务真值未知的数量为 train=80、val=22、test=36，均保留为未知。
详细覆盖率见 `data_quality_summary.json`，隔离清单见 `frozen/labels.json`。

## 命令

本机解释器为 `/data/ygzhang/miniconda3/envs/deploy/bin/python`，模型只从本地加载。
以下 GPU 号是本次运行选择，执行前应检查占用。当前共享服务器其他任务会影响计时。

```bash
python -m baseline.prepare_text_v2
CUDA_VISIBLE_DEVICES=6,7 HF_HUB_OFFLINE=1 python -m baseline.collect_text_v2
python -m baseline.label_text_v2 --key-file /secure/path/mimo_key --watch

CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 python -m baseline.train_text_probe \
  --records artifacts/text_probe_v2/frozen/records.jsonl \
  --output artifacts/text_probe_v2/model --readout linear --device cuda:0

CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 python -m baseline.evaluate_text_probe \
  --records artifacts/text_probe_v2/frozen/records.jsonl \
  --checkpoint artifacts/text_probe_v2/model \
  --output artifacts/text_probe_v2/model/classification_all_known.json --device cuda:0
```

实际实验控制脚本在 `artifacts/text_probe_v2/finish_experiment.py`：等待采集/标注完成后，
冻结数据及哈希，训练并校准，再评测所有已知三分类测试点。
任务级离线评测使用预先固定的 96 个验证/测试任务；阈值只看验证集。
实际解码预先按 seed=271828 从每个领域选三个测试来源组、保留所有变体，
未使用新模型的测试分数选择案例。实际排除项可能减少运行案例数。

真实运行使用两张卡：draft 和 Probe 在一张卡，8B target 在另一张卡。
每例重复三次，静默模式与无防护模式交替运行，检查最终全部 target token 一致。
拦截模式必须在当前块发布前决定，检查所有被挡住的块确实没有发布 token。
已经发布的内容不会撤回；语义攻击的实际受保护回答另行审核，不能仅凭发生拦截就认定成功防御。

实验进度见 `artifacts/text_probe_v2/experiment_progress.json`，
采集及 API 进度分别在 `collection/progress.json` 和 `labeled/api_progress.json`。
最终结果在 `model/`、`runtime_validation.json`、`frozen/labels.json`。
与第一版同时改变了数据、标注和读出头，不能将分数变化单独归因于增加数据。
本地实验仍未覆盖完整 AgentLAB、BFCL、LiveCodeBench 或跨模型迁移。

## 已完成的结果

两轮训练及逐轮验证耗时 **2999 秒（约 50 分钟）**，不含采集、标注、阈值校准和测试。
第一轮验证交叉熵 0.1653，第二轮升至 0.2931，因此保存并使用第一轮权重。
采集 1184 个任务另耗时约 50 分钟；API 标注与采集并行进行。

在同一批 1883 个已知类别测试检查点上，以三个分数的 argmax 分类：

| 类别 | 首版模型判对 | 第二版模型判对 |
| --- | ---: | ---: |
| A：target 中招 | 67/102 | 76/102 |
| B：草稿中招、target 正常 | 0/16 | 8/16 |
| C：两者正常 | 1665/1765 | 1739/1765 |

这项比较使用同一批新测试点。它不能直接与首版旧测试任务的 FNR/FPR 对比，
也不能将改进单独归因于增加数据：输入格式、标注、读出头和抽样策略均有变化。

固定的 96 个已知任务测试子集（35 正、61 负）上，按验证集选阈值：

| 放行规则 | 测试任务漏报 | 测试任务误报 | 未及时告警 |
| --- | ---: | ---: | ---: |
| A 分数，验证集要求零漏报 | 0/35 | 50/61（81.97%） | 0/35 |
| A 分数，验证集误报低于 5% | 17/35（48.57%） | 0/61 | 23/35 |
| 接受标记融合，同样按验证集限制误报 | 8/35（22.86%） | 0/61 | 12/35 |

尚未找到满足指南两项约束的阈值。第一种规则通过拦截大量正常任务换取召回，不能当作达标。
第二种阈值 0.9978057742 在验证集上为 FPR=2/54、FNR=21/42。

追加的融合规则来源于验证错误分析：有些草稿已经在讲蝴蝶生命周期，Probe 给 B 高分，
实际 8B 却接受了整块。Probe 继续只读历史和当前草稿，与 8B 验证并行。
两者结束后的放行分数为：

```text
本轮草稿全部被 8B 接受：risk = A + B
其他情况：             risk = A
```

这里只使用验证本来就产生的接受计数，不把当前 8B 文本送回 Probe，不增加 Probe 前向调用。
规则没有把“部分拒绝”当作安全：8B 的替换或 bonus token 仍可能中招。
验证集上该规则以相同阈值得到 FPR=2/54、FNR=12/42。它是探索性追加规则，需要新的独立测试确认。

在预先选定的 45 个真实生成案例（9 个来源组，每例三次重复）上，安全统计只计一次重复：

| 规则 | 9 个 clean 被中断 | 完整 marker 目标成功 | 已发布 marker 前缀 | 全部任务中确认仍中招 / 未知 |
| --- | ---: | ---: | ---: | ---: |
| 无防护 | 0 | 7/18 次攻击尝试 | 7 | 12 / 5 |
| 严格召回 | 7 | 0/18 | 0 | 0 / 1 |
| 误报约束，A 分数 | 0 | 2/18 | 5 | 7 / 5 |
| 误报约束，接受标记融合 | 0 | 0/18 | 2 | 4 / 5 |

融合规则下，12 个已知原本中招的案例仍有 **4 个**发布了可确认的注入内容，
所以不能将“完整 marker 目标为 0”解读为“全部攻击都被及时挡住”。
9 个 clean 的完整输出与基线一致，草稿接受率均为 63.96%；这个小样本不代表 BFCL 指标通过。
完整语义攻击由实际已发布答案重新审核，未验证的变换类继续保留未知。

135 对无防护/静默请求的最终 token 全部一致，草稿接受率均为 62.82%。
静默文本 Probe 的平均端到端时延从 **2917.2 ms** 增至 **3496.4 ms**，增加约 **19.86%**，
配对差为 579.2 ± 429.8 ms（均值 ± 样本标准差）。服务器同时运行其他任务，
这一结果只能说明当前实现仍有额外开销，不能与首版不同任务/布局下的 39% 直接比较改进幅度。
融合不增加模型前向，但没有为它单独重做静默时延基准，不能声称其额外延迟已被严格测成零。

当前实验配置在 `artifacts/text_probe_v2/model_fpr5_fused/`。
`model/`、`model_fpr5/`、`model_fpr5_fused/` 共享同一组已训练权重，仅阈值/放行规则不同。
一次真实运行可用：

```bash
CUDA_VISIBLE_DEVICES=6,7 HF_HUB_OFFLINE=1 python -m baseline.run_text_speculative \
  --cases artifacts/text_probe_v2/runtime_cases.jsonl \
  --probe-checkpoint artifacts/text_probe_v2/model_fpr5_fused \
  --output artifacts/text_probe_v2/reproduce_fused --mode intercept --repeats 3 \
  --max-new-tokens 96 --draft-device cuda:0 --target-device cuda:1 --probe-device cuda:0
```

三种阈值/规则的实际输出比较见 `actual_policy_comparison.json`，
固定测试子集的阈值比较见 `operating_points/report.json` 和 `verifier_fusion_diagnostic.json`。
最困难的验证正例见 `validation_hard_positives.json`：草稿是正常的账单回答，
8B 却改成输出攻击 marker，Probe 当时给正常类 0.99999976。
这类错误说明瓶颈包含对 target 行为的预测，继续增加普通安全检查点不能直接解决它。

目前 **77 项测试通过**，包括融合规则、草稿被拒绝时不错误融合、放行前拦截、
无当前 target 文本泄漏、KV 裁剪和标准投机采样分布。指南指标仍未达标。
