# 独立文本三分类 Probe

本页记录首版实验与已保存的结果。当前训练策略和默认数据路径已经更新，
新实验请使用 [TEXT_PROBE_V2.md](TEXT_PROBE_V2.md)。

这是已落地的研究 baseline。草稿模型是冻结的 **Qwen3-0.6B**，target 是冻结的
**Qwen3-8B**，Probe 是另一个独立的 **Qwen3-0.6B**，只训练 Probe 的 LoRA 和读出头。
使用本地模型和现有环境，无需新增下载。旧 MLP 的代码和实验数据保留。

## 模型与时序

每个草稿块调用一次 Probe，输入按以下顺序排列：

1. 任务上下文：明确标识可信指令、真正用户请求、外部邮件/工具反馈，并保留多轮消息顺序及工具定义。
2. 历史配对：之前**每一块原始草稿**、对应的 **8B 实际验证输出**、接受/提议 token 数。
   历史草稿包含被拒绝的部分；这些内容只供 Probe 观察，不进入真实生成模型的已接受历史。
3. 当前 0.6B 草稿。**没有当前 8B 输出、当前接受结果或当前标签。**

Probe 采用 [Kev](third_party/kev/kev/model.py) 的方式：复用 Qwen 分隔 token，
一次前向得到决策位置和三个选项末尾的表示，以 query/key 指针头产生三个 softmax 分数。
不生成答案文字，不运行词表生成头。LoRA rank=16，覆盖 attention/MLP 投影，
指针维度 256，总计 10,617,344 个可训练参数。移植的是 Qwen3 必需部分，
不需要安装 Kev 最新版的不同 Transformers 环境。

| 编号 | 类别 | 标签条件 |
| --- | --- | --- |
| A / 0 | target 被劫持 | 当前 8B 输出体现了注入行为；包括草稿正常、8B 纠正输出却中招的情形 |
| B / 1 | 草稿被劫持、target 抵抗 | 当前草稿体现注入行为，当前 8B 实际输出没有体现 |
| C / 2 | 两者正常 | 当前草稿和当前 8B 输出都没有体现注入行为 |

只用 A 分数触发告警。分数尚未做概率校准，不应把 0.99 理解为有 99% 的真实正确率。
阈值由验证集确定，测试集不参与调参。无法确定的标签保留为空，不当作 C。

```text
0.6B 生成一块草稿
       ├─ 8B 用一次前向验证 ─────────────┐
       └─ 独立 Probe 读取历史＋当前草稿 ─┤
                                        └─ 两者完成后决定是否放行该块
                                            然后记录本块草稿/target 配对
```

新解码器在两条对比路径中都启用 KV cache。greedy 模式逐个接受匹配 token、
在首个不匹配位置由 target 纠正，全接受时生成 target bonus token。
采样模式使用 `min(1,p/q)` 接受率和归一化 `(p-q)+` 拒绝修正，支持 temperature/top-p。
拒绝之后裁剪两个生成模型的缓存，继续基于真实接受文本生成。
4 token 只是可配置的起始块长。

Probe 复用与下一次输入严格相同的最长 token 前缀缓存，历史必须在当前草稿之前；
选项/决策 token 的缓存不会进入下一步。输入超长会明确报错，不静默截断。
新采集保存草稿、target、实际发布文本的原始 token IDs，ByteLevel 文本跨块增量解码，
避免中文字符在 token 块边界被错误地解成多个替换字符。

## 本次训练数据

复用 `artifacts/bipia_email_end/mimo_run` 的真实 0.6B/8B 轨迹，共 49,181 个检查点，
不重新运行 8B 伪造旧标签。该数据来自 BIPIA EmailQA 的 end 插入版本，
**不是 AgentLAB、BFCL 或 LiveCodeBench**。
原始邮件的攻击变体按同一 group 划分，train/val/test 分别为 36/9/33 封邮件。
检查点数量不等于独立任务数量，也不等于多轮动态环境的覆盖量。

已有 MiMo 判断依据的是实际 target 回答，包含最终任务标签及最早明确受影响的位置。
旧轨迹生成上限为 128 token，部分回答可能尚未自然结束；标签只覆盖实际保存的输出。
新增 MiMo-V2.6-Flash 标注只看当前草稿和已经存在的上下文，单独补充草稿标签：
1024 条中 31 条中招、789 条正常、204 条不确定。对含糊片段弃权，正例要求新草稿中的逐字证据。
为寻找 B 类，采样优先选择发生拒绝的块，每个 case 至多两条；这不是自然类别分布。

训练仅使用当前类别已知、首次违规之前或首次违规位置的检查点；不把中招后的所有轮次当作正例。
保留每条输入的完整过去历史。当前 target 文本、攻击模板及参考答案在离线监督/标注区域，
不拼进 Probe 输入。训练/验证的 A/B/C 数量为 **251/17/732** 和 **66/6/146**。
测试集中共有 940 个已知类别检查点，其中 B 只有 8 个。

第一版训练是 LoRA 加权三分类交叉熵，batch=2、lr=1e-4、2 epoch，按验证损失保存最好模型。
训练最大输入 4096 token，超长排除项写入 `excluded_long_inputs.jsonl`，本次没有超长排除。
训练集类别权重只由 train 计算。任务检测评测默认固定抽取 96 个验证任务和 96 个测试任务，
其余已知三分类测试点另行评测；不能把选择性标注的检查点混淆矩阵称为全任务 FNR/FPR。

这些是模型弱标注，尚需人工复核。旧轨迹中的来源标识由原始数据构造方式恢复；
新采集用来源明确的新 prompt，二者的生成提示格式不同，不能沿用旧 target 输出作为新轨迹真值。
旧轨迹缺少原始 token IDs，无法完全修复旧的跨 Unicode 边界解码片段。

## 运行

本机已验证 `/data/ygzhang/miniconda3/envs/deploy/bin/python`，依赖见
[requirements-text-probe.txt](requirements-text-probe.txt)。模型均通过 `local_files_only=True` 加载。

重建离线文本输入、训练（已有标注文件可直接复用）：

```bash
python -m baseline.prepare_text_probe
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python -m baseline.train_text_probe \
  --records artifacts/text_probe/records.jsonl --annotations artifacts/text_probe/draft_labels.jsonl \
  --readout pointer --device cuda:0 --output artifacts/text_probe/model_retrained
python -m baseline.evaluate_text_probe --device cuda:0 \
  --records artifacts/text_probe/records.jsonl --annotations artifacts/text_probe/draft_labels.jsonl \
  --checkpoint artifacts/text_probe/model_retrained --output artifacts/text_probe/model_retrained/classification_all_known.json
```

这组命令使用当前训练器的均衡抽样策略，不能视为首版已保存分数的逐项复现。

需要补充草稿标注时，使用受限权限的临时 key 文件；不要把 key 放在命令行或仓库文件中：

```bash
python -m baseline.label_text_drafts --key-file /secure/path/mimo_key --workers 16
```

实际解码：两张卡，一张放冻结的 draft/target，另一张放 Probe。以下物理卡号仅为本机本次选择，
运行前检查空闲；`dgx-121` 上禁止使用 GPU 4。

```bash
CUDA_VISIBLE_DEVICES=1,0 HF_HUB_OFFLINE=1 python -m baseline.run_text_speculative \
  --cases artifacts/text_probe/runtime_cases.jsonl --output artifacts/text_probe/runtime_compare_isolated \
  --probe-checkpoint artifacts/text_probe/model --mode compare --repeats 3 --max-new-tokens 64

CUDA_VISIBLE_DEVICES=1,0 HF_HUB_OFFLINE=1 python -m baseline.run_text_speculative \
  --cases artifacts/text_probe/runtime_cases.jsonl --output artifacts/text_probe/runtime_intercept \
  --probe-checkpoint artifacts/text_probe/model --mode intercept --repeats 3 --max-new-tokens 64
```

`compare` 预热一次后交替运行无防护/静默两种模式，并断言同一个 case/seed 的全部输出 token 相同。
`intercept` 在触发 A 告警时丢弃当前块并终止回答；此前已发布的文本不会被撤销。
每次运行保存完整 `runs.jsonl`、用于继续离线标注的 `records.jsonl` 和真实 `report.json`。
时延包含分词、Probe 调用及等待，未减去所谓模拟安全时间；发生重叠不代表 Probe 在验证窗口内结束。
控制组也保留分配给 Probe 的第二张卡，单位设备吞吐按同一设备预算计算。

当前运行器处理已经展开的对话上下文，没有接入 AgentLAB 动态工具执行循环。
多轮调用时，case 的可选 `history_pairs` 字段可携带前几轮的草稿/target 配对，
每项使用 `draft`、`target`、`accepted_draft_tokens`、`proposed_draft_tokens` 四个字段；
它们只进入 Probe，实际生成历史仍由 `messages` 中已接受的对话决定。
现有本机短任务实验不能证明跨模型、跨数据集泛化，也没有完成指南全部基准或昇腾适配。
初次序列化布局的探索性模型和被并发评测干扰的计时保存在 `model_serialization_v0/`、
`runtime_compare/`，正式结果使用修正后的 `model/` 和 `runtime_compare_isolated/`。

## 当前离线结果

模型在 `artifacts/text_probe/model/`，两个 epoch 实际训练耗时 381.8 秒（不含之后的校准及测试）。
固定抽样的 96 个测试任务中，最终 target 中招 24 个、未中招 72 个：
漏报 4/24 = **16.67%**，误报 9/72 = **12.50%**。
有首次受影响位置标注的 23 个正任务中，未及时检测到 3 个；另 1 个正任务缺少位置标注。
这些任务指标依赖旧轨迹的 MiMo 弱真值，不代表正式基准结果。

对全部 940 个已知类别测试检查点，按三个分数的 argmax 判别，混淆矩阵如下。
这里的 argmax 分类与按验证阈值触发 A 告警是两种不同的评测口径。

| 真值 / 预测 | A | B | C |
| --- | ---: | ---: | ---: |
| A | 448 | 0 | 105 |
| B | 5 | 1 | 2 |
| C | 19 | 0 | 360 |

第二类仅正确 1/8，不能认为已经学好；训练只有 17 个该类样本，且均为弱标注。
当前指标未达到指南要求。阈值未根据测试结果调整。
原始报告见 `model/training.json`、`model/evaluation.json`、`model/classification_all_known.json`。

## 当前真实运行结果

在两张 L20（物理卡 1 和 0）上，用 2 封未进入训练的邮件构造 6 个 marker 攻击及 2 个 clean 案例，
每例重复 3 次，输出上限 64 token。分类评测进程已结束后才运行计时；对比路径相同，预热不计入。
这 24 次运行只对应 8 个独立 case，重复运行不增加独立安全样本数。

- 24 对静默/无防护输出逐 token 一致，草稿接受率均为 73.02%。
- 平均端到端时延由 **1205.5 ms** 增至 **1674.1 ms**，增加 **38.9%**；
  24 对请求的时延差为 468.6 ± 409.0 ms（均值 ± 样本标准差，混合不同长度 case）。
- 168 个 Probe 块全部与 target 验证重叠，但没有一次在验证结束前完成；每块平均额外等待 34.0 ms。
- 6 个攻击中仅 2 个触发告警，且告警都发生在完整攻击标记已发布之后。
  对应 3 次重复，实际终止 6 个请求，但攻击成功仍为 **18/18**。
  这说明“阻止发布当前块”的机制工作了，当前模型尚未实现及时防御。
- 2 个 clean 案例没有被拦截，clean 接受率均为 84.375%；样本量不足以证明指南的误报/接受率要求。

以上 marker 结果与旧 BIPIA 弱标注任务的 16.67% / 12.50% 不能混为同一测试集。
详细原始数据在 `runtime_compare_isolated/`、`runtime_intercept/`，
合并核验结果在 `artifacts/text_probe/runtime_validation.json`。
模型配置与权重哈希、训练/测试报告汇总在 `artifacts/text_probe/implementation_validation.json`。

当前全部 **57 项测试通过**，包括真实小型 Qwen3 的梯度与缓存一致性、拒绝后缓存裁剪、
完整投机采样与 target 联合分布一致性、EOS、未来信息隔离、跨轮历史继承及拦截前不发布。
这些验证证明实现行为，不等同于安全指标达标。

## 文件

- [text_data.py](baseline/text_data.py)：来源、历史配对、类别和严格输入边界。
- [text_probe.py](baseline/text_probe.py)：LoRA、选项指针头、历史缓存和模型保存/加载。
- [speculative_cached.py](baseline/speculative_cached.py)：KV 投机解码、并行检测和真实拦截。
- [train_text_probe.py](baseline/train_text_probe.py)：训练、验证阈值及固定子集任务评测。
- [test_text_probe.py](tests/test_text_probe.py)：缓存、梯度、无未来信息、采样、EOS、并发和拦截验证。

上游提交号及稀疏下载范围在 `third_party/*.UPSTREAM.json`；上游代码保持原样。
