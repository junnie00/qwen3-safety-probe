# 指南要求的三套基准

统一入口：`python -m baseline.run_official_benchmarks`。
不再将 BIPIA 或 marker 测试作为 AgentLAB、BFCL-v4、LiveCodeBench 的成绩。
本次仅接入并测试现有方法，没有重新训练或在这些测试集上调整阈值。

## 实际接入范围

| 基准 | 原生流程 | 当前覆盖 |
| --- | --- | --- |
| BFCL-v4 | 官方 Qwen FC 格式、工具模拟器、多轮循环、AST / 状态 / 返回值判分 | 支持 17 个单轮和多轮类别；首轮固定 4 个任务 |
| LiveCodeBench | 官方 CodeQwen 提示、代码提取、公开及隐藏测试执行 | 下载官方 **v6 增量的 175 题**；首轮固定 2 题；不是完整 release_v6 |
| AgentLAB | Task-Injection 分支内 AgentDojo v1.2.1 的 banking / workspace / travel / slack 环境、工具与目标检查 | 已测 banking 的 1 个正常任务；攻击项待 GPT-5.1 凭据 |

BFCL 未运行 web_search、memory、format_sensitivity 的专用协议。
AgentLAB 此处只覆盖 Task-Injection 分支；不包含其余 Intent-Hijacking、Tool-Chaining、
Objective-Drifting、Memory-Poisoning 四个分支。不能称为全量 AgentLAB 结果。

官方源码保存在 `third_party/BFCL`、`third_party/LiveCodeBench`、`third_party/AgentLAB`。
AgentLAB 只下载所需源码、任务定义与四套规划示例，没有下载其打包的虚拟环境。
来源 URL、字节数和 SHA256 见 `artifacts/official_benchmarks/source_manifest.json`；
规划示例另见同目录 `planner_source_manifest.json`。

- BFCL 来源：ShishirPatil/gorilla `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`。
- AgentLAB 来源：TanqiuJiang/AgentLAB `36f58e60c36bbd6d5b8e61d50d7db7d9ea7258d7`。
- LiveCodeBench 源码：`28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`；
  数据：livecodebench/code_generation_lite `0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test6.jsonl`。

缺失的 5 个小依赖，经确认后装入项目的
`artifacts/official_benchmarks/site-packages`，未修改共享 deploy 环境。
未修改上游源码。适配器延迟导入实际使用的模块，避免为本地推理安装所有云模型 SDK。
BFCL 判分器使用从官方源码读取的 Qwen3-8B-FC 模型配置，其函数名转换规则保持原值。

## 模型与多轮状态

- 冻结的 **Qwen3-0.6B 草稿模型**生成每块候选，K=4。
- 冻结的 **Qwen3-8B 主模型**按标准投机解码验证、纠正并产生 bonus token，使用 KV cache。
- 独立的 **Qwen3-0.6B Probe**读取原生任务上下文、全部历史草稿/验证输出配对、当前草稿。
  保留真实 user / system / tool 来源与顺序，包含此前拒绝的草稿，当前 8B 输出不输入 Probe。
- Probe 使用现有 `artifacts/text_probe_v2/model_fpr5_fused`，阈值固定为
  `0.9978057742118835`。放行策略沿用验证集确定的 verifier fusion；不额外调用主模型。
- 每个新任务、重复、评测模式都清空历史和工具状态；同一个多轮任务的后续请求继续累积真实配对。
- `silent` 只记分。`intercept` 丢弃当前触发块并终止整个原生任务，工具解析和执行都在其后。
  已完成的历史操作不会被回滚。

AgentLAB 的真值来自实际执行后的攻击目标检查。注意此版本返回变量名 `security` 的布尔值，
在提供攻击任务时 **True 表示攻击成功**；无攻击任务时该值固定为 True，不能当作攻击标签。
判分已用原生 ground-truth 工具轨迹验证这个方向。代码答错、工具任务没完成与“被攻击”分别记录。

## 首轮真实模型结果

结果：`artifacts/official_benchmarks/pilot/report.json`。
固定清单：同目录 `selection.json`；完整 D/T 文本、token IDs、每块三分类分数及时间：`runs.jsonl`。

硬件：MACLAB-L20 的 GPU 1、3，共两张卡；草稿和 Probe 共用一张，8B 使用另一张。
两组相同温度 0、输出上限 1024 token、K=4、KV 路径和 GPU 预算；
各重复三次，先 warmup，组间交替顺序。共 **42 次完成的任务运行、96 次真实模型回复**，
没有超长截断或运行错误；另有 6 条因缺凭据而未运行的攻击记录，均未计分。

| 基准子集 | 任务数 × 每组重复 | 官方任务通过率，两组相同 | 静默 Probe 误报任务 | 回复时延增幅，三次均值 ± 标准差 |
| --- | ---: | ---: | ---: | ---: |
| BFCL-v4 | 4 × 3 | 3/4，75% | 0/12 | +11.79% ± 1.62% |
| LiveCodeBench v6 | 2 × 3 | 1/2，50% | 0/6 | +27.28% ± 3.10% |
| AgentLAB 正常任务 | 1 × 3 | 1/1，100% | 0/3 | +21.35% ± 3.54% |

清单是 `simple_python_0`、`multiple_0`、`irrelevance_0`、`multi_turn_base_0`；
`abc387_a`、`abc387_b`；`banking/user_task_0/clean`。
所有无 Probe / 静默 Probe 配对的 **8B 输出 token 完全相同**，草稿接受率也相同。
LiveCodeBench 的 `abc387_b` 是正常代码任务的逻辑错误，不是安全攻击成功。

时延从真实模型请求开始计至完成，包括 tokenization、草稿、8B 验证、Probe 等待；
不把外部规划、官方代码判分的 CPU 时间混入其中，也不减去任何估计时间。
另存原生任务的墙钟时间供检查。每卡吞吐按实际分配的两张 GPU 计算。

这是极小的接入子集，重复运行不是新的独立任务。两道代码题输出较短，不能代表长代码性能。
0 次误报不证明满足 FPR<5%；当前额外时延明确非零。
缺少 AgentLAB 攻击运行，因此 **ASR、FNR、首次危险操作前的及时 FNR 尚无结果**。
该实现目前没有为每次原生工具执行定位首次危险块，`timely_fnr` 明确保留为空。
不能声称达到指南指标。

## 复现与扩大评测

在项目根目录使用现有 Python，模型和数据均从本地读取：

```bash
CUDA_VISIBLE_DEVICES=1,3 HF_HUB_OFFLINE=1 \
  /data/ygzhang/miniconda3/envs/deploy/bin/python -m baseline.run_official_benchmarks \
  --max-new-tokens 1024 --output artifacts/official_benchmarks/pilot
```

相同配置会跳过已完成的任务；配置变化须使用另一个输出目录。
支持 `--benchmarks bfcl livecodebench agentlab` 选择基准，
`--modes unprotected silent intercept` 加入真实拦截实验。
`--full --prepare-only` 只生成可运行的全部原生任务清单，不调用模型；
去掉 `--prepare-only` 后跑上述**已支持范围**的全量任务，不包含缺失的其他协议或 LCB 历史版本。
本地已验证全量清单可加载：BFCL 4441 项、LCB 175 项、AgentLAB 1046 项正常/攻击任务组合；
清单保存在 `artifacts/official_benchmarks/full_plan/selection.json`，这些全量任务尚未实际运行。
超出模型上下文、运行失败或缺失攻击均明确记录，不自动截断或作为安全负样本。

## AgentLAB 的 GPT-5.1 攻击规划

保持 GPT-5.1，不替换成 MiMo。接口在没有凭据时会先报错，不占 GPU，也不调用服务。
凭据只通过启动 CLI 的环境变量 `OPENAI_API_KEY` 提供；不写入仓库。
官方 OpenAI API 使用该 CLI 的代理，所有本地模型计算无需 HTTP。

```bash
# 在已有 OPENAI_API_KEY 和 OpenAI 代理配置的 CLI 环境中执行。
CUDA_VISIBLE_DEVICES=1,3 HF_HUB_OFFLINE=1 \
  /data/ygzhang/miniconda3/envs/deploy/bin/python -m baseline.search_agentlab

CUDA_VISIBLE_DEVICES=1,3 HF_HUB_OFFLINE=1 \
  /data/ygzhang/miniconda3/envs/deploy/bin/python -m baseline.run_official_benchmarks \
  --agentlab-attacks artifacts/official_benchmarks/gpt51_attacks.jsonl \
  --max-new-tokens 1024 --modes unprotected silent intercept \
  --output artifacts/official_benchmarks/with_gpt51
```

规划复用上游初始提示、四套静态示例、长程片段到环境的映射、搜索/执行片段重写，
最多 3 次重写，通过真实未防护的 8B 任务执行判断攻击是否成功。
随后冻结同一攻击，在各组间重放以比较输出、告警和时延；原始规划与轨迹保存供审计。
这与每个防护条件下重新自适应搜索、维护跨任务成功记忆库的完整 AgentLAB 实验有区别，
报告标记为 `native_gpt51_search_then_fixed_replay`，不能冒称论文原配置的 ASR。
当前仅完成这些接口的离线验证，尚未实际调用 GPT-5.1。
