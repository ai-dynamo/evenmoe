# 汇总、波动与完成判定

保留全部正式轮、预热、失败和重试记录。有效但偏慢的轮次纳入主统计；不得因吞吐低、偏离中位数或箱线图判为离群点而排除。请求不完整、计时错误、外部作业干扰等可按预先声明的规则排除，但保留原记录、证据与原因。重试使用新的 attempt，不覆盖旧记录；重新启动服务必须使用新的 session，并在计划中明确追加会话，不能把多个不完整启动拼成一次启动。

预热通过只表示满足进入正式测量的条件，不保证正式轮稳定。完成状态、有效性、波动标记分别报告。单个 seed 的复测只能说明该输入样本下的重复性，不能宣称跨模型、硬件或 seed 稳定。

## 两级统计

- 每次独立服务启动内：全部正式轮原始值、median、样本 SD（ddof=1）、min/max、极差/median、CV（样本 SD/mean）。保留轮次顺序，不把低吞吐作为无效原因。
- 同一 seed 的不同启动之间：各 session median，以及这些中位数的 median 和样本 SD。三个启动各测三轮是九个测量点、三个独立启动，不能当作九个独立启动。
- 跨 seed：先取各 seed 的 session medians 的中位数，再报告这些 seed 中位数的 median 和样本 SD；与同 seed 时间噪声分开解释。默认不报告方差。
- 单样本 SD 填 null/—，不能填零。未完成配置仍输出已有逐轮及 session 描述统计并标记 incomplete，不输出冒充完整矩阵的总聚合。

任何“波动百分比”须写明公式。默认极差/median = (max-min)/median；它不同于 CV。不把通过预热检查当作正式轮稳定的证据。

用户要求“第一轮平均数 → 第二轮 seed8”时，第一轮对三个 seed 各自的三轮中位数取算术平均，第二轮取 seed8 的三轮中位数；注明包含输入采样差异。

## 原始记录与汇总

保存逐请求原始证据，再生成一行一条记录的 JSONL。每行包含配置指纹 config_id、stage、policy、concurrency、seed、session、attempt、round、phase、起止时间、实际 token 数、elapsed_seconds、ktokens_s、status、included、flags 及源证据路径。phase 为 warmup 或 formal。计时口径由 Prefill/Decode 协议决定；时间单位与时钟域必须标注。

```json
{"config_id":"example-fingerprint","stage":"prefill","policy":"baseline","concurrency":8,"seed":8,"session":1,"attempt":1,"round":1,"phase":"formal","status":"ok","included":true,"flags":[],"ktokens_s":15.1}
```

以上为最小统计示例，不替代完整计时/token 证据。失败可没有吞吐，不填零；有可用吞吐的排除轮仍保留该值。included=false 的成功正式轮必须有 exclusion_reason，并在源证据中说明；不能用“偏慢”作为排除理由。同一逻辑测量最多选择一个 included attempt，按预先声明的完整性规则选择，不能按吞吐选择。

```bash
python .agents/skills/workbuddy-moe-benchmark/scripts/summarize.py --plan local-plan.json --rounds all-rounds.jsonl --output summary.json
python .agents/skills/workbuddy-moe-benchmark/scripts/plot_rounds.py --summary summary.json --output-dir plots
```

汇总脚本只用标准库，兼容旧记录（默认 formal、attempt=1，成功正式轮 included=true），保留 round_records、session_statistics、session_medians 和 between_session_by_seed。按一个配置指纹/模型/后端分别运行，不能混合不同硬件配置。它不验证 GPU 原始证据，raw_audit_performed=false。输出不可覆盖。

绘图依赖 matplotlib，独立于无绘图依赖的统计工具。箱线图按算法、并发、seed 分组，叠加所有有测量值的正式轮：颜色标识服务启动，点形标识 round，完整 session/round/attempt 对应关系保存在点位清单；被排除的点使用叉号且不纳入箱线图。无吞吐的失败记录列入 plot_manifest.json，不伪造点。另图按 round 顺序连接同次启动的有效测量；少量样本的箱线图仅作描述，不作显著性推断。导出 PNG、SVG 和点位清单，预热独立保存在 summary.json，不混入正式箱线图。

## 实际 batch 与报告交付

按 [实际 batch 协议](batch.md) 采集并报告每轮、每 rank、每执行阶段的请求数/token 数分布。用相同 session/round/attempt 关联吞吐，检查慢轮是否伴随 batch 分布变化。相关性不是因果证明；无法采集时标 unavailable，不用配置并发代替。诊断轮与正式测量的证据必须分开。

交付总体统计、启动级统计、全部逐轮 JSONL/CSV、箱线散点图、轮次趋势图、预热记录、batch 分布、缺失和排除说明。对同 seed 的稳定性验证，建议每配置三次独立启动、每次三轮，轮数在实验前固定；不要观察结果后追加到“稳定”为止。

发布前核对模型/硬件/后端、输入/tokenizer/placement/runtime 指纹、实际计时窗口、请求/token 完整性及 GPU 干扰。没有实机测量时只能称工具测试或历史数据重分析；不能把模拟数据当作新性能结果。原始私有提示、模型、凭据、内网地址和大 trace 不进入 Git；可提交去标识的小型测试夹具。该 skill 不自动 commit/push。

## 专家路由产物

配置要求保存 top-k ID 时，交付 [routing 协议](routing.md)规定的 manifest、step 元数据、原始 ID 数组、token 身份、映射版本与完整性报告，而不只交付专家直方图。报告覆盖率、缺失原因和诊断轮关联；有物理副本却缺实际 dispatch 目标时，只报告逻辑专家热度，不能声称已得到 owner rank 负载。原始 trace 放产物存储，Git 只提交协议及代码。
