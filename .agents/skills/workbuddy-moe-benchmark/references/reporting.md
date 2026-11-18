# 汇总、复测与完成判定

默认不报告方差；报告 median ± sample SD（ddof=1）。只有一个统计样本时SD不可估计，填null/—，不能填0。

层级：每seed每独立session先取三轮中位数；多个session再取该seed的session中位数；跨seed报告seed中位数的中位数及样本SD。仅一个seed的pilot则跨独立session中位数统计，不能冒充跨seedSD。同一服务3轮不是3次独立启动。seed带来的输入差异与同seed复测的时间噪声分开解释。

用户要求“第一轮平均数 → 第二轮seed8”：第一轮对三个seed各自的三轮中位数取算术平均，第二轮取seed8的三轮中位数。表轴为算法×并发，每格两值用箭头连接，缺失用—；无需额外输出相对百分比。

## 标准化输入

先由服务端/客户端审计器验证请求、token数、哈希、trace、完成标记和无干扰边界，再将有效测量转换成一行一个JSON的文件：

```json
{"stage":"prefill","policy":"eplb_waterfill","concurrency":128,"seed":8,"session":1,"round":1,"status":"ok","ktokens_s":15.1}
```

`ktokens_s` 来自对应phase已定义的计时口径，不能把 input TPS 当 output TPS。失败也可记录同样key、status非ok（例如 interference/oom）；无需 throughput。session和round从1开始。重复attempt不能都标为同一有效key：明确选择完整、通过审计的attempt，并在源产物记录排除原因，绝不按吞吐选择。

配置填入 `assets/experiment.json` 的矩阵维度，运行：

```bash
python .agents/skills/workbuddy-moe-benchmark/scripts/summarize.py --plan local-plan.json --rounds audited-rounds.jsonl --output summary.json
```

脚本为标准库工具，拒绝重复/额外key、混合phase、非有限值和非正吞吐；缺失/失败配置标incomplete且不输出完整聚合值；独立记录session/seed medians。它**不验证原始GPU证据**，输出 `raw_audit_performed=false`；数字齐全不等于整个实验有效。首次运行输出必须不存在，避免覆盖历史报告。

## 发布前核查

每条件确认 manifest、实际样本数、精度/后端、全局并发、完整输出长度、数据/tokenizer/placement/runtime哈希、预热稳定性、有效计时窗口、trace 和GPU归属。无新结果时报告具体失败/等待状态，不沿用旧成绩暗示新轮完成。中断轮不与完整轮拼接成独立session。

不提交outputs、HF归档、模型、逐请求私有内容、trace、凭据或内网地址。可分享去标识的小汇总与重现配置，但用户输入/数据许可仍需遵循其来源要求。提交本skill的命令可采用 `git add .agents/skills/workbuddy-moe-benchmark`，不是 `git add .`；本skill不会自动commit/push。
