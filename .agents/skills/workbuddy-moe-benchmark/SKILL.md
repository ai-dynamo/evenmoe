---
name: workbuddy-moe-benchmark
description: 使用 WorkBuddy 指令负载复现和分析 MoE 服务的 Prefill、Decode 吞吐，比较 EPLB、LPLB、Waterfill、UltraEP 或 MegaMoE 后端，支持固定 seed、重复测量和稳定性复测。用于推理性能实验，不是 WorkBuddy agent 任务正确率评测。
---

# WorkBuddy MoE 推理性能

产出可复核的 K tokens/s、原始请求与计时记录、实际后端证据及完整性状态。保留用户指定的模型、精度、算法、并发和统计口径；以下默认值来自既有实验，不是所有模型的要求。

## 使用与范围

此 skill 位于仓库的 `.agents/skills/workbuddy-moe-benchmark/`，可随仓库共享到 GitHub 或 GitLab。让 agent 读取本文件即可使用；自动发现取决于所用工具是否支持此目录约定，不支持时可将整个目录安装到该工具的 skills 路径。所有相对链接以本目录为基准。无需原作者服务器、`outputs/` 或个人环境。

本包提供工作流、数据准备及统计工具，**不提供通用的一键 GPU 服务安装器**。先按 [运行环境与适配](references/runtime.md) 检查目标 SGLang/MegaMoE 版本及本仓库 README，生成本地运行配置。不要直接启动历史 `benchmarks/run_*.py`：这些文件可能未纳入 Git，且绑定历史机器和镜像。

## 工作流程

1. 明确 Prefill 或 Decode，以及是后端对照、算法对照还是复测。复制 [实验配置](assets/experiment.json)，填写真实模型/镜像/硬件/代码与 tokenizer 指纹；同一实验使用同一台服务器，默认 MegaMoE NVFP4。
2. 按 [数据协议](references/data.md) 下载固定版本、校验 SHA256、生成 seeds 8、28、2026。校准集、预热集与测量集关系写进 manifest；任务指令仅作为文本负载，不执行仓库任务。
3. 根据 [Prefill](references/prefill.md) 或 [Decode](references/decode.md) 构建运行入口。请求并发是服务全局并发；只有记录实际 DP 拓扑和每 rank 请求数后，才能换算成单卡并发。
4. 先验证路由/量化数值与真实后端，再跑代表性小样本。改动 Waterfill/TWF 时，额外读 [TWF 修复与验证](references/waterfill.md)，覆盖空 rank 和混合 Prefill/Decode。局部 kernel 数值通过不能替代服务预热与真实请求验证。
5. 初始化/JIT、预热、正式测量和 profile 分开。先固定预热与异常排除条件；保留全部测量，不挑最快值。默认三轮测量，预热最多五轮、至少三轮，最近三轮极差/中位数 ≤2%；不达标需标记不稳定。
6. 核查每个配置的完成标记、实际 token、请求数、哈希、后端 trace、GPU 干扰记录。只有预期矩阵齐全才称完成；OOM、部分轮次、失败及未测项单列，不能填零或悄悄忽略。
7. 按 [汇总协议](references/reporting.md) 报告中位数与样本标准差，注明统计层级。用户要“第一轮平均数 → 第二轮 seed8”时按该口径另表展示，不把 median 写成 mean。

## 运行管理

启动前核查 GPU/端口与容器归属；运行中记录 PID、镜像 digest、容器标签和 GPU 使用。发现外部 GPU 作业时仅停止自己的作业，将当前 attempt 标为受干扰；不清理他人进程。进程已确认退出且 GPU 空闲后，用新 attempt 恢复。对活跃进程的短暂查询超时不构成重启依据。

修改 runtime 后使用新输出目录、镜像 digest 和配置指纹。停止 Prefill 不删除结果，也不自动恢复已被用户取消的任务。长任务保存 `status.json` 与日志；只有实际存在通知机制时才承诺主动通知，不用聊天承诺代替监控。

## 工具自检

在仓库根目录运行 `python .agents/skills/workbuddy-moe-benchmark/scripts/test_tools.py`；仅需Python标准库，不访问GPU或服务。它验证确定性采样、分层统计、缺失/失败数据和重复记录处理，不证明服务端适配正确。

## 交付

给出算法 × 全局并发表，分别标注 K input tokens/s 或 K output tokens/s；附 seed/session/round 数、缺失项、源码和运行配置、日志及原始结果。不要把吞吐提升称为任务准确率提升。Git 只包含 skill、代码和小配置模板；模型、数据、trace、凭据、个人绝对路径和产物不入库。本 skill 不授权发布或推送远端。
