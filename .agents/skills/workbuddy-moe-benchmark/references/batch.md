# 服务端实际 batch

配置的全局并发只是客户端在途请求上限，不等于每 rank 的执行 batch。运行时适配器在实际模型执行边界采集元数据，标明采集点是调度、提交还是完成；CUDA 异步提交时间不能冒充 GPU 完成时间。step_id 只在所属 rank 内解释，不假定不同 DP rank 的同编号 step 同步。

## 最小逐 step 记录

```json
{"policy":"baseline","concurrency":8,"seed":8,"session":1,"round":1,"attempt":1,"rank":0,"tp_rank":0,"dp_rank":0,"ep_rank":0,"step_id":17,"timestamp_ns":123456789,"clock_domain":"host-monotonic","execution_phase":"prefill","requests":4,"tokens":1024,"padded_tokens":1024}
```

同时保存配置指纹、服务启动 UUID、运行时 commit、计数 hook 位置、rank 拓扑、计数是否复制、单位定义及采样覆盖率。将记录关联至正式轮的起止边界，排除预热和前后轮串入，校验无缓冲丢失/重复；工具无法单独证明覆盖完整。

- requests：本次执行实际参与的请求数，不是排队数或全部活跃请求数。
- tokens：本次实际处理的有效 token 数。chunked Prefill 只数当前 chunk，不数完整上下文长度。
- padded_tokens：采集边界处含 padding 的输入张量容量，不能小于 tokens；记录采集在 padding 前还是后，不能把输入容量称为 kernel 内部 tile padding，也不要把 padding 当成有效吞吐。
- execution_phase：区分 prefill、decode、mixed、target_verify、draft、idle。混合批次需附 prefill_tokens/decode_tokens；MTP 的验证候选 token 数不等于最终接受的输出 token 数。
- TP 可能复制同一请求和 token，不得将各 TP rank 相加。DP 可独立处理不同请求，合并前仍需验证去重和窗口；EP 接收的 token-expert assignment 是另外一种算子负载，另字段记录，不能充当请求 batch。

优先使用已有 CPU 调度元数据和预分配缓冲，轮末导出。不得为了逐 step 计数在计时路径新增 GPU 同步或大量同步日志。记录开关状态；用匹配的开启/关闭小样本评估采集开销，若开销不可忽略，吞吐主测与重型诊断拆开。诊断轮的分布只能标 diagnostic_only，不能称为正式慢轮的实际 batch。没有可靠 hook 时明确 unavailable，并交付待适配点。

## 汇总与关联

```bash
python .agents/skills/workbuddy-moe-benchmark/scripts/summarize_batches.py --steps steps.jsonl --coverage all_steps --output batches.json
```

coverage 为 all_steps、sampled 或 diagnostic_only；是采集者声明，输出 coverage_verified=false，须用启动/结束计数、丢失计数等外部证据审计。脚本按 policy/concurrency/seed/session/round/attempt/rank/phase 分组，不跨 rank 相加。输出 step 数，以及 requests/tokens/padded_tokens 的 min、P10/P50/P90、max、mean（每 step 等权）。idle 零负载合法，须单独阶段报告。分位数使用排序后的线性插值。

报告在逐轮吞吐旁链接或展开相应各 rank 的 batch 分布，必要时画 token 数分布或吞吐与 batch P50/P90 的散点图。批次步长差异很大时可补充按时间加权分布，但必须另标权重，不替代每步统计。不能仅凭 batch 变化证明吞吐波动的原因。

## 与专家路由关联

请求数和 token 数无法还原专家分布。需保留 top-k ID 时，按[路由保存协议](routing.md)保存原始数组、token 身份及专家到 rank 的映射，用 run/session/round/attempt、rank、step、模型角色关联本文件的 batch 记录。独立诊断轮只能与吞吐配置配对，不能冒充同一次执行。
