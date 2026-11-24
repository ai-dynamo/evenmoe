# Prefill 协议

默认全局并发 `[8,32,64,128,256]`，seeds `[8,28,2026]`，7策略，每配置一个服务器会话、3测量轮：105 sessions、315 rounds、315000正式请求。此为完整默认矩阵；pilot 或用户指定子集按实际配置重新计算期望数量。

- 每轮1000条自然长度 instruction；贪心生成1 token，ignore EOS，关闭 prefix/radix cache；eager与否在所有 arms 保持一致。
- 固定请求池，客户端最多同时 C 个请求，完成后补充直到1000条全部完成。所有策略使用同一 seed 的同一顺序。
- 首先用独立校准池预热至少3轮、最多5轮；最近3轮吞吐极差/中位数≤2%判为通过预热检查；不代表正式轮稳定。每正式轮前再跑一次校准池预热。JIT/加载不混入正式时间，达到上限未通过则如实标记。该 3–5 轮及 2% 判据是可配置默认值；跨对照统一，不能反复重跑直到满意。
- 请求到响应完成的整轮 wall time：`K input tokens/s = sum(actual_prompt_tokens) / elapsed_seconds / 1000`。它是仅生成1 token的服务吞吐近似，并非纯 GPU Prefill kernel 耗时；不要改称 TTFT。
- 请求全成功，actual prompt tokens 与准备数据一致，completion tokens=1，cached tokens=0。处理客户端异常、截断、空响应和缺失元数据；不允许默认当零。
- 保存每轮开始/结束边界与实际 token 和时间；额外 profile 用独立轮，不能合入吞吐统计。

参考参数：context length16384、max KV tokens262144、mem fraction0.85、max running requests2048、chunked prefill 全局8192。历史8路Attention-DP解析后每 rank1024，必须检查服务端归一化结果，不能重复除8。新模型/显存配置可调整，但对照一致并生成新实验标识。

复测复用模型、镜像、placement、请求顺序及tokenizer；第一轮三seed均值与第二轮seed8的比较包含输入采样差异。要单独判断稳定性，应比较相同seed；用户要求只打印前述数值时照做并注明口径，不强加相对百分比。

稳定性复测建议同 seed 每并发三次独立服务启动、每次三轮（例如 C8/C32 共 6 次启动、18 个正式轮），与跨 seed 泛化分别报告。全部正式轮保留并绘图，按 [实际 batch](batch.md) 记录服务端各 rank 的执行请求数和 token 数；全局并发不能替代实际 batch。
