# Decode 协议与计时

先 pilot，再完整矩阵。历史 TWF pilot 使用 seed8、全局并发128/512、每请求512个输出token；原实现/仅1024B对齐/仅TWF/两者同时开启，固定 EPLB+Waterfill placement，2次独立启动，每次每并发3测量轮，共8 sessions、48个测量 cohort。顺序第一遍正序、第二遍逆序。TWF组合包含融合路由，不是纯算法单因素。

完整算法矩阵需独立声明 seeds、并发、会话数和刷新策略，不能把 pilot 称为完整结果。可扩展512/1024/2048，但先确定请求池大小、KV容量与实际满载情况；不静默降低并发或输出长度。

## 请求方式

每个 cohort 恰好 C 条请求并发提交，取对应seed工作负载前 C 条，不补充。贪心、ignore EOS、固定输出512、stream interval1、禁用prefixcache、eager。预热用校准池；正式测量用测试seed。记录逐请求ID、hash、输入/输出token、流式事件、首token和完成时刻；输入和生成长度必须核对。

## 主指标：服务端纯 Decode 已完成步骤

客户端到达不同步会混入 Prefill，故采用所有rank稳定满载的共同区间：

1. 每rank在一次 Decode step 的输出已完成后记录单调时钟 `ns`、请求ID列表、各请求输出长度、local/global decode、graph状态。历史hook位于既有输出拷回CPU的同步之后；不要在测量路径额外逐step `cuda.synchronize()`。
2. 每rank保留该cohort的请求，确认无其他cohort混入；均衡DP8时每rank有C/8请求；输出长度全部在 `[128,384]`；只含纯Decode且非graph。不满载/窗口不足判无效，不能当慢结果补齐。
3. 各rank首条记录时间的最大值为 start，末条时间的最小值为 end；统计 `start < ns <= end` 的**完成**输出token，每rank至少128步。
4. `K output tokens/s = sum(completed_tokens) / ((end-start)/1e9) / 1000`。

上述CPU时间戳比较只适用于同一主机同一时钟域；不能相减不同GPU的CUDA事件时间，也不能直接跨服务器相减CPU monotonic时间。其他拓扑必须重新定义计时边界。CUDA异步提交时间不是完成时间。

辅助报告：逐请求TPOT（排除自身TTFT但可能混入其他请求Prefill）、流式gap/每次token增量，以及共同客户端窗口吞吐。客户端共同窗口从最晚首token到最早末token，包含传输/调度，需与服务端主指标分列；不得相互替代。

无hook时只能报告明确标注的客户端指标；先适配并验证hook后再给纯Decode结果。可用合成8rank记录验证窗口计算，缺rank、重复请求、跨cohort、少于128步必须拒绝。

## Warmup 与验证

同Prefill采用固定3~5预热窗口的稳定性判据，每正式cohort前一轮预热。Profile放在测量后，单独C128 cohort，在所有请求首token到达后采集约8步，检查全部8张卡的kernel。

TWF首次接入先验证C1、3、8的空rank场景，再C128/512长Decode，再trace；不要只用均匀batch微基准宣布修复。保存所有失败attempt，服务预热结束和health ready都不等于测量完成。
