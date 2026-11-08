# 运行环境、算法和可移植边界

先读当前 repo 的 README、实际安装的 SGLang/FlashInfer/DeepGEMM 代码和 CLI help。历史服务器参数不保证新版本接受；记录最终解析后的 server args，不能只留启动命令。

历史参考为单机8卡H20、DeepSeek-V3、TP8/Attention-DP8/EP8、FlashInfer NVFP4 MegaMoE、eager。框架、模型、GPU或精度变化时重新验证，不把历史成绩当新平台基线。

## 配置指纹

记录 GPU UUID/型号/拓扑、驱动和 CUDA、镜像可拉取地址及 digest、SGLang/FlashInfer/DeepGEMM/本 repo commit、未提交补丁哈希、启动参数、模型配置/权重 revision、tokenizer哈希、placement哈希、量化方式。不要依赖仅存在原服务器的 `local/...` 镜像 tag；若未提供可构建的 Dockerfile 或可拉取镜像，明确环境仍需适配。

本机历史 `MLB_CALIBRATED_V3_NVFP4` 等命名不能证明校准：已有转换审计标注 `stored FP8 weights; no calibration`。权重量化、EPLB placement calibration 与性能 warmup 是三件不同的事。用户要求已校准权重时，必须核验量化算法和校准产物，不能用该环境变量替代。

## 策略映射（历史参考）

| 展示名 | 路由策略 | 冗余 routed experts | 共享专家处理 |
|---|---|---:|---|
| baseline | baseline | 0 | 原路径 |
| EPLB | static | 32 | 原路径 |
| EPLB + LPLB | lplb | 32 | 原路径 |
| EPLB + Waterfill | static+waterfill | 32 | 合入 MegaMoE 并分配目的 rank |
| EPLB + LPLB + Waterfill | lplb+waterfill | 32 | 同上 |
| UltraEP | ultraep | 32 | 原路径 |
| UltraEP + Waterfill | ultraep+waterfill | 32 | 合入 MegaMoE |

参考模型256 logical routed experts，加32冗余得到288 physical routed；每卡36 routed slots；加入每卡一个 shared slot 后296 physical slots、top9。检查打包权重与 placement 完全一致。仅替换 JSON 映射而未搬移对应权重会破坏结果。

EPLB 使用独立52任务校准集收集专家分布并冻结 placement。各算法与 seed 使用同一校准来源，不能用测量集生成 placement。UltraEP 的刷新条件/周期、代表性 Prefill 阈值必须固定并记录；历史纯 Decode 使用已提交 placement，不把“未刷新”当算法失效。

共享专家融合可能改变其量化路径；比较必须披露该组合变化。实际 kernel trace 要看到 `nvfp4_mega_moe`，不能仅凭日志标签宣称 NVFP4；排查 `requires_deepep` 等造成的后端回退。TWF 还需出现 `tilewf_kernel`。

## 给新服务器编写入口

参数化 host、model、image、port、cache、output、seeds、policies、concurrencies。GPU服务控制与 HTTP 客户端解耦，所有产物落到一个新 run root。每个 attempt 保存 manifest、launch、runtime、配置/路由/内核审计、warmup、请求 JSONL、结果、GPU遥测、完成或失败标记。

历史文件如 `benchmarks/prepare_workbuddy_prefill.py`、`run_workbuddy_prefill.py`、`workbuddy_decode_client.py`、`run_workbuddy_decode_pilot_fixed.py` 可用于本地源码对照，**不是本 skill 必需文件**，也不能在新 checkout 假设存在。未移植服务端 adapter 或计时 hook 时，不声称一键复现成功。本 skill 自带的数据与汇总脚本可独立使用；真实 GPU 服务仍须目标平台的依赖、后端适配和验证。
