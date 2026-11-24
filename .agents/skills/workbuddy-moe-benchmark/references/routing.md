# 逐 step / 逐层专家路由保存

在需要专家热度、rank 不均衡度或 batch 内专家分布时读取本协议。普通 batch 计数无法还原 top-k ID。此 skill 规定产物与核验方式，不提供跨模型通用路由 hook；适配器必须核实当前后端的路由、分片及专家 ID 语义后接入。配置要求保存但适配器不可用时，标记 unavailable 及原因，不得声称已经采集。

## 采集与吞吐隔离

配置模板的 `routing_capture` 要求保留原始路由，默认安排独立 `diagnostic_only` 轮，复用吞吐实验的 workload 哈希、seed、输入顺序、算法、模型、并发及生成参数。仅在用户授权的运行范围内执行；用户仅要求修改 skill 或禁止复跑时，不启动诊断。

诊断保存自己的 run/session/round/attempt 和 `paired_throughput_run_id`，明确两次执行不能逐 step 一一配对，即使 seed 相同实际 batch 也可能不同。不得将诊断 trace 当作正式慢轮的实际路由。若需要同轮关联，必须在正式轮内采集并通过匹配的开关开销检查；完整性核验通过也不能证明无性能开销。

优先使用已有路由输出和有界异步缓冲，在不改变路由选择的前提下保存。不能假定各后端都有 Python top-k hook；融合路由需对应后端的真实输出接口。吞吐路径不新增逐层同步 GPU→CPU 拷贝或逐条同步磁盘写入。缓冲区满时明确丢失计数并将覆盖状态降为 incomplete，不能静默丢弃。重型同步采集只在标记清楚的诊断轮使用，不能合入主吞吐统计。

## 必须保留的产物

1. `routing/manifest.json`：schema_version、run_id、配置指纹、模型/运行时/采集代码版本与哈希、workload/tokenizer 哈希、seed、算法、并发、采集开关、hook 位置（dispatch 前/后）、覆盖范围、过滤规则、拓扑、ID 空间和采集开销。区分 intended coverage 与 verified coverage；缺失原因单列。
2. `routing/steps.jsonl`：沿用 [batch 协议](batch.md)，补充 run_id、session_uuid、model_role、execution_id、token_axis_id、placement_id，并保留请求数、有效 token、张量容量、阶段和 CPU 提交时间。step_id 只在源 rank 内唯一；execution_id 只有经适配器验证表示同一次分布式执行时才能用于跨 rank 对齐。
3. `routing/routes/`：逐 rank、逐 step、逐 MoE layer、逐调用的原始 top-k ID 数组，不能仅保存直方图。记录 layer_id、call_index、model_role、execution_phase、shape、dtype、valid_mask、token_axis_id、ID 空间及 placement_id。target 与 draft 的 layer_id 和专家空间分别编号；共享专家若不在 top-k 内，另记而不补造 ID。
4. `routing/token_axes/`：路由数组每行的 token 身份和有效性。至少保存 request_id、请求内绝对 token position、chunk 范围或 token ordinal。Decode/MTP 还需 role、验证块/候选位置；树状验证需 node/branch ID，避免同一位置的候选混淆。Prefill 只记录当前 chunk 的 token。padding、无效候选和哨兵 ID 单独掩码，不能算专家命中。
5. `routing/placements/`：每个生效版本的 logical_expert → physical_replica → owner_rank/local_slot 映射、拓扑组成员和生效执行边界。静态 EPLB 保存初始完整映射；动态迁移保存每次更新，路由记录引用准确版本。不能只保存最终映射。
6. `routing/completion.json`：预期/实际 step、layer-call 和有效 token-route 计数，缺失/重复/丢失/未完成异步拷贝数，以及每个文件的校验和。保存失败或截断的分片及原因，不将文件存在等同于采集完整。

原始 ID 可存为分片 `.npz` 或其他带 dtype/shape 的无损数组；JSONL 只作索引，使用相对路径与 SHA256。压缩可用但不得损失 ID、掩码或映射版本。下面是单条路由索引示例，字段值仅作示意：

```json
{"schema_version":1,"run_id":"diagnostic-001","session_uuid":"launch-001","policy":"baseline","concurrency":8,"seed":8,"session":1,"round":1,"attempt":1,"source_rank":0,"tp_rank":0,"dp_rank":0,"ep_rank":0,"step_id":17,"execution_id":"target-exec-17","model_role":"target","execution_phase":"prefill","layer_id":3,"call_index":0,"token_axis_id":"axis-17-r0","placement_id":"placement-000","id_space":"logical","replica_group_id":"tp-group-0","replication":"replicated","hook_location":"pre_dispatch","artifact":"routes/rank0-part000.npz","array_key":"step17_layer3_call0_ids","shape":[1024,8],"dtype":"int32","valid_mask_key":"step17_layer3_call0_valid","physical_assignment_key":null}
```

`valid_mask` 必须能区分有效 token 行及有效 top-k slot；不能把零专家 ID 当 padding。K 可随模型/层变化，按实际 shape 保存。路由权重可选，只有分析加权热度时才需要；纯命中数不需要权重。

## 去重与专家归属

- 同一个逻辑 expert_id 不等于当前 rank 的本地专家下标；保存 `logical`、`physical` 或 `local_slot` 的 ID 空间并用对应映射解释。不要用整除 expert_id 的方式推断归属。
- TP 上相同路由副本只计一次。先按执行、模型角色、层/调用、token 身份核对副本 ID 一致性，再保留一个副本；记录差异，不能默认 rank0 代表全部数据。分片 token 的 rank 要按 token 身份合并，不能按 TP rank 名称直接丢弃。
- `source_rank` 是采集者，不一定是执行该专家的 owner rank。没有实际 dispatch 目标时只能得到逻辑专家热度。专家有多个物理副本时，logical ID 加候选映射不足以确定实际 owner，必须额外保存每个 token-slot 的 physical replica / destination rank；缺失时标记 rank 负载 unavailable，不能平均分摊猜测。
- DP 请求/执行流独立，跨 DP 汇总前核对 request 和 token 身份；不能把不同 DP 的相同 step_id 对齐。rank 不均衡度只在同一可对齐执行和相同 EP 组内计算，包含零负载 rank。
- MTP 分开统计 target verify、draft 以及普通 decode；候选路由负载全部计入实际计算，但不能称为最终接受 token 热度。若报告 accepted-only，须另存接受掩码及其与候选身份的关联。

## 保存后可复现的统计

从有效且去重后的 token-slot 记录计算 `count(expert=e)`，单位为 token–expert assignment。固定 K 且每个 token 的 ID 唯一时，总量应等于 valid_tokens × K；其他情况按 valid_mask 实际 slot 数核验。若需要“接收过该专家的唯一 token 数”，显式按 token 身份去重并另标口径。

支持单个执行 batch/层的 expert 柱状图、step × expert 热力图、owner rank 内各专家分布及 step × owner rank 热力图。rank 负载为其实际接收的 assignment 数，`max(rank_load) / mean(rank_load)`；全零执行记 undefined，不记 0 或 1。图中明确 model_role、层、阶段、映射版本、计数单位、覆盖率与是否诊断轮，不能混合层或 MTP 阶段后不加说明。

核验数组形状/ID 范围、掩码、token-axis 对齐、映射版本、重复键和完成标记；对于 TP 副本核验一致性，对于物理 dispatch 核验守恒。仅有部分层/step 时按 sampled 或 incomplete 报告。轻量 batch 总数校验不能代替逐层路由完整性校验。

## 已保存数据的示例

[WorkBuddy V4 target 非 hash 层诊断](workbuddy-v4-target-imbalance.md)给出 baseline 与静态 EPLB 的指标与采集限制，示范 assignment 与唯一 token 两种口径、全批次与全部 step 的分开报告。若 TP 路由副本不一致，不能继续按相同副本去重；需依据真实 dispatcher 语义和各 owner rank 的实际选择重建执行负载，否则标记无法重建。此时不要强制套用一个全局 top-k 数组的 tokens × K 守恒。
