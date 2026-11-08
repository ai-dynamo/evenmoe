# Waterfill / TWF 的适配与已验证故障

TWF = Tile-aware Waterfill，按计算tile与行数共同估算负载，分配共享专家token。所参考实现按64行tile规划，并融合静态expert映射、计数、P2P交换和top9输出。NVFP4的计算布局未必与原FP8相同，性能要实测。

来源（代码版本参考，不要求整仓替换）：
- SGLang演示：https://github.com/xutizhou/sglang-moe-lb-waterfill-dev/commit/a85e2a43307d436477d096647066cd4899b64f87 。该commit主要是复现文档/脚本，不能将全部优化归因于它。
- DeepGEMM：https://github.com/xutizhou/DeepGEMM/tree/5a9f6f078935f362e5d54cab89709fe58bf43dc2 。SM90 symmetric data对齐到1024B，另有FP8 expert-wave调度优化。
- EvenMoE依赖：https://github.com/xutizhou/evenmoe/tree/c56ada7f3dd82dc9fa17815eabca011b2b5eb065 。配套依赖不是NVFP4兼容性的证明。

上游demo为FP8+CUDA graphs；本实验为FlashInfer NVFP4+eager。1024B对齐、TWF、expert-wave调度是不同变化，分开验证；不能把FP8调度直接复制到NVFP4。当前TWF仅验证固定static EPLB+Waterfill，不能自动推广LPLB/UltraEP。若移植第三方源码，按其许可证保留归属，本skill未打包第三方CUDA源码。

## 实际bug：部分rank走错collective

SGLang某版本的 `global_forward_mode` 只在TBO启用且可执行时有效，否则为None。错误分支：

```python
mode = batch.global_forward_mode or batch.forward_mode
if mode.is_decode():
    twf_route()
else:
    original_route()
```

此时本地有请求的rank进入TWF，而本地IDLE rank执行原Waterfill，collective不一致。首次TWF symmetric-memory rendezvous挂起，随后可能在其他量化调用处报告 `unspecified launch failure`。CUDA异步错误位置不一定是根因。均匀batch或直接调用路由的数值测试不会覆盖这个集成问题。

已验证的修复逻辑（需确认目标版本提供相同语义字段）：

```python
def is_collective_decode(batch):
    if batch is None or batch.is_extend_in_batch:
        return False
    mode = batch.global_forward_mode
    if mode is None:
        mode = batch.forward_mode
    return mode.is_decode_or_idle()
```

`is_extend_in_batch` 是DP全局同步的“是否任一卡正在extend”，排除混合Prefill；纯Decode允许IDLE参与。每个EP rank在每层必须执行相同collective序列，包括0token rank。不能用忽略超时、跳过空rank或退回全本地分配来掩盖问题。

必须验证：单请求、3活跃5空rank、全满Decode、混合extend/decode/idle、稀疏prefill、TBO元数据存在/缺失；再真实启动默认warmup及C1/3/8/128/512请求，最后核查各rank NVFP4/TWF trace。此次修复仅改分支，未改CUDA内核。

数值用例应含全空、小batch、63/64/65边界和不均匀count；核查top8专家及权重不变、shared权重/总数守恒、输出误差及NaN/Inf。P2P交换超时明确失败，不能用零peer counts继续运行。
