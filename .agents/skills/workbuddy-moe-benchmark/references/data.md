# 固定数据与确定性采样

数据来自 https://github.com/Tencent/workbuddy-bench ，HF 数据集 `tencent/workbuddy-bench`。
历史固定版本：HF `3e8ad839685c589d83a1dd60fe2d15eabb75f27a`；参考仓库 `625b2233093ae4f23e76be28c1f341d41cc70373`。辅助脚本下载 HF 固定版本并校验其 SHA256SUMS，**不下载或宣称验证参考仓库 checkout**。换版本需新建实验。

在仓库根目录执行（Python ≥3.11，curl；prepare 阶段需安装与服务端匹配的 transformers/tokenizer 依赖）：

```bash
python .agents/skills/workbuddy-moe-benchmark/scripts/prepare_workloads.py download --out outputs/workbuddy-data
python .agents/skills/workbuddy-moe-benchmark/scripts/prepare_workloads.py prepare --out outputs/workbuddy-data --tokenizer /path/to/local/model
```

下载遵循当前网络代理设置。只解包普通文件和目录，拒绝越界路径及链接；不执行任务脚本或递归解压任务工作区。Tokenizer 仅从本地读取，允许模型自定义 tokenizer 代码，应使用已确认来源的模型目录。服务端若不是同一种 tokenizer 加载方式，先比对实际 prompt token 数，不可假定 AutoTokenizer 与服务完全等价。

固定负载：

| subset | 全部任务 | 校准任务 | 每 seed 测量请求 |
|---|---:|---:|---:|
| code | 80 | 16 | 308 |
| web | 70 | 14 | 269 |
| office | 50 | 10 | 192 |
| sec | 60 | 12 | 231 |

- 共260任务，校准52，测试208，校准和测试文本 SHA256 不重叠。
- 读取顶层 `instruction.md`；没有时解析 `task.toml`，只读取首个声明 step 的 instruction。保留原始 UTF-8/newline，不拼接后续步骤，不添加 chat template，不截断。
- 每 subset 按 `sha256("workbuddy-prefill-split-v1/{subset}/{task_id}")` 排序，前20%做校准。各池再按 task_id 排序。
- 每 seed 独立使用 `random.Random(seed)`；按 code/web/office/sec 顺序有放回采样表中数量，最后用同一 RNG shuffle。默认 seeds=8、28、2026；校准预热负载用 seed=0。
- 每 seed 共1000请求；保存 Python 版本、prompt/hash、任务ID、token数、抽样顺序和 split。重复实验复用同一 JSONL，不重新随机。
- 每个请求记录 `expected_prompt_tokens`，服务结果必须匹配。脚本默认 tokenizer `add_special_tokens=True`，历史输入长度上限16384；改变模板或上下文范围需新协议。

输出：`source.json`、`tasks.json`、`assets.json`、`workloads/calibration.jsonl`、`workloads/seed*.jsonl`。`assets.json` 包含本地绝对路径，属于运行产物，不上传 Git。跨机器时 JSONL 的 source 路径可能不同，因此额外比较有序 `(subset, task_id, prompt_sha256, expected_prompt_tokens)`，不要将路径差异误判为采样差异。

Decode pilot 只使用 seed8、首 C 条请求，不保持满载补充；这不是1000条请求在 C 并发下运行。C>1000 时不能直接取前 C 条并声称并发达标，需事先声明新的确定性扩样协议。固定输出长度、贪心、ignore EOS 用于吞吐，不等于自然任务 rollout 或解题正确率。
