# Contributing to EvenMoE

We welcome new model integrations, load-balancing algorithms, framework integrations, correctness fixes, and documentation.
Contributions should reuse existing interfaces to make each extension testable, composable, and maintainable.

## 1. How the project should grow

EvenMoE is a framework-independent library for MoE load balancing:

- **L1 placement** decides which physical expert slots hold each logical expert's weights.
- **L2 routing** selects a physical replica of the chosen logical expert for each token and, where the framework supports it, a destination rank for the shared expert.
- **Framework integration** supplies model expert information and TopK results to EvenMoE, exposes existing framework communication operations for aggregating load, and notifies EvenMoE when the expert placement changes.

DeepSeek-V3 is the end-to-end example currently presented in the README. New model integrations should reuse these general capabilities.
Supporting a model, implementing an algorithm, and running it in a framework are separate accomplishments that need separate explanations and validation.
An algorithm's presence in the source tree does not mean it supports every model, framework, execution stage, or device.

Our goal is to **reuse existing framework integration when adding an algorithm, and reuse existing algorithms when adding a model**.
If the existing interfaces cannot express a new capability, explain the gap and use case before discussing an extension to the public interfaces.

## 2. Choose a contribution path

| Contribution | Main implementation location | Minimum deliverables |
|---|---|---|
| Support a new model | Model and MoE integration in the framework; reuse existing EvenMoE adapters and algorithms | Connect expert information, TopK routing, placement result handoff, and placement commit notifications; reuse existing integration where possible |
| Add an L1 algorithm | `policies/l1/` and registration in the unified entry point | A `PlacementPlanner` implementation, placement mapping validity tests, and an algorithm description |
| Add an L2 algorithm | `policies/l2/`, pipeline configuration, and the construction entry point | A `TokenRouter` implementation, capability declarations, and semantics and composition tests |
| Add a framework or integration capability | `adapters/<framework>/` and the framework repository | Data conversion, lifecycle integration, framework versions, and validation across both repositories |
| Fix a bug, documentation, or tests | The modules directly related to the issue | A reproducible issue, a focused fix, and relevant validation |

## 3. Start with these entry points

| File or directory | What to look for |
|---|---|
| [README.md](README.md) | Project scope, current usage, installation, and reported validation results |
| [core/types.py](src/evenmoe/core/types.py) | The actual fields and semantics of requests, plans, snapshots, and decisions |
| [core/l1.py](src/evenmoe/core/l1.py), [core/l2.py](src/evenmoe/core/l2.py) | L1/L2 protocols and the collective abstraction |
| [core/orchestrator.py](src/evenmoe/core/orchestrator.py) | `MoELoadBalancer` construction, algorithm calls, and placement notifications |
| [core/routing_pipeline.py](src/evenmoe/core/routing_pipeline.py) | Algorithm expressions, composition rules, capabilities, and applicability |
| [adapters/sglang/](src/evenmoe/adapters/sglang/), [adapters/vllm/](src/evenmoe/adapters/vllm/) | Framework boundaries and data conversion |
| [tests/](tests/), [CI configuration](.github/workflows/ci.yml) | Existing behavioral contracts and automated checks |

## 4. Design constraints

### 4.1 Module responsibilities: where code belongs

The five areas below—**Framework, Adapter, Core, Policy, and Kernels**—describe responsibilities, code locations, examples, and work that belongs elsewhere.
Paths below are relative to `src/evenmoe/`. "Framework code" means code in the SGLang, vLLM, or other framework repository.

**Framework: decide when to call EvenMoE and apply its results during model execution.**

- **Responsibilities**: Run the model, provide load data, decide when to rebalance experts, manage communication groups, move expert weights, and update the expert mappings used for execution.
- **Location**: The framework's model execution, MoE layers, and load-balancing integration code. "Glue" refers to the code that calls EvenMoE from these locations.
- **Example**: The framework decides to rebalance and calls EvenMoE for a new placement. After moving weights and completing the required synchronization, it updates the mappings and notifies EvenMoE to refresh algorithm state that depends on the placement.
- **Keep elsewhere**: Do not duplicate an algorithm's solver or routing logic in the framework for each new integration. These implementations belong in EvenMoE policies.

**Adapter: connect existing framework data and operations to EvenMoE interfaces.**

- **Responsibilities**: Read model information and current expert mappings from the framework to construct EvenMoE requests and snapshots; convert EvenMoE results into framework formats; wrap existing framework communication operations behind interfaces EvenMoE can call.
- **Location**: The corresponding framework directory, such as `adapters/sglang/` or `adapters/vllm/`.
- **Example**: SGLang's `to_placement_snapshot()` converts the currently committed mappings into an EvenMoE `PlacementSnapshot`; `to_sglang_maps()` converts a candidate plan into SGLang mapping formats.
- **Division of responsibility**: The adapter handles data conversion and interface integration. Policies make balancing decisions. The adapter reads the framework's committed expert mappings and converts them into EvenMoE's format.

**Core: provide public interfaces and coordinate algorithm calls.**

- **Responsibilities**: Define public types such as requests, plans, snapshots, and decisions; provide the `MoELoadBalancer` entry point; invoke configured policies; check policy composition and runtime requirements.
- **Location**: Files such as `core/types.py`, `core/orchestrator.py`, and `core/routing_pipeline.py`.
- **Example**: Parse the routing algorithm configuration, select the corresponding policies, and aggregate load through the supplied communication interface when needed.
- **Keep elsewhere**: Do not read SGLang's `RuntimeContext` or vLLM internal objects in Core, or add model-specific execution branches. Adapters convert framework fields; policies contain algorithm details.

**Policy: implement algorithms for expert placement and replica selection.**

- **Responsibilities**: L1 uses load and device topology to return a `PlacementPlan`; L2 uses the request and committed placement to return a token-routing `RoutingDecision`.
- **Location**: L1 algorithms in `policies/l1/`; L2 algorithms in `policies/l2/`.
- **Example**: A new replica-selection algorithm takes logical expert IDs and existing replica mappings, then chooses the physical replica for each token.
- **Keep elsewhere**: Do not modify the framework's active expert mappings directly, or decide when the framework pauses execution or commits a new placement. Returning a plan does not mean its weights are already in place.

**Kernels: implement the computation needed by policies.**

- **Responsibilities**: Implement low-level counting, solving, remapping, and related operations for policies or shared computational entry points.
- **Location**: The corresponding operation or algorithm directory under `kernels/`.
- **Example**: Add a faster implementation of an existing counting operation while preserving its inputs, outputs, and meaning.
- **Keep elsewhere**: Do not read framework context, decide when to rebalance, or update the framework's active expert placement inside a computational implementation.

**Use the following division of responsibility for operations spanning modules.**

- **Communication**: The adapter wraps framework communication operations needed by L2 behind a common interface that EvenMoE can call, for example to aggregate load across GPUs.
- **Weight movement and placement activation**: The framework determines when to move weights and handles synchronization and the final placement switch. The SGLang adapter's `commit_placement()` notifies EvenMoE after the framework has moved weights and switched the placement; it does not move weights on the framework's behalf.

For example, an L1 update proceeds in this order:

1. The framework decides to rebalance and prepares load data; the adapter converts it into a `PlacementRequest`.
2. Core calls the L1 policy to obtain a candidate `PlacementPlan`. The model still uses the old placement at this point.
3. The adapter converts the plan's format; the framework moves weights, synchronizes, and switches mappings.
4. The adapter constructs a snapshot from the committed placement and notifies EvenMoE to refresh algorithm state. Subsequent L2 routing uses this placement.

**Determine the scope of changes from the contribution type:**

| Change | Usual location | When other layers need changes |
|---|---|---|
| Add an L1/L2 algorithm | `policies/` and the corresponding registration, configuration, and tests | Add a kernel when a new computational operation is needed. Explain and extend Core only when the public interfaces cannot express the algorithm's inputs or outputs. |
| Support a new model in an existing framework | Framework model integration, configuration, examples, and tests | Extend the adapter if it cannot express the model information. A new policy is unnecessary when existing algorithms can be reused. |
| Integrate a new framework | `adapters/<framework>/` and call sites in the new framework | Reuse existing Core, policies, and kernels first. Explain any missing shared capabilities separately. |
| Optimize a computational operation | The corresponding `kernels/` implementation and validation | If the input/output interface changes, update the calling policy or shared computational entry point. |

## 5. Add model support: DeepSeek-V3.2 as an example

Check three integration points when adding a model: **expert information, TopK routing, and placement result handoff and commit notifications**. Reuse existing integration where it supports the model; otherwise, update the relevant SGLang integration or EvenMoE adapter.
This example uses the [SGLang branch with EvenMoE integration](https://github.com/xutizhou/sglang-moe-lb-waterfill-dev/tree/e4189e8fac9c53b6698c23c1411d148350ae0137) and the [EvenMoE implementation](https://github.com/ai-dynamo/evenmoe/tree/main). SGLang paths are relative to `python/sglang/srt/`; EvenMoE paths are relative to `src/evenmoe/`.

### 5.1 Expert information: can the model's expert layout reach EvenMoE?

**Where to check:**

- The SGLang model's `get_model_config_for_expert_location()`: does it provide the correct layer count, logical expert count, and expert group count?
- SGLang's `model_executor/model_runner.py`: does it populate `context.resources.mlb_model_info` with the expert information?
- EvenMoE's `adapters/sglang/placement.py`: can it construct placement requests and snapshots from this information and the framework's per-layer mappings?

**What to change:** If the model lacks the expert-information interface, implement it in the SGLang model class. Map differently named configuration fields in that interface. Changes to values such as the expert count usually do not require EvenMoE changes. If expert counts differ by layer, the current shared model information cannot express them; extend both the per-layer information in SGLang context and the EvenMoE adapter's per-layer reads.

**Why V3.2 needs no change:** `DeepseekV3ForCausalLM` and `DeepseekV32ForCausalLM` both inherit from `DeepseekV2ForCausalLM` and already share this interface. Their routed expert and group counts are also the same, so the expert-information and placement conversions can be reused.

### 5.2 TopK routing: does the model use the existing EvenMoE call path?

**Where to check:**

- SGLang's `layers/moe/topk.py::_apply_moe_load_balancer()` and `eplb/moe_load_balancer_glue.py::route_topk_with_mlb()`: after the model produces logical TopK results, does it call EvenMoE through this path and pass the result to dispatch?
- EvenMoE's `adapters/sglang/routing.py`: can `to_routing_request()` and `to_sglang_routing_output()` convert the model's TopK results and correctly pass expert IDs, weights, and scaling factors?

**What to change:** If the model uses a different TopK/dispatch path, add the `route_topk_with_mlb()` call in that SGLang path. If the result format differs, first convert it to the helper's `StandardTopKOutput` format. Extend the adapter only if it cannot express the required data. Preserve the model's expert selection and weight normalization.

**Why V3.2 needs no change:** It reuses the existing MoE implementation and shared TopK path, which already calls EvenMoE, with compatible input and output formats. No additional integration or model-specific L2 policy is needed.

### 5.3 Placement handoff: receive EvenMoE results and notify EvenMoE when they take effect

**Where to check:**

- EvenMoE's `adapters/sglang/placement.py::to_sglang_maps()` and its SGLang callers: do they convert the L1 `PlacementPlan` into framework mappings and pass them to SGLang's existing placement update flow?
- The same adapter's `commit_placement()` and its SGLang callers: do they notify EvenMoE after the placement switch and read the committed placement through `to_placement_snapshot()`?

**What to change:** If the model uses a different placement update path, connect the placement result conversion and handoff, along with the notification after the switch. Reuse the shared path when it already covers the model.

**Why V3.2 needs no change:** It uses the same placement representation and SGLang update flow. Both receiving EvenMoE's placement results and notifying EvenMoE after updates are already connected.

Once these three integration points are connected, reuse applicable L1/L2 algorithms. To use Waterfill, also check that SGLang can execute and combine shared-expert work across ranks according to EvenMoE's `shared_expert_rank`. The presence of a shared expert in the model does not itself provide this integration.

## 6. Add an L1 or L2 algorithm

### 6.1 Choose the directory and required interfaces

Paths below are relative to `src/evenmoe/`.

| Algorithm responsibility | New file location | Required implementation | Return value |
|---|---|---|---|
| L1: choose physical slots for experts | `policies/l1/<algorithm>.py` | `plan(request: PlacementRequest)` | `PlacementPlan` |
| L2: choose an expert replica for each token | `policies/l2/<algorithm>.py` | `phase = RoutingPhase.PHYSICAL_ROUTING` and `route_tokens(request, policy)` | `RoutingDecision` with physical expert IDs and weights |
| L2: choose a rank for shared-expert execution | `policies/l2/<algorithm>.py` | `phase = RoutingPhase.SHARED_EXPERT` and `route_tokens(request, policy)` | `RoutingDecision` preserving physical expert IDs and weights, with `shared_expert_rank` added |

The interfaces are `PlacementPlanner` in [core/l1.py](src/evenmoe/core/l1.py) and `TokenRouter` in [core/l2.py](src/evenmoe/core/l2.py).
For a new L1 algorithm, explicitly inherit from `PlacementPlanner` and implement `plan()`.
For a new L2 algorithm, explicitly inherit from `TokenRouter`, declare `phase`, and implement `route_tokens()`.
Larger algorithms can be split into files in a directory with the algorithm's name.

The following examples use the existing EPLB and Static implementations to explain the interfaces and call paths. Add new algorithms under EvenMoE's `policies/`; once name parsing and instance creation are connected, the framework enables them by name.

### 6.2 Add L1: implement plan and connect algorithm selection

**Step 1: In `policies/l1/<algorithm>.py`, inherit from `PlacementPlanner` and implement `plan()`.**

See [EPLBPlacementPlanner](src/evenmoe/policies/l1/eplb/planner.py):

```python
class EPLBPlacementPlanner(PlacementPlanner):
    def plan(self, request: PlacementRequest) -> PlacementPlan:
        ...
```

This shows the interface shape; the link contains the full implementation. `plan()` reads load from `request.stats.logical_count` and physical slot count, slots per rank, and device grouping from `request.topology` to compute a placement.
Implement the new placement logic in this method and return these fields:

| `PlacementPlan` field | Contents |
|---|---|
| `physical_to_logical_map` | The logical expert assigned to each physical slot |
| `logical_to_all_physical_map` | Candidate physical slots for each logical expert; use `-1` for padding |
| `logical_to_physical_count` | The number of valid physical replicas for each logical expert |

**Step 2: Follow Section 6.4 to add the algorithm to EvenMoE's default planner mapping.**

The framework selects a built-in algorithm through `PlacementRequest.policy`.
`MoELoadBalancer.plan_placement()` finds the corresponding planner and calls its `plan()` method.

**Step 3: Call the unified entry point. This example uses the existing `deepseek` algorithm.**

```python
from evenmoe import (
    MoELoadBalancer, PlacementRequest, PlacementStats, PlacementTopology,
)

mlb = MoELoadBalancer()
request = PlacementRequest(
    stats=PlacementStats(logical_count=[[20, 5]]),
    topology=PlacementTopology(
        num_physical_experts=4,
        num_local_physical_experts=2,
        num_groups=1,
        num_nodes=1,
    ),
    policy="deepseek",
)
plan = mlb.plan_placement(request)
print(plan.physical_to_logical_map)
```

After Step 2, change `policy` to the new algorithm's name to use the same call path.
The result is a candidate placement. The framework activates it after loading or moving the weights.

### 6.3 Add L2: implement route_tokens and connect from_algorithm

**Step 1: In `policies/l2/<algorithm>.py`, inherit from `TokenRouter`, declare `phase`, and implement `route_tokens()`.**

The existing [StaticL2Router](src/evenmoe/policies/l2/replica.py) uses this interface:

```python
class StaticL2Router(TokenRouter):
    phase = RoutingPhase.PHYSICAL_ROUTING

    def route_tokens(
        self, request: RoutingRequest, policy: RoutingPolicyConfig,
    ) -> RoutingDecision:
        ...
```

This shows the interface shape. Static reads `request.logical_topk_ids`, looks up physical replicas through `request.placement.default_physical_for_logical`, puts the result in `RoutingDecision.routed_physical_topk_ids`, and preserves `request.topk_weights`.
Implement a new physical-replica selection algorithm through the same interface.

**Step 2: Follow Section 6.4 to connect the algorithm name, execution phase, capability declarations, and instance creation in `from_algorithm()`.**

The framework constructs EvenMoE with `MoELoadBalancer.from_algorithm("algorithm_name", ...)` during initialization.
Each `RoutingRequest` supplies the current batch's data and uses the algorithm selected at initialization.

**Step 3: Call the unified entry point. This example uses the existing `static` algorithm.**

```python
import torch
from evenmoe import MoELoadBalancer, PlacementSnapshot, RoutingRequest

mlb = MoELoadBalancer.from_algorithm(
    "static",
    ep_size=2,
    source_rank=0,
    experts_per_rank=2,
)
# This example assumes the matching weights are already loaded.
# In a framework integration, the adapter reads the current placement.
snapshot = PlacementSnapshot(
    layer_id=0, num_logical_experts=2, num_physical_experts=4,
    ep_size=2, num_local_physical_experts=2,
    physical_to_logical_map=torch.tensor([1, 0, 1, 0]),
    logical_to_physical_candidates=torch.tensor([[1, 3], [0, 2]]),
    logical_to_physical_count=torch.tensor([2, 2]),
    default_physical_for_logical=torch.tensor([1, 0]),
)
request = RoutingRequest(
    layer_id=0,
    logical_topk_ids=torch.tensor([[0, 1], [-1, 0]]),
    topk_weights=torch.tensor([[0.7, 0.3], [0.0, 1.0]]),
    placement=snapshot,
)
decision = mlb.route_tokens(request)
assert decision.routed_physical_topk_ids.tolist() == [[1, 0], [-1, 1]]
assert torch.equal(decision.topk_weights, request.topk_weights)
```

After Step 2, replace the name passed to `from_algorithm()` with the new algorithm's name.

**For a shared-expert algorithm, keep the method signature and change these three things:**

1. Set `phase` to `RoutingPhase.SHARED_EXPERT`.
2. When SGLang calls `mlb.route_tokens(request)`, it still supplies `request.logical_topk_ids`: the logical expert IDs selected by the model. Inside that call, EvenMoE first determines physical IDs through the physical-replica routing policy, or the default mapping if none is configured, and prepares the routed-expert load for each rank. It then calls the shared-expert algorithm's `route_tokens()`. Read these prepared results from `request.routed_physical_topk_ids` and `request.routed_rank_load` to choose the shared-expert rank for each token. SGLang does not need to convert logical IDs to physical IDs before calling EvenMoE.
3. Preserve the existing physical IDs and weights in `RoutingDecision`, and put each token's destination rank in `shared_expert_rank`.

See [replica.py](src/evenmoe/policies/l2/replica.py) and [waterfill.py](src/evenmoe/policies/l2/waterfill.py) for existing implementations.

### 6.4 Connect the algorithm to the repository's configuration entry points

Complete the relevant integration below so the framework can select the new L1 or L2 algorithm by name:

| Goal | File and location | What to add |
|---|---|---|
| Select L1 by name | `core/orchestrator.py` → `MoELoadBalancer.__init__()` | Import the new planner and add its name-to-instance mapping to the default `placement_planners`; use this name in `PlacementRequest.policy` |
| Parse the L2 name and runtime requirements | `core/routing_pipeline.py` → `RoutingPipeline._from_string()`; update `is_applicable()` as needed | Recognize the new name, assign it to the correct execution phase, and declare the statistics, state, and runtime conditions it requires |
| Create the implementation from the L2 name | `core/orchestrator.py` → `MoELoadBalancer.from_algorithm()` | Import the new router; instantiate it when its name is selected and add it to `routers`, so the factory returns a configured EvenMoE instance |

## 7. Compare against a baseline and report performance

New models, policies, and extensions that affect execution should provide a **baseline versus candidate comparison on the same workload**.
The report should explain where the change helps, the size of the benefit, its costs, and where it does not help.

### 7.1 Choose the right baseline

| Change | Baseline | What to hold fixed |
|---|---|---|
| Integrate a new model | The same checkpoint on the framework's native execution path, compared with EvenMoE enabled | Model, precision, hardware, and requests. Differences between V3 and V3.2 do not establish an EvenMoE benefit. |
| Add an L1 policy | An existing applicable L1 algorithm; optionally also report results without rebalancing | Load recordings, initial placement, replica budget, and L2. Each algorithm produces its own final placement. |
| Add an L2 policy | An existing applicable router on the same placement, such as `static`, `dynamic`, or another target algorithm | Per-layer mappings and replica counts; change only L2 |
| Optimize another execution path | The previous implementation of the same functionality | Inputs, output semantics, and runtime configuration |

Name the specific baseline algorithm and configuration; "default settings" is not enough.
If the candidate requires more replicas or memory, list the resource change separately. To compare the algorithms themselves, also report results under the same resource budget.

### 7.2 Hold key experimental conditions fixed

- **Same conditions**: Use the same model, hardware, parallel configuration, and request data. Record code versions and key runtime parameters, and explain configuration differences beyond the change under evaluation.
- **Fair comparison**: Apply the same tuning, warmup, and measurement rules to the baseline and candidate. Keep calibration and evaluation data separate.
- **Correctness first**: Compare model outputs or task accuracy and record request failures. Compare performance after correctness passes, and report both improvements and regressions.

### 7.3 Repeat A/B measurements to separate gains from variation

1. **Fix the measurement plan first**: Record warmup rounds, requests per round, measurement rounds, and failure-handling rules. Both sides should complete compilation and graph capture, then warm up under the same rules. Measure cold start separately if startup cost is the optimization target.
2. **Run at least three independent A/B pairs**: A is the baseline and B is the candidate. Start each run independently or reset it to the agreed initial state, alternating A→B and B→A order across pairs. Repeating many requests in a single process is not a substitute for independent repetitions.
3. **Save raw results for every round**: Calculate metrics for each independent run, then summarize the median and variation across runs. Request-level p95/p99 describes tail latency; variation across independent runs describes measurement stability. Report them separately.
4. **Do not declare a win from a small difference alone**: At minimum, show the difference for each pair. A claim of a stable benefit should include a confidence interval based on independent paired A/B differences or a predefined decision rule. If the interval crosses zero, the sample is insufficient, or the difference is comparable to noise, report the benefit as uncertain. If more runs are needed, decide the new plan in advance and retain all results; do not stop at the first significant result.

Retain timeout, OOM, and invalid-output records. If a confirmed environment failure requires a rerun, record the reason and rerun the corresponding A/B pair. Do not select only the candidate's fastest run.

### 7.4 Report throughput, latency, and balancing costs together

| Question | Data to report |
|---|---|
| Actual processing rate | Successful/total requests, failure count, total duration, and output tokens/s. Label input or total tokens/s separately if reported. |
| User waiting time | TTFT, TPOT or ITL, and request end-to-end latency at p50/p95. Add p99 and sample counts when extreme tail latency matters. |
| Serving capacity | Requests or tokens completed under the same latency target. Explain the tradeoff if throughput rises while tail latency worsens. Use the same concurrency/arrival-rate test points, not just each configuration's peak. |
| Whether balancing works | Actual per-rank load over the same measurement window, such as `max(rank_load) / mean(rank_load)`. This helps explain the benefit but does not replace end-to-end metrics. |
| Added overhead | Time spent solving, collecting load, running collectives, moving weights, updating placement, and rebuilding graphs, plus additional GPU memory |

TTFT is time to first token; TPOT is average time per output token after the first; ITL is the interval between successive output tokens.
A one-output-token experiment has no TPOT and cannot establish a benefit for sustained decode.
See [SGLang bench_serving](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/bench_serving.mdx) for serving metrics and request-level records. Save the actual script version and arguments used.

**L1 results must include more than steady-state gains under the new placement.** Report the update interval, the total cost of an update, and a complete evaluation window that includes updates.
If the placement is generated offline before serving starts, report preparation costs separately and state how long the placement is assumed to be reused.
Online rebalancing statistics, weight movement, and synchronization costs must be included in end-to-end results.

**L2 results must include more than routing kernel time.** Counting, solving, and communication added to each batch are part of its cost.
When operations overlap, use the actual critical path and end-to-end time; do not simply sum overlapping component durations.

### 7.5 Use this format for reproducible results

Start with a one-sentence experiment definition: **"On which model, hardware, requests, and resource budget, what changed, and which metric is the primary outcome?"**
Then report absolute values and changes, not percentages alone:

| Scenario / input and output lengths / load settings | Metric | Baseline | Candidate | Change | Independent A/B pairs and variation / interval |
|---|---|---|---|---|---|
| `<fill in>` | Output tokens/s ↑ | `<measured>` | `<measured>` | `<calculated>` | `<measured>` |
| Same | TTFT p95, ms ↓ | `<measured>` | `<measured>` | `<calculated>` | `<measured>` |
| Same | TPOT p95, ms ↓ | `<measured or N/A>` | `<measured or N/A>` | `<calculated>` | `<measured>` |
| Same | End-to-end latency p95, ms ↓ | `<measured>` | `<measured>` | `<calculated>` | `<measured>` |
| Same | Successful / total requests | `<measured>` | `<measured>` | — | — |
| Same | Update cost / update interval, additional GPU memory | `<measured>` | `<measured>` | — | — |

Use these formulas consistently; negative values indicate regressions:

```text
Throughput improvement (%) = (candidate / baseline - 1) × 100
Latency reduction (%) = (1 - candidate / baseline) × 100
```

Attach three kinds of supporting material:

- **Reproduction entry points**: Complete launch and benchmark commands, configurations, and versions for both sides; how request data and placements were generated, with checksums.
- **Raw records**: A/B order, timestamps, results for every round, request details, correctness results, and failure logs. Generate summary tables from these records with a script.
- **Conclusions and limitations**: Scenarios with stable gains, no benefit, or regressions, and additional resource use. State the scope of reduced-layer models, offline traces, or individual-operation results; do not extrapolate them to full-model serving performance.

The following is an example attachment layout, not a required directory structure or private tool:

```text
performance-report/
  report.md          # Experiment definition, comparison results, and conclusions
  configs/           # Baseline/candidate configuration and complete commands
  manifest.json      # Code, model, data, and placement versions and checksums
  raw/               # Raw results and logs for every pair and round
  summarize.py       # Generate report summaries from raw/
```

## 8. PR completion criteria

Keep a PR focused on one explainable behavior change. Contributors should describe the implementation's scope; maintainers review boundaries and evidence.
If companion framework changes are required, describe the dependencies and a reproducible combination of both repositories. Merging EvenMoE code does not by itself complete framework integration.

You can use this PR outline:

```markdown
## Problem and result
What practical problem does this solve? What observable behavior changes?

## Scope and design
Contribution type: model / L1 / L2 / policy composition / framework integration / fix / documentation.
Which modules are involved? Does this change public semantics, defaults, or dependencies?
Which models, frameworks, execution stages, and hardware does it support?

## Validation
- Completed: commands, environment, and results.
- Not run or skipped: reasons and missing evidence.
- Performance claims: baseline, reproduction steps, and recorded results; write "not applicable" if none.

## Compatibility and limitations
Are existing uses affected? Which combinations remain unsupported?
List companion framework PRs and validated commits when required.

## Documentation and sources
Where are usage, capability limits, and support information updated?
Where does ported or adapted code come from? Are the relevant licenses and notices preserved?
```

Before submitting, check the items relevant to your change:

- [ ] **Integration and execution**: The new model or algorithm can be enabled through configuration and has completed end-to-end execution on the target model and framework.
- [ ] **Correctness**: Provide output-error or task-accuracy results against the baseline using the same model and inputs. For a bug fix, confirm that the original problem no longer occurs.
- [ ] **Performance**: When performance changes, report baseline and candidate throughput, latency, and additional overhead under the same conditions, including regressions.
- [ ] **Usage and reproduction**: Provide launch configuration, test commands, and results, and state the validated models, frameworks, and runtime scope.

#### Signing Off Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```
