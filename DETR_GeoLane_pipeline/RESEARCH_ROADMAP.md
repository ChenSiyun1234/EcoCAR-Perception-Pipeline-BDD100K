# Joint Conflict Research Roadmap

This revision focuses on the core problem: an architecturally more advanced joint model still underperforms YOLOP because the optimization path is worse.

## What changed

### 1. Task-specific adapters
Shared FPN features are still learned jointly, but each task now receives a small residual adapter stack before entering its encoder. This follows the modern idea of keeping a strong shared backbone while adding cheap task-specific adaptation instead of forcing both heads to consume identical features.

### 2. Staged training
The training schedule now has three stages:
- Stage A: detection-only warmup (`det_only_epochs`)
- Stage B: weak lane coupling (`lane_task_warmup_weight`)
- Stage C: full joint training

This matches the practical lesson from recent multi-task work: do not let the harder structured task dominate before the object branch and shared features are stable.

### 3. Delayed gated cross-attention
Cross-branch attention is no longer fully active from epoch 0. It starts later (`cross_attn_start_epoch`) and ramps in gradually. This reduces early branch pollution.

### 4. Preflight research notebook
`notebooks/03_research_preflight.ipynb` measures:
- lane supervision coverage
- one-batch gradient conflict between detection and lane losses
- feature gap between the two task branches after adapters

## How to use
1. Run notebook 03 first.
2. Save the printed values for:
   - `grad_cosine`
   - `det_grad_norm`
   - `lane_grad_norm`
   - `p3/p4/p5 cosine`
3. Run notebook 00.
4. Report epoch 1, 3, 5, 10 metrics back for comparison.

## How to interpret the preflight signals
- `grad_cosine < 0`: the tasks are fighting each other.
- lane gradient norm >> detection gradient norm: lane branch is dominating too early.
- feature cosine extremely close to 1.0 at all scales: branches are still nearly identical and may need stronger task adaptation.
- feature cosine too low very early: branches may be over-decoupled.

## Experiment ladder
1. Baseline with adapters + staged training.
2. Same but cross-attn disabled.
3. Same but stronger backbone.
4. Same but detection-only warm start checkpoint.
5. Same but improved lane query / topology / temporal module.
