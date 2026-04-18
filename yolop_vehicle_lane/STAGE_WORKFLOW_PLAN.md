# 3-Stage Workflow — Stage1 / Stage2 / Stage3

The project is organized as three disjoint stages. Run them in order.
Do not mix stage boundaries; each stage has its own configs,
notebooks, checkpoint / metrics / tb-log directories, and (for stage2
and stage3) its own `lib/` subpackage.

```
yolop_vehicle_lane/
├── lib/                         # shared primitives (models, loss, dataset base, utils)
├── stage1/
│   ├── configs/                 # YOLOPv2 best-row + focal-only + YOLOP reference YAMLs
│   └── notebooks/00..07
├── stage2/
│   ├── lib/                     # DETR-lane model, warm-start, distill, grad-balance
│   ├── configs/                 # 5 branch YAMLs (1c, 1c-warm, 3c-warm, 3c-distill, 3c-distill-gradnorm)
│   └── notebooks/00..07
└── stage3/
    ├── configs/                 # integrated placeholder + materialized YAMLs
    └── notebooks/00..07
```

## Stage roles

| stage | purpose | class taxonomy | what it changes | output |
|---|---|---|---|---|
| **stage1** | strict fair-comparison YOLOP-style baseline | `stage1_vehicle_merged` (1 class: `vehicle`) | NOTHING — reference line | `stage1/checkpoints/yolopv2_best_row/*` |
| **stage2** | independent branch experiments | `stage2_1c_vehicle_merged` or `stage2_3c_extended` | lane head (DETR-style) + optional warm-start + optional distill + optional grad-balance + optional taxonomy extension | `stage2/checkpoints/<branch_name>/*` |
| **stage3** | first controlled integration | inherited from stage2 winner | merges the winning stage2 ideas into one fine-tune | `stage3/checkpoints/integrated/*` |

## Execution order (strict)

1. **Stage1** — run `stage1/notebooks/00 → 07` in order. Produces the
   reference baseline and its artefacts.
2. **Stage2** — run `stage2/notebooks/00 → 07`. Multiple training
   notebooks (02-05) run sequentially, each training one branch to
   its own checkpoint dir. Notebook 06 aggregates. Notebook 07 profiles.
3. **Stage3** — run `stage3/notebooks/00 → 07`. `00` reads stage2's
   comparison CSV and materializes a concrete YAML; the rest fine-tune,
   compare, export, profile.

## Stage1 / notebooks

| # | file | role |
|---|---|---|
| 00 | `00_rebuild_dataset_and_lane_cache.ipynb` | BDD images + centerline lane masks + Drive tar.gz persistence |
| 01 | `01_augmentation_lab.ipynb` | visualize the real training dataloader (Mosaic + MixUp) |
| 02 | `02_train_yolopv2_vehicle_lane_baseline.ipynb` | train — `CONFIG='YOLOPv2-best-row'` default |
| 03 | `03_eval_and_backbone_ablation.ipynb` | rectangular 640×384 eval on val |
| 06 | `06_final_train_eval_export.ipynb` | ONNX + TorchScript export via `model.predict()` |
| 07 | `07_a5000_video_profile.ipynb` | A5000 latency + MFU + video overlay |

Stage1 protocol (locked): **strict YOLOP-style, `nc=1` merged-vehicle**.
motorcycle / bicycle are **not** included in stage1 training or eval.

## Stage2 / notebooks

| # | file | role |
|---|---|---|
| 00 | `00_prepare_stage2_taxonomy.ipynb` | verify 1c + 3c class protocols + stage1 ckpt presence |
| 01 | `01_inspect_branch_configs.ipynb` | dump per-branch YAML surface + warm-start/distill ckpt checks |
| 02 | `02_train_raw_branch.ipynb` | raw branch (1c, no warm-start, no distill, no grad-balance) |
| 03 | `03_train_warmstart.ipynb` | + warm-start from stage1 best-row (1c or 3c) |
| 04 | `04_train_distill.ipynb` | + detection distillation from stage1 teacher |
| 05 | `05_train_distill_gradbalance.ipynb` | + gradient balancing (uncertainty by default; `pcgrad` alternative) |
| 06 | `06_branch_comparison_and_export.ipynb` | per-branch table + winner selection |
| 07 | `07_a5000_video_profile.ipynb` | A5000 profiling per branch |

Stage2 is independent from stage1 — no notebook mutates `stage1/`
artefacts. Stage2 consumes `stage1/checkpoints/yolopv2_best_row/best.pth`
as a read-only teacher / warm-start source.

## Stage3 / notebooks

| # | file | role |
|---|---|---|
| 00 | `00_assemble_integrated_config.ipynb` | read stage2 comparison CSV → write `integrated_from_<winner>.yaml` |
| 01 | `01_verify_inherited_assets.ipynb` | confirm chains + do a dry warm-start load |
| 02 | `02_train_integrated.ipynb` | fine-tune the integrated configuration |
| 03 | `03_compare_vs_stage1_stage2.ipynb` | final 3-row comparison table |
| 06 | `06_final_integrated_export.ipynb` | ONNX / TorchScript via `model.predict()` |
| 07 | `07_a5000_video_profile.ipynb` | A5000 profile of the integrated model |

Stage3 is the **first** integration stage and does not re-explore
architectural alternatives. It consumes evidence produced by stage2
notebook 06.

## Drive persistence conventions

- Datasets live under `EcoCAR/datasets/` (shared across pipelines).
- Stage-specific artefacts live under
  `yolop_vehicle_lane/<stage>/{checkpoints,metrics,tb_logs}/<run_name>/`.
- No notebook writes many small files to Drive directly — see
  `claude_code_dataset_debug_handoff.md` for the compression rule.

## Fair-comparison rule (stage1 vs stage2)

- Stage1 is the scientific reference: strict YOLOP-style merged
  vehicle class, identical protocol for reproduction.
- Stage2's **1c** branch (`detrlane_vehicle1c*.yaml`) uses the same
  class protocol as stage1 and is directly comparable.
- Stage2's **3c** branches add `motorcycle` and `bicycle` as a
  deliberate taxonomy extension — they are **not** a drop-in
  comparison against stage1's 1c protocol. Report 3c results as an
  added-capability metric alongside the 1c fair-comparison number.
