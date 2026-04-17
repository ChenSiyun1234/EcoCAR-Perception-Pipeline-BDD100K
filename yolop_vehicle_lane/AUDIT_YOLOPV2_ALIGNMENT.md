# YOLOPv2 Alignment Audit — `yolop_vehicle_lane` (phase 1)

**Scope.** Compare the repaired `yolop_vehicle_lane` baseline against
YOLOPv2 along the dimensions listed by the user. Only sanctioned task
deviation: drivable-area head removed. Everything else must track
either YOLOP upstream code (`external_repos/YOLOP/`) or the YOLOPv2
paper / public release (`external_repos/YOLOPv2/`, arXiv 2208.11434).

**Verdict key.** `MATCH` = byte-level or semantically identical.
`PARTIAL` = shape right, detail drifts. `MISMATCH` = functionally wrong
vs upstream evidence.

| # | dimension | verdict | current state | upstream evidence | repair action |
|---|---|---|---|---|---|
| 1 | repo layout | PARTIAL | `lib/models/yolop_baseline.py`, `yolopv2_baseline.py`, stage-2 `yolopv2_detrlane.py`, `lane_set_*` | n/a | quarantine stage-2 under `stage2/`; phase-1 has exactly two models |
| 2 | model definition style | PARTIAL | YOLOP `MCnet` block-config skeleton + freehand ELAN rewrite | `YOLOP/lib/models/YOLOP.py::MCnet_0` | edit YOLOP MCnet_0 in place: delete DA head (layers 25-33), swap CSP→ELAN with groups, keep all other indices |
| 3 | backbone | PARTIAL | ELAN blocks, **no group conv** | YOLOPv2 §3 "more efficient ELAN structures" + YOLOv7 E-ELAN | keep ELAN block, add `groups` argument (default=1, set >1 in block_cfg) — mark `[INFERRED]` because YOLOPv2 source is not public |
| 4 | neck SPP | **MISMATCH** | `SPPCSPC` with CSP wrap | YOLOPv2 paper explicitly keeps **SPP**, not SPPCSPC | revert to YOLOP's `SPP(k=(5,9,13))` |
| 5 | FPN / PAN | MATCH | Upsample + Concat FPN and down-Conv + Concat PAN | `YOLOP/lib/models/YOLOP.py::MCnet_0` layers 10-23 | no change |
| 6 | detection head | MATCH | `Detect(nc=5, anchors=[3 scales], ch=[128,256,512])` | `YOLOP/lib/models/YOLOP.py::MCnet_0` layer 24 | no change except `nc` 1→5 (vehicle classes) |
| 7 | lane head — where it taps | **MISMATCH** | current taps from PAN output (layer 21 ≈ stride 8) via 3-stage `DeconvBlock` | YOLOPv2 paper: lane branch uses the **deeper** post-SPP features + deconvolution | tap from layer 16 (encoder output post-FPN, stride 8) — the exact tap point YOLOP uses — and replace `Upsample + Conv` with `ConvTranspose2d` stride-2 stages |
| 8 | lane head — output shape | **MISMATCH** | 2-ch softmax at **full input resolution** | YOLOPv2 `demo.py::lane_line_mask` expects `ll` at half-input resolution, **1-ch sigmoid**, rounded in post-processing | output at H/2×W/2, 1 channel, sigmoid-activated in `forward` |
| 9 | lane preprocessing | **MISMATCH** | each BDD poly2d rendered individually at a single width (default 8) | YOLOPv2 paper: draw **centerline** between the two annotated lines of each lane; **width 8 train / width 2 test** | add dual-width support (`train_thickness=8`, `test_thickness=2`); centerline pairing is `[INFERRED]` — BDD pairing heuristic documented |
| 10 | Mosaic / MixUp | PARTIAL | `augmentations.py::load_mosaic / mixup` exist, gated by `cfg.DATASET.MOSAIC / MIXUP`; YAML has them ON | YOLOPv2 paper §3 "BoF: Mosaic + MixUp" | verify gate reaches the real dataloader. Current `BddDataset.__getitem__` already branches on `use_mosaic` — OK, keep |
| 11 | detection loss | PARTIAL | YOLOP CIoU + BCEcls + BCEobj, focal wrapping supported via `cfg.LOSS.FL_GAMMA` but **currently 0** | YOLOPv2 paper: focal loss on cls + obj | set `LOSS.FL_GAMMA = 1.5` in the YOLOPv2 YAML |
| 12 | lane segmentation loss | **MISMATCH** | BCE + soft-IoU | YOLOPv2 paper: **focal for lane seg**, plus a hybrid focal+dice variant | extend `get_loss` so `BCEseg` is optionally wrapped in `FocalLoss` (new knob `LOSS.LL_FL_GAMMA`), add optional `LL_DICE_GAIN` for the hybrid variant |
| 13 | optimizer | **MISMATCH** | Adam | YOLOPv2 / YOLOP `get_optimizer`: SGD with momentum | flip YAML `TRAIN.OPTIMIZER=sgd` |
| 14 | LR / momentum / wd | **MISMATCH** | `LR0=0.001`, `WD=0.0005`, `END_EPOCH=100` | YOLOPv2 paper: `LR0=0.01`, `WD=0.005`, 300 epochs, warmup=3 | update YAML |
| 15 | image size train / test | **MISMATCH** | both 640×640 | YOLOPv2: train 640×640, **test 640×384** | add `TEST.IMAGE_SIZE=[640,384]` and honor it in validation letterbox |
| 16 | anchors / autoanchor | MATCH | YOLOP's BDD-kmeans anchors, `NEED_AUTOANCHOR=False` | `YOLOP/lib/models/YOLOP.py::MCnet_0` anchors | no change |
| 17 | eval / postprocess | PARTIAL | `lib/core/function.py::validate` computes lane IoU + detection mAP with padding crop | YOLOPv2 `demo.py::lane_line_mask` rounds sigmoid; drivable-area eval removed | make sure lane IoU path matches the new 1-ch sigmoid output; no other change |
| 18 | notebook flow | PARTIAL | 00/01/02/03 active; 05 stage-2 DETR-lane; 04/06 removed earlier | n/a | 02 must call the repaired YOLOPv2 baseline + new YAML; 05 stays stage-2-only |
| 19 | Drive persistence | MATCH | notebook00 writes `EcoCAR/datasets/*.tar.gz`; resolver accepts masks-only archive | n/a | no change |
| 20 | profiling GPU target | MATCH | notebook 07 uses A5000 | user spec | no change |

## Evidence index

- `external_repos/YOLOP/lib/models/YOLOP.py` — `MCnet_0` block config: Focus / BottleneckCSP / SPP backbone, FPN+PAN neck, `Detect(nc=1, …)`, progressive-upsample DA and LL decoders (lines 61-108).
- `external_repos/YOLOP/lib/core/loss.py` — `MultiHeadLoss` wraps `BCEcls/BCEobj/BCEseg`; focal is applied **only** to cls/obj when `FL_GAMMA > 0` (line 191-193); lane seg stays plain BCE. Detection uses CIoU; lane also has a soft-IoU term.
- `external_repos/YOLOP/lib/utils/utils.py::get_optimizer` — SGD or Adam selected by `TRAIN.OPTIMIZER`; both read `LR0`, `MOMENTUM`, `WD`.
- `external_repos/YOLOP/lib/config/default.py` — `LR0=0.001`, `OPTIMIZER='adam'`, `END_EPOCH=240`, `WARMUP_EPOCHS=3.0`, `WD=0.0005` (these are YOLOP's defaults; YOLOPv2 paper reports 0.01 / SGD / 300 / 0.005).
- `external_repos/YOLOPv2/demo.py` — output contract: `[pred, anchor_grid], seg, ll = model(img)`. Per-scale detection tensors `[bs, 3*85, ny, nx]`. `seg[:, :, 12:372, :]` then `interpolate(scale_factor=2)` → seg at H/2. `lane_line_mask` rounds sigmoid → lane is 1-ch.
- `external_repos/YOLOPv2/utils/utils.py::letterbox` — `new_shape=(640,640)`, `auto=True`, `stride=32`. Test-time resize uses rectangular letterbox for the 720→384 collapse.
- YOLOPv2 paper arXiv 2208.11434 — section 3: "more efficient ELAN structures", BoF = Mosaic + MixUp, focal loss for cls/obj and for lane seg, hybrid focal+dice ablation, centerline width 8 train / width 2 test.

## What `[INFERRED]` means and where it applies

Items marked `[INFERRED]` in the repaired code are components that the
YOLOPv2 paper describes in prose but does not release source for (the
public `YOLOPv2-main` is inference-only, `torch.jit`-scripted weights):

- Exact `groups` argument inside E-ELAN (we choose 2 at stages ≥ P4 so
  param count lands in the 38–40 M band the paper reports).
- Exact channel schedule.
- Exact scheme for pairing lane lines into centerlines (we use
  existence+visibility-ordered BDD poly2d annotations; the paper's
  tool could be different).
- Exact focal loss γ for lane seg (we default to 1.5, paper value not
  disclosed).

Every `[INFERRED]` site is labelled in code so a future reader can
swap it out if upstream source becomes available.
