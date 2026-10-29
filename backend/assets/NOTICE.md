# Shape prototype attribution

`open100-dev-shapes.npz` contains resized/binarized crops from development sheets
0, 1, 2 and 3 of PID2Graph v1, Complete / PID2Graph OPEN100, Jan Marius Stürmer,
Marius Graumann and Tobias Koch (DLR), 2025; original drawings by Energy Impact
Center / OPEN100. Source: https://zenodo.org/records/14803338

The NPZ weights and their provenance JSON are adaptations distributed under
Creative Commons Attribution-ShareAlike 4.0 International:
https://creativecommons.org/licenses/by-sa/4.0/

Changes: resize to at most 2400px; crop development bounding boxes; binarize;
deduplicate shape examples. No other sheets or production user drawings are used.
These shape hints are not standardized identifiers or endorsed classifications.
This notice does not assign a new license to unrelated application source code.

## Learned symbol objectness forest (v5)

`symbol-objectness.xml.gz` is a CPU-trained OpenCV HOG/ink/context random
forest, not a pretrained neural detector. Training used PID2Graph OPEN100
sheets 0–3, twelve PID2Graph synthetic sheets listed in its metadata, and
manually labeled process regions of Pahala concept-design sheets 23–24.
Training crops and source PDFs are not shipped. The PID2Graph-derived forest
and provenance are distributed under CC BY-SA 4.0 with the attribution above.
Pahala source: US EPA-hosted B&C drawings dated July 23, 2021:
https://www.epa.gov/system/files/documents/2021-08/pahala-wwtp-design-build-drawings-2021-07-23.pdf
Government hosting does not establish an open license for third-party drawings.
The model contains learned numerical tree splits, not the source drawings;
training-data rights should be reviewed before broader commercial reuse.

Changes: bounded multi-scale proposals; 32px HOG, foreground grid and context
features; exact/jittered training boxes and hard negatives; 128-tree binary
objectness forest. Scores are vote fractions, not calibrated probabilities.
It predicts candidate extents only; types, standard IDs and connection approval
remain subject to human review. See evaluation/v5 for data roles and limitations.

## Tiled neural symbol detector (v6)

`symbol-neural.onnx` is a fine-tuned YOLOX-Nano detector with one generic symbol
class. Official upstream implementation: https://github.com/Megvii-BaseDetection/YOLOX
commit `6ddff4824372906469a7fae2dc3206c7aa4bbaee`, Apache License 2.0
(copy at evaluation/v6/upstream/YOLOX-LICENSE.txt). Initial COCO weights:
https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth

Changes: replace the 80-class head with one generic class; CPU fine-tuning on
607 labeled tiles from 10 real sheets/ROIs and twelve synthetic sheets; select
epoch16 using validation; export raw predictions to ONNX; bounded high-resolution
tiles and overlap suppression in OpenCV. No semantic classes/standard IDs or
approved connections are inferred. The upstream COCO scores do not describe
this fine-tuned detector's diagram performance.

Training sources add the v5 Virden5/7 and Ruttan50/62 regions to the previous
OPEN1000–3 / Pahala23–24 / synthetic12 sources. Public source URLs and hashes
are in evaluation/v5/real-sources.json and evaluation/v6. Former v5 test regions
are explicitly training data for v6. No production user uploads were used.
PID2Graph-derived fine-tuned weights/provenance retain CC BY-SA4.0 attribution
above. Manitoba-hosted third-party engineering PDFs are not assumed to have an
open redistribution license; source PDFs, page images and crops are excluded
from Git. This notice does not relicense unrelated application code.

## Line, port and junction recognition (v7)

`topology.onnx` is a small six-head U-Net trained from scratch on CPU, exported
with OpenCV/PyTorch parity checks. Weak GraphML line/port projections use only
PID2Graph OPEN100 sheets0–3 and the twelve synthetic training sheets listed in
evaluation/v7/TRAINING-DATA.json. In addition,1,200 procedurally generated scenes
supply exact masks. Graph topology labels are weak supervision; a degree-four
node does not certify a visible junction dot. Derived weights/provenance retain
the CC BY-SA4.0 PID2Graph attribution above. No third-party source page images or
production uploads are included in these topology weights' training set.

Six heads: solid/dashed strokes, port, T, connected-four-way, plain crossing.
Scores are uncalibrated and all observations need review. Plain-cross heatmap
F1 at0.5 is zero on procedural validation; geometric crossing rules remain
necessary. New learned routing did not beat the existing router on validation,
so deployed v7 retains v6 adjacency and supplies recognition evidence only.
See evaluation/v7 for rejected routing variants, frozen artifacts and limits.

## Sparse port/junction context forests (v8)

`port-context.xml.gz`, `junction-context.xml.gz` and `topology-context.json`
replace full-page topology CNN maps in the default v8 inference path. These
CPU OpenCV random forests use point-local HOG/ink/arm features and reviewed
ports/junctions from five real development sheets/ROIs,60 visually reviewed
OPEN100 source points, weak GraphML port supervision from OPEN1000–3 and the
same12 synthetic training sheets, and300 exact procedural scenes. Former v7
three test ROIs are explicitly training data for v8, never new test evidence.
Training manifests and separate validation/test source labels are in evaluation/v8.

The PID2Graph-derived forests/provenance retain CC BY-SA4.0 attribution above.
Public third-party engineering PDFs and source crops are not redistributed.
Votes are uncalibrated; a high-confidence displayed junction and a lower-score
routing association hypothesis are distinct records, both requiring review.
Validated old connection candidates are retained; only threshold-gated visible
paths are added within the same500-edge output budget. Native DXF connections
continue to use original geometry; no semantic/flow approval is inferred.

## v9 symbol detector and periodic line recovery

The generic YOLOX-Nano weights were further fine-tuned on the same licensed
PID2Graph/OPEN100 and synthetic training sources plus public prior-project ROI
annotations explicitly reclassified as training. Additional glyph labels:73
from former v6/v7/v8-R1 tests, never the new Winkler/Winnipeg test projects.
Initial v9 pass: v6 selected epoch16,12epochs64batches; selected epoch4 then
warm-started a scale-shrink pass8epochs64batches. Selected scale epoch8 uses
random0.35-1.0 shrink/placement to match small native drawing glyph sizes.
Exact weights, data, iteration failures, validation choice and source hashes:
evaluation/v9/. Original YOLOX Apache2.0 code attribution and PID2Graph-derived
CC-BY-SA4.0 weight/provenance attribution above continue to apply.
Point-context RTrees weights are unchanged from v8-R2. Routing thresholds and
pixel-supported periodic line recovery were selected only on old-project
validation. All outputs remain review candidates; no standard IDs or physical
flow/safety approval are invented. Government-hosted source PDFs, page pixels,
private runtime data and full government OCR are excluded from Git/images.

## v10 regression repair and inference reuse

The selected generic detector linearly interpolates the floating tensors of
the existing v6 epoch16 and v9 scale epoch8 checkpoints at equal weights.
Integer buffers come from v9. It is one network per page, with no new gradient
training and no change to the source training data or licenses above.
The coefficient and configuration were selected on explicitly reused old
validation, historical OPEN1006–11 repair pages and previous35 test labels.
These are development data for v10, not unseen-test evidence. Enlarged source
annotations on the same untrained Winkler/Winnipeg pages were withheld from
selection; prior whole-image/project exposure is disclosed in evaluation/v10.
Local template support replaces discarded whole-page proposals in neural-only
mode; foreground integral counts and batched dash predicates reuse exact pixel
evidence. Port/junction forests, candidate budgets and human review remain.
