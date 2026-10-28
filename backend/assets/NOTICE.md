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
