# Sparse context weights, actual-bus training revision

This supplement describes the current `topology-context.json` revision
`v8-r2-real-instrument-bus`. It supersedes the v8 training sample counts in
NOTICE.md for the two current sparse forests; licensing/primary attribution in
NOTICE.md remains applicable. The inference code/schema is unchanged from v8.

R2 adds the three R1 test ROIs (Virden10/12, Pahala18) to training after R1 failed
to improve their connection recall. They are explicitly development/training,
not R2 held-out evidence. Current manual training covers8 government-source
ROIs with71 ports and31 positive/negative junction points, plus60 reviewed
OPEN100 source junction/corner points. The existing weak GraphML and300 exact
procedural training sources remain. Pahala18 adds real black-dot instrument-bus
junctions absent from the first sparse model's real positive training points.

See evaluation/v8r2/TRAINING.json, PLAN.json and MODEL-SELECTION.json for hashes,
data roles, threshold selection and limitations. New test ROIs use Pahala14 and
Virden13/14 only. Public third-party source PDFs/page images/crops/full OCR are
not redistributed. PID2Graph-derived weights/provenance retain the CC BY-SA4.0
attribution given in NOTICE.md. No user production uploads are used for training.
