# Port graph v3 evaluation record

The deployed v2 baseline is Git commit `3bb6ff7`. Its entire backend is loaded
in an isolated snapshot, including helper modules, rather than accidentally
importing the working geometry module.

- OPEN100 0–3: development and shape prototype construction.
- OPEN100 4–11: historical regression; these are no longer an unseen test set.
- First predeclared Dataset PID synthetic 8: negative validation/stress set.
  Unconditional v3 regressed and was rejected. Its full outputs remain under
  `results/rejected-*`; they are not release scores.
- Second predeclared Dataset PID synthetic 8: final unseen confirmation after
  supported-style gating and an exact v2 fallback were frozen.

`FROZEN.initial.json` retains the failed first configuration. `FROZEN.json` records
the final frozen sources and the reason for changing the initial design.
`stress-plan.json` / `stress-data.json` and `confirmation-plan.json` retain the
selection and source provenance. The final report is generated after all release
comparisons finish. All dataset-derived results/figures inherit CC BY-SA 4.0;
see `../NOTICE.md` and `../../backend/assets/NOTICE.md`.
