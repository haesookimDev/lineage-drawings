# Public drawing data and attribution

Evaluation source: **PID2Graph v1**, Jan Marius Stürmer, Marius Graumann,
Tobias Koch; German Aerospace Center (DLR), 2025.
[Zenodo record 14803338](https://zenodo.org/records/14803338),
DOI [10.5281/zenodo.14803338](https://doi.org/10.5281/zenodo.14803338).

The 12 **Complete / PID2Graph OPEN100** drawings originate from the
[OPEN100](https://www.open-100.com/) public design by the
[Energy Impact Center](https://www.energyimpactcenter.org/).
They are public engineering design documents, not evidence of an operating or
commissioned plant. The synthetic subsets are not used in this evaluation.

The Zenodo record's machine-readable `metadata.license.id` is
**cc-by-sa-4.0**. Dataset files and their adapted visualizations, cropped examples,
prediction annotations and derived benchmark result files in `evaluation/results/`
and `evaluation/figures/` are provided under
[Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).
Changes: preview resizing to at most 2400 pixels, prediction overlays, crops,
class-agnostic localization matching and topology-primitive contraction for
undirected equipment-adjacency scoring. These are our predictions/measurements,
not annotations endorsed by the original creators. There is no additional
restriction on those CC BY-SA materials. No model weights were trained.

The original PNG/GraphML pairs are downloaded separately into a Git-ignored
`.benchmark-data/open100/` folder. Exact original ZIP members, CRC32 and SHA256
are pinned in `data_manifest.json`. Source archive MD5 as listed by Zenodo:
`90f782220de97e7e249d2595c49ddc1c` (the entire 9.3 GB archive is not downloaded or
MD5-verified; selected members are CRC32-verified and SHA256-pinned).

Paper: Stürmer, Graumann & Koch, *From Engineering Diagrams to Graphs:
Digitizing P&IDs with Transformers*, IEEE DSAA 2025,
[DOI 10.1109/DSAA65442.2025.11248012](https://doi.org/10.1109/DSAA65442.2025.11248012),
[author preprint](https://arxiv.org/abs/2411.13929).

This data notice does not assign a new license to unrelated application code.


The v3 extension also evaluates two disjoint eight-sheet selections from the
public synthetic **Complete / Dataset PID** subset (Digitize-PID, Paliwal et al.,
2021). These are synthetic, not real OPEN100 plans. Selection, hashes and roles
are recorded under `v3/stress-*.json` and `v3/confirmation-*.json`. The v3
prototypes use only OPEN100 development 0–3; see `../backend/assets/NOTICE.md`.
All v3 dataset-derived result JSON/CSV and figures follow CC BY-SA 4.0 as above.
