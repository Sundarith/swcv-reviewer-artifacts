# Reproduce the current aggregation result

From the repository root, use the primary filtered condition reported in the paper:

```bash
PYTHONPATH=src python -m swcv.reproduce --run swcv_union_filtered
```

Its `full1000` result is **763/1,000 (76.30%)**, using the primary 120,238-reference corpus and twenty neighbors. The command reconstructs votes from saved retrieval and checks the frozen predictions. It also preserves historical subgroup output; the main paper reports the full 1,000-question result.

For both benchmarks, all language-model table cells, class metrics, ablations and uncertainty analyses, follow the [reviewer walkthrough](../docs/REPRODUCING_PAPER.md). No network calls, model downloads, GPU access or API keys are required for saved-result checks.

## Earlier exclusion condition

The preserved `swcv_exact_baseline` condition removes target IDs and exact-description matches before the additional related-reference screening. It scores 765/1,000 (76.50%); it is not the primary 76.30% paper condition. Its optional verification command is:

```bash
PYTHONPATH=src python -m swcv.reproduce --run swcv_exact_baseline
```

Both commands reproduce archived votes and scores; neither regenerates the retrieval ranking or changes an experiment.
