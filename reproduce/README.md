# Reproduce saved results

From the repository root:

```bash
PYTHONPATH=src python -m swcv.reproduce --run swcv_exact_baseline
PYTHONPATH=src python -m swcv.reproduce --run swcv_union_filtered
```

No network calls, model downloads, GPU access or API keys are required. This rebuilds votes/scores from saved retrieval rather than regenerating the retrieval ranking. Original 335-question and remaining 665-question scores are also reported separately.
