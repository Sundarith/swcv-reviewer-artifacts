# Evaluation

The executable offline entry point is `python -m swcv.reproduce` (implementation in `src/swcv/reproduce.py`). It validates all expected question IDs and reports strict canonical single-CWE accuracy for saved SWCV predictions.

This command reports strict single-label accuracy. Identifier-set precision/recall/F1 and class-level macro/micro metrics use different averaging rules and should be labeled separately.

Detailed class diagnostics for every selected reader are available in [CTI-Bench reader metrics](../supporting_results/reader_metrics1000/README.md) and [CTIConnect metrics](../supporting_results/cticonnect290/README.md). Their guides distinguish class macro/micro P/R/F1 from mean per-question identifier-set F1 and document each benchmark's extraction and accepted-alternative rules.
