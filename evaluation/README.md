# Evaluation

The executable offline entry point is `python -m swcv.reproduce` (implementation in `src/swcv/reproduce.py`). It validates all expected question IDs and reports strict canonical single-CWE accuracy for saved SWCV predictions.

This command reports strict single-label accuracy. Identifier-set precision/recall/F1 and class-level macro/micro metrics use different averaging rules and should be labeled separately.
