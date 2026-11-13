"""Paired tests on frozen current CTI-Bench answers; requires NumPy only."""
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
N_RESAMPLES = 200_000
SEED = 20261112

def read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    out = {r['question_id']: r for r in rows}
    assert len(out) == len(rows) == 1000
    return out

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def exact_mcnemar(wins, losses):
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.0

base = ROOT.parent / 'class_metrics1000'
gold = read(base / 'gold.jsonl')
agg = read(base / 'predictions.jsonl')
manifest = json.loads((ROOT / 'inputs.json').read_text())
results = {}
score_rows = []
for offset, (name, spec) in enumerate(manifest['readers'].items()):
    path = ROOT / spec['file']
    raw = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == spec['release_prediction_sha256']
    reader = read(path)
    assert set(reader) == set(gold) == set(agg)
    paired = []
    counts = [0, 0, 0, 0]
    for qid in sorted(gold, key=int):
        truth = gold[qid]['gold']
        ids = set(re.findall(r'CWE-\d+', reader[qid]['prediction'], re.I))
        ids = {label.upper() for label in ids}
        # Match the frozen CTI-Bench scorer: whole-answer extraction;
        # no Unicode-hyphen or numeric zero-padding changes.
        a = float(agg[qid]['predictions']['cosine_sum'] == truth)
        b = 2 / (len(ids) + 1) if truth in ids else 0.0
        exact = ids == {truth}
        counts[0 if a and exact else 1 if a else 2 if exact else 3] += 1
        paired.append((a, b))
        score_rows.append([name, qid, truth, agg[qid]['predictions']['cosine_sum'],
                           ';'.join(sorted(ids)), a, b, int(exact)])
    pair = np.array(paired)
    assert abs(100 * pair[:, 1].mean() - spec['reported_f1_percent']) < 1e-10
    assert int(pair[:, 0].sum()) == 763
    # Multinomial draws over identical paired score categories are exactly
    # equivalent to resampling question pairs with replacement.
    groups = Counter(paired)
    values = np.array(list(groups))
    rng = np.random.default_rng(SEED + offset * 2)
    draws = rng.multinomial(1000, np.array(list(groups.values())) / 1000, size=N_RESAMPLES)
    means = draws @ values / 1000
    differences = means[:, 0] - means[:, 1]
    observed = float((pair[:, 0] - pair[:, 1]).mean())
    # Independently swap method labels within each question under the null.
    # Group equal absolute differences; binomial draws count positive signs.
    diff_counts = Counter(abs(float(a - b)) for a, b in paired if a != b)
    prng = np.random.default_rng(SEED + offset * 2 + 1)
    null = np.zeros(N_RESAMPLES)
    for difference, count in sorted(diff_counts.items()):
        null += difference * (2 * prng.binomial(count, .5, size=N_RESAMPLES) - count) / 1000
    extreme = int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12))
    p = (extreme + 1) / (N_RESAMPLES + 1)
    ci = lambda xs: (100 * np.percentile(xs, [2.5, 97.5])).tolist()
    results[name] = {
        'n': 1000, 'aggregation_mean_f1_percent': 100 * pair[:, 0].mean(),
        'reader_mean_f1_percent': 100 * pair[:, 1].mean(),
        'difference_points': 100 * observed,
        'paired_bootstrap_95_CI_points': ci(differences),
        'aggregation_bootstrap_95_CI_percent': ci(means[:, 0]),
        'reader_bootstrap_95_CI_percent': ci(means[:, 1]),
        'two_sided_paired_randomization_p': p,
        'randomization_monte_carlo_se': float(math.sqrt(p * (1 - p) / (N_RESAMPLES + 1))),
        'binary_exact_match_secondary': {
            'both_correct': counts[0], 'aggregation_only_correct': counts[1],
            'reader_only_correct': counts[2], 'both_wrong': counts[3],
            'reader_correct': counts[0] + counts[2],
            'exact_two_sided_mcnemar_p': exact_mcnemar(counts[1], counts[2])},
        'bootstrap_seed': SEED + offset * 2, 'randomization_seed': SEED + offset * 2 + 1}
# Holm adjustment across the two newly reported F1 comparisons only.
ordered = sorted(results, key=lambda k: results[k]['two_sided_paired_randomization_p'])
previous = 0.0
for rank, name in enumerate(ordered):
    previous = max(previous, min(1., (len(ordered) - rank) * results[name]['two_sided_paired_randomization_p']))
    results[name]['f1_randomization_holm_p_two_comparisons'] = previous
with (ROOT / 'question_scores.csv').open('w', newline='') as stream:
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['comparison', 'question_id', 'gold', 'aggregation_prediction',
                     'reader_extracted_ids', 'aggregation_f1', 'reader_f1', 'reader_exact_match'])
    writer.writerows(score_rows)
output = {'bootstrap_resamples': N_RESAMPLES, 'randomization_draws': N_RESAMPLES,
    'difference_direction': 'aggregation minus reader, percentage points',
    'comparisons': results,
    'limitations': 'Conditional on frozen predictions; question-level independent-case assumption. No family clustering or model-run variability. Baselines selected retrospectively on these same questions. Holm adjustment covers only these two F1 tests, not historical exploration; intervals are pointwise, not simultaneous. Secondary binary McNemar tests address exact match, not mean question F1.'}
(ROOT / 'results.json').write_text(json.dumps(output, indent=2) + '\n')
files = [p for p in ROOT.iterdir() if p.is_file() and p.name != 'checksums.json']
checks = {p.name: digest(p) for p in sorted(files)}
checks.update({'../class_metrics1000/' + n: digest(base / n) for n in ['gold.jsonl', 'predictions.jsonl']})
(ROOT / 'checksums.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(output, indent=2))
