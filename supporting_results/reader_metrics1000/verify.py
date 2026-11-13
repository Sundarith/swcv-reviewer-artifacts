"""Verify hashes, confusion marginals and all 99 reported reader F1 scores."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for name, expected in json.loads((ROOT / 'checksums.json').read_text()).items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
metrics = json.loads((ROOT / 'metrics.json').read_text())
for name, values in metrics.items():
    actual, predicted, tp = Counter(), Counter(), Counter()
    with (ROOT / 'class_results' / f'{name}_confusion.csv').open() as stream:
        for row in csv.DictReader(stream):
            n = int(row['questions'])
            actual[row['canonical_gold']] += n
            ids = [] if row['complete_predicted_id_set'] == '<EMPTY>' else row['complete_predicted_id_set'].split(';')
            for label in ids:
                predicted[label] += n
                if label == row['canonical_gold']:
                    tp[label] += n
    assert sum(actual.values()) == values['n'] == 1000
    with (ROOT / 'class_results' / f'{name}_per_cwe.csv').open() as stream:
        for row in csv.DictReader(stream):
            label = row['cwe']
            assert (int(row['support']), int(row['predicted']), int(row['tp'])) == (actual[label], predicted[label], tp[label])
manifest = json.loads((ROOT / 'inputs.json').read_text())
assert len(manifest['conditions']) == 99 and len(metrics) == 107
for c in manifest['conditions']:
    assert hashlib.sha256(gzip.decompress((ROOT / c['file']).read_bytes())).hexdigest() == c['uncompressed_sha256']
    assert abs(metrics[c['name']]['mean_question_identifier_set_metrics']['f1'] * 100 - c['reported_f1_percent']) < 1e-9
print('All hashes, 107 confusion totals/marginals and 99 headline F1 scores verified.')
