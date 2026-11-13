"""Rebuild current CTI-Bench classifier metrics and confusion matrices offline."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines()]

def write_csv(name, header, rows):
    with (ROOT / name).open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)

gold_rows = read('gold.jsonl')
records = read('predictions.jsonl')
gold = {r['question_id']: r['gold'] for r in gold_rows}
assert len(gold_rows) == len(gold) == len(records) == 1000
assert len({r['question_id'] for r in records}) == 1000
assert {r['question_id'] for r in records} == set(gold)
methods = sorted(records[0]['predictions'])
assert all(sorted(r['predictions']) == methods for r in records)
summary = {}
for method in methods:
    pairs = Counter((gold[r['question_id']], r['predictions'][method]) for r in records)
    actual = Counter(gold.values())
    predicted = Counter(r['predictions'][method] for r in records)
    labels = sorted(set(actual) | set(predicted))
    matrix = [[pairs[g, p] for p in labels] for g in labels]
    assert sum(map(sum, matrix)) == 1000
    assert all(sum(row) == actual[g] for g, row in zip(labels, matrix))
    assert all(sum(row[j] for row in matrix) == predicted[p] for j, p in enumerate(labels))
    rows = []
    for label in labels:
        tp = pairs[label, label]
        precision = tp / predicted[label] if predicted[label] else 0
        recall = tp / actual[label] if actual[label] else 0
        f1 = 2 * tp / (actual[label] + predicted[label])
        rows.append([label, actual[label], predicted[label], tp, precision, recall, f1])
    correct = sum(pairs[label, label] for label in labels)
    summary[method] = dict(n=1000, correct=correct, accuracy=correct / 1000,
        macro_precision=sum(r[4] for r in rows) / len(labels),
        macro_recall=sum(r[5] for r in rows) / len(labels),
        macro_f1=sum(r[6] for r in rows) / len(labels),
        micro_precision=correct / 1000, micro_recall=correct / 1000,
        micro_f1=correct / 1000, classes=len(labels))
    write_csv(f'{method}_confusion.csv', ['gold / predicted'] + labels,
              ([label] + row for label, row in zip(labels, matrix)))
    write_csv(f'{method}_per_cwe.csv', ['cwe', 'support', 'predicted', 'correct', 'precision', 'recall', 'f1'], rows)
assert summary['cosine_sum']['correct'] == 763
assert summary['majority']['correct'] == 757
(ROOT / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
files = sorted(p for p in ROOT.iterdir() if p.suffix in {'.csv', '.json', '.jsonl', '.py'})
(ROOT / 'checksums.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
    for p in files if p.name != 'checksums.json'}, indent=2) + '\n')
print(json.dumps(summary, indent=2))
