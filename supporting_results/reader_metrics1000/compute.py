"""Reproduce current CTI-Bench reader/classifier class diagnostics offline."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import official_metrics as official

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'class_metrics1000'

def load(path):
    data = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    rows = [json.loads(line) for line in data.decode().splitlines()]
    result = {r['question_id']: r for r in rows}
    assert len(result) == len(rows) == 1000
    return result

def csvfile(name, header, rows):
    with (ROOT / name).open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)

manifest = json.loads((ROOT / 'inputs.json').read_text())
assert hashlib.sha256((ROOT / 'official_metrics.py').read_bytes()).hexdigest() == manifest['scorer_sha256']
gold = load(BASE / 'gold.jsonl')
controls = load(BASE / 'predictions.jsonl')
conditions = {}
for c in manifest['conditions']:
    path = ROOT / c['file']
    assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == c['uncompressed_sha256']
    conditions[c['name']] = load(path)
for name in next(iter(controls.values()))['predictions']:
    conditions['classifier__' + name] = {q: {'prediction': r['predictions'][name]} for q, r in controls.items()}
(ROOT / 'class_results').mkdir(exist_ok=True)
metrics = {}
support_bins = []
with (ROOT / 'question_scores.jsonl.gz').open('wb') as file:
    with gzip.GzipFile(filename='', fileobj=file, mode='wb', mtime=0) as stream:
        for name, predictions in conditions.items():
            assert set(predictions) == set(gold)
            actual = Counter(r['gold'] for r in gold.values())
            predicted = Counter()
            tp = Counter()
            confusion = Counter()
            items = []
            case_scores = []
            for qid in sorted(gold, key=int):
                answer = predictions[qid]['prediction']
                canonical = gold[qid]['gold']
                # Preserve the selected CTI-Bench whole-response extraction:
                # no Unicode-hyphen preprocessing, final-line selection or repair.
                ids = official.extract_ids(answer, 'cwe')
                item = official.score_id_item(answer, {'target_type': 'cwe', 'target_id': canonical})
                items.append(item)
                predicted.update(ids)
                if canonical in ids:
                    tp[canonical] += 1
                confusion[canonical, ';'.join(sorted(ids)) or '<EMPTY>'] += 1
                case = dict(condition=name, question_id=qid, canonical=canonical,
                            ids=sorted(ids), precision=item.precision, recall=item.recall,
                            f1=item.f1, exact_match=item.exact_match)
                case_scores.append(case)
                stream.write((json.dumps(case) + '\n').encode())
            labels = sorted(set(actual) | set(predicted))
            per_class = []
            for label in labels:
                p = tp[label] / predicted[label] if predicted[label] else 0.
                r = tp[label] / actual[label] if actual[label] else 0.
                f1 = 2 * tp[label] / (actual[label] + predicted[label])
                per_class.append([label, actual[label], predicted[label], tp[label], p, r, f1])
            total_tp = sum(tp.values())
            metrics[name] = {
                'n': 1000,
                'mean_question_identifier_set_metrics': official.aggregate_scores(items)['overall'],
                'canonical_class_metrics': {
                    'macro_precision': sum(r[4] for r in per_class) / len(labels),
                    'macro_recall': sum(r[5] for r in per_class) / len(labels),
                    'macro_f1': sum(r[6] for r in per_class) / len(labels),
                    'micro_precision': total_tp / sum(predicted.values()) if predicted else 0.,
                    'micro_recall': total_tp / 1000,
                    'micro_f1': 2 * total_tp / (1000 + sum(predicted.values())),
                    'classes': len(labels)},
                'exact_correct': sum(item.exact_match for item in items),
                'empty_id_sets': sum(not row['ids'] for row in case_scores),
                'multiple_id_sets': sum(len(row['ids']) > 1 for row in case_scores)}
            assert sum(confusion.values()) == 1000
            csvfile(f'class_results/{name}_per_cwe.csv',
                    ['cwe', 'support', 'predicted', 'tp', 'precision', 'recall', 'f1'], per_class)
            csvfile(f'class_results/{name}_confusion.csv',
                    ['canonical_gold', 'complete_predicted_id_set', 'questions'],
                    ([a, b, n] for (a, b), n in sorted(confusion.items())))
            for bin_name, select in [('support_at_most_5', lambda n: n <= 5), ('support_above_5', lambda n: n > 5)]:
                cases = [row for row in case_scores if select(actual[row['canonical']])]
                support_bins.append([name, bin_name, len(cases), sum(row['exact_match'] for row in cases),
                                     sum(row['f1'] for row in cases) / len(cases) if cases else 0.])
for c in manifest['conditions']:
    assert abs(100 * metrics[c['name']]['mean_question_identifier_set_metrics']['f1'] - c['reported_f1_percent']) < 1e-9
previous = json.loads((BASE / 'metrics.json').read_text())
for name, old in previous.items():
    new = metrics['classifier__' + name]['canonical_class_metrics']
    for field in ['macro_precision', 'macro_recall', 'macro_f1', 'micro_precision', 'micro_recall', 'micro_f1']:
        assert abs(old[field] - new[field]) < 1e-12, (name, field)
(ROOT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
csvfile('support_bins.csv', ['condition', 'gold_class_support_bin', 'questions', 'exact_correct', 'mean_question_f1'], support_bins)
print('All 99 reader F1 cells and eight existing classifier metric rows reproduced; 107 confusion summaries each contain 1,000 cases.')
