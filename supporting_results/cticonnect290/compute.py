"""Current CTIConnect metrics and paired tests; offline, Python + NumPy."""
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
import numpy as np
import official_metrics as official

ROOT = Path(__file__).resolve().parent
DRAWS = 200_000
SEED = 20261116

def raw(path):
    return gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()

def rows(path):
    return [json.loads(line) for line in raw(path).decode().splitlines()]

def indexed(path, key='question_id'):
    items = rows(path)
    result = {str(r[key]): r for r in items}
    assert len(result) == len(items) == 290
    return result

def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + '\n')

def csvfile(name, header, values):
    with (ROOT / name).open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(values)

def mcnemar(wins, losses):
    n = wins + losses
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.

manifest = json.loads((ROOT / 'inputs.json').read_text())
for item in [*manifest['conditions'], *(manifest[k] for k in ['gold','retrieval','selected_predictions','exclusions','scorer'])]:
    assert hashlib.sha256(raw(ROOT / item['file'])).hexdigest() == item['uncompressed_sha256'], item['file']
ranking = indexed(ROOT / manifest['retrieval']['file'])
selected = indexed(ROOT / manifest['selected_predictions']['file'])
allowed = set(json.loads((ROOT / 'exclusions.json').read_text())['filtered_eligible_rows'])
controls = []
for qid, row in ranking.items():
    assert len(row['hits']) == 20 and all(h['row'] in allowed for h in row['hits'])
    predictions = {}
    for k in [10, 20]:
        weighted = Counter()
        unweighted = Counter()
        for hit in row['hits'][:k]:
            for label in hit['cwe_ids']:
                weighted[label] += hit['similarity']
                unweighted[label] += 1
        predictions[f'aggregation_k{k}'] = min(weighted, key=lambda c: (-weighted[c], c))
        predictions[f'majority_k{k}'] = min(unweighted, key=lambda c: (-unweighted[c], c))
        assert predictions[f'aggregation_k{k}'] == selected[qid]['predictions'][str(k)]
    predictions['nearest_neighbor'] = min(row['hits'][0]['cwe_ids'])
    controls.append({'question_id': qid, 'predictions': predictions})
(ROOT / 'control_predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in controls))
save('control_freeze.json', {'sha256': hashlib.sha256((ROOT/'control_predictions.jsonl').read_bytes()).hexdigest(),
    'rule': 'One full vote or full cosine contribution per attached CWE; lexical CWE tie break; existing ordered top20; k10 retained, not retuned',
    'n': 290, 'scoring_gold_loaded_after_control_ledger_written': True})
truth = indexed(ROOT / 'gold.jsonl', 'id')
assert set(truth) == set(ranking)
normalizer = re.compile(r'(?<=CWE)[\u2010-\u2015\u2212\ufe63\uff0d](?=\d)', re.I)
conditions = {c['name']: indexed(ROOT / c['file']) for c in manifest['conditions']}
for name in controls[0]['predictions']:
    conditions[name] = {r['question_id']: {'prediction': r['predictions'][name]} for r in controls}
metrics = {}
support_bins = []
question_scores = {}
(ROOT / 'class_results').mkdir(exist_ok=True)
for name, predictions in conditions.items():
    assert set(predictions) == set(truth)
    actual = Counter(r['ground_truth']['target_id'] for r in truth.values())
    predicted = Counter()
    tp = Counter()
    confusion = Counter()
    items = []
    score_records = {}
    for qid in sorted(truth):
        answer = normalizer.sub('-', predictions[qid]['prediction'])
        gold = truth[qid]['ground_truth']
        canonical = gold['target_id']
        ids = official.extract_ids(answer, 'cwe')
        item = official.score_id_item(answer, gold, task='rcm', item_id=qid)
        items.append(item)
        predicted.update(ids)
        if canonical in ids: tp[canonical] += 1
        confusion[canonical, ';'.join(sorted(ids)) or '<EMPTY>'] += 1
        score_records[qid] = {'f1':item.f1,'official_correct':bool(item.exact_match),
                             'canonical':canonical,'ids':sorted(ids)}
    question_scores[name] = score_records
    labels = sorted(set(actual) | set(predicted))
    class_rows = []
    for label in labels:
        p = tp[label] / predicted[label] if predicted[label] else 0.
        r = tp[label] / actual[label] if actual[label] else 0.
        f = 2 * tp[label] / (actual[label] + predicted[label])
        class_rows.append([label, actual[label], predicted[label], tp[label], p, r, f])
    total_tp = sum(tp.values())
    micro_p = total_tp / sum(predicted.values()) if predicted else 0.
    micro_r = total_tp / 290
    micro_f = 2 * total_tp / (290 + sum(predicted.values()))
    metrics[name] = {'n':290,'official_accepted_alternative_mean_question_metrics':official.aggregate_scores(items)['overall'],
        'canonical_class_metrics':{'macro_precision':sum(r[4] for r in class_rows)/len(labels),
            'macro_recall':sum(r[5] for r in class_rows)/len(labels),
            'macro_f1':sum(r[6] for r in class_rows)/len(labels),
            'micro_precision':micro_p,'micro_recall':micro_r,'micro_f1':micro_f,'classes':len(labels)},
        'canonical_exact_correct':sum(s['ids']==[s['canonical']] for s in score_records.values()),
        'official_correct':sum(i.exact_match for i in items),
        'empty_id_sets':sum(not s['ids'] for s in score_records.values()),
        'multiple_id_sets':sum(len(s['ids'])>1 for s in score_records.values())}
    assert sum(confusion.values()) == 290 and sum(actual.values()) == 290
    for bin_name, select in [('support_at_most_5', lambda n:n<=5), ('support_above_5', lambda n:n>5)]:
        members=[r for r in score_records.values() if select(actual[r['canonical']])]
        support_bins.append([name,bin_name,len(members),sum(r['official_correct'] for r in members),sum(r['f1'] for r in members)/len(members) if members else 0])
    if name not in {c['name'] for c in manifest['conditions']}:
        csvfile(f'class_results/{name}_matrix.csv', ['canonical_gold / predicted']+labels,
                ([label]+[confusion[label,p] for p in labels] for label in labels))
    csvfile(f'class_results/{name}_per_cwe.csv', ['cwe','support','predicted','tp','precision','recall','f1'], class_rows)
    csvfile(f'class_results/{name}_confusion.csv', ['canonical_gold','complete_predicted_id_set','questions'],
            ([a,b,n] for (a,b),n in sorted(confusion.items())))
for c in manifest['conditions']:
    assert abs(metrics[c['name']]['official_accepted_alternative_mean_question_metrics']['f1']*100-c['reported_f1_percent'])<1e-9
assert metrics['aggregation_k10']['official_correct'] == 161
assert metrics['aggregation_k20']['official_correct'] == 158
save('metrics.json', metrics)
csvfile('support_bins.csv',['condition','gold_class_support_bin','questions','official_correct','official_mean_question_f1'],support_bins)
comparison_names = ['majority_k10',
    max((c for c in manifest['conditions'] if c['method']=='naive_rag'),key=lambda c:c['reported_f1_percent'])['name'],
    max(manifest['conditions'],key=lambda c:c['reported_f1_percent'])['name']]
comparisons = {}
for offset, name in enumerate(comparison_names):
    pairs = [(question_scores['aggregation_k10'][q]['f1'], question_scores[name][q]['f1']) for q in sorted(truth)]
    groups = Counter(pairs)
    values = np.array(list(groups))
    draws = np.random.default_rng(SEED+offset*2).multinomial(290,np.array(list(groups.values()))/290,size=DRAWS)
    means = draws @ values / 290
    obs = float(np.mean([a-b for a,b in pairs]))
    counts = Counter(abs(a-b) for a,b in pairs if a != b)
    null = np.zeros(DRAWS)
    rng = np.random.default_rng(SEED+offset*2+1)
    for diff,count in sorted(counts.items()): null += diff*(2*rng.binomial(count,.5,size=DRAWS)-count)/290
    p = (1+int(np.count_nonzero(np.abs(null)>=abs(obs)-1e-12)))/(DRAWS+1)
    binary = [0]*4
    for q in truth:
        a=question_scores['aggregation_k10'][q]['official_correct'];b=question_scores[name][q]['official_correct']
        binary[0 if a and b else 1 if a else 2 if b else 3] += 1
    ci = lambda x:(100*np.percentile(x,[2.5,97.5])).tolist()
    comparisons[name] = {'aggregation_f1_percent':metrics['aggregation_k10']['official_accepted_alternative_mean_question_metrics']['f1']*100,
        'comparator_f1_percent':metrics[name]['official_accepted_alternative_mean_question_metrics']['f1']*100,
        'difference_points':100*obs,'paired_bootstrap_95_CI_points':ci(means[:,0]-means[:,1]),
        'aggregation_bootstrap_95_CI_percent':ci(means[:,0]),'comparator_bootstrap_95_CI_percent':ci(means[:,1]),
        'paired_randomization_p':p,'randomization_monte_carlo_se':math.sqrt(p*(1-p)/(DRAWS+1)),
        'official_binary_secondary':{'counts_both_aggregation_only_comparator_only_neither':binary,'exact_mcnemar_p':mcnemar(binary[1],binary[2])},
        'bootstrap_seed':SEED+offset*2,'randomization_seed':SEED+offset*2+1}
previous = 0.
for rank,name in enumerate(sorted(comparisons,key=lambda n:comparisons[n]['paired_randomization_p'])):
    previous=max(previous,min(1.,(3-rank)*comparisons[name]['paired_randomization_p']))
    comparisons[name]['holm_p_three_comparisons']=previous
save('statistics.json',{'n':290,'resamples':DRAWS,'difference_direction':'aggregation_k10 minus comparator in percentage points',
    'comparisons':comparisons,'limitations':'Retrospective comparator/k selection; conditional frozen predictions; independent-question assumption; no family clustering or model-run uncertainty. Holm covers these three CTIConnect F1 tests only; intervals are pointwise. Official binary accepted-alternative correctness differs from fractional F1 for readers. Naive RAG uses20 references versus aggregation10.'})
with gzip.GzipFile(filename='',mode='wb',fileobj=(ROOT/'question_scores.jsonl.gz').open('wb'),mtime=0) as stream:
    for name,scores in question_scores.items():
        for qid,row in sorted(scores.items()):stream.write((json.dumps({'condition':name,'question_id':qid,**row})+'\n').encode())
print(json.dumps({'conditions_verified':len(manifest['conditions']),'controls':len(controls[0]['predictions']),'comparisons':comparisons},indent=2))
