"""CPU baselines over the frozen filtered CTI-Bench RCM evidence."""
import csv
import fcntl
import hashlib
import json
from collections import Counter
from pathlib import Path

import bm25s
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = ROOT / 'logs/experiments/rcm335_nvd_first_fixed_pool_all_models_20261018_v1/shared'
FULL = ROOT / 'logs/experiments/rcm1000_source_priority_all_models_20261017_v1/shared'
NEIGHBORS = OUT.parent / 'ctibench1000_k_sensitivity_20261108'
MASK = OUT.parent / 'ctibench1000_final_group_filters_20261107/source_supported_exclusions.json'
TEMPERATURES = (.01, .025, .05, .1, .2, .5, 1.)
SEED = 20261110
K = 20


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(name, value):
    path = OUT / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def vote(label_sets, weights):
    totals = Counter()
    for labels, weight in zip(label_sets, weights, strict=True):
        for label in labels:
            totals[label] += float(weight)
    return min(totals, key=lambda label: (-totals[label], label))


def softmax_weights(scores, temperature):
    # The normalization is shared by all labels and cancels in the argmax.
    return np.exp((scores - scores.max()) / temperature)


def main():
    assert not (OUT / 'predictions.jsonl').exists()
    paths = [Path(__file__), MASK, NEIGHBORS / 'predictions.jsonl',
             BASE / 'documents.jsonl', BASE / 'document_embeddings.npy',
             BASE / 'queries.jsonl', BASE / 'query_embeddings.npy',
             FULL / 'query_embeddings.npy']
    save('protocol.json', {
        'inputs_sha256': {str(path.relative_to(ROOT)): sha(path) for path in paths},
        'libraries': {'numpy': np.__version__, 'sklearn': sklearn.__version__, 'bm25s': bm25s.__version__},
        'reference_pool': 120238, 'k': K, 'dev_seed': SEED, 'dev_size': 1000,
        'temperature_grid': TEMPERATURES,
        'temperature_selection': 'Most development CVEs whose structured label set contains the prediction. Ties favor larger temperature. Development records are excluded as a group from their own search index. Benchmark votes use the full filtered corpus.',
        'development_limitation': 'Random reference holdout, not a temporal or vulnerability-family split.',
        'inverse_distance': '1 / max(1 - cosine, 1e-6)',
        'bm25': 'bm25s defaults, English stopwords, full stored reference text and original question, top 20 unweighted label vote',
        'dense_supervised': 'Multilabel ridge (alpha=1) on unit embeddings, unit total target mass per record, argmax over fitted class scores',
        'text_supervised': 'Multinomial NB (alpha=1) with unigram and bigram TF-IDF (min_df=2, up to 100000 features). Fit supervised weights on all structured mappings with unit total sample weight per record.',
        'tie_breaks': 'Corpus order for retrieval ties, lexical CWE ID for prediction ties',
        'gold': 'Open only after prediction ledger is frozen',
    })
    assert sha(NEIGHBORS / 'predictions.jsonl') == read(NEIGHBORS / 'prediction_freeze.json')['sha256']
    ranking = rows(NEIGHBORS / 'predictions.jsonl')
    assert [r['row_index'] for r in ranking] == list(range(1000))
    eligible = read(MASK)['filtered_eligible_rows']
    assert len(eligible) == len(set(eligible)) == 120238
    all_docs = rows(BASE / 'documents.jsonl')
    docs = [all_docs[i] for i in eligible]
    del all_docs
    labels = [json.loads(row['raw_data'])['cwe_ids'] for row in docs]
    texts = [doc['text'] for doc in docs]
    assert all(labels)
    classes = np.array(sorted({label for group in labels for label in group}))
    class_idx = {label: i for i, label in enumerate(classes)}
    dense = np.load(BASE / 'document_embeddings.npy', mmap_mode='r')[eligible]
    query = np.load(FULL / 'query_embeddings.npy')
    for row, vector in zip(rows(BASE / 'queries.jsonl'), np.load(BASE / 'query_embeddings.npy'), strict=True):
        query[row['row_index']] = vector

    print('Selecting softmax temperature', flush=True)
    dev = np.sort(np.random.default_rng(SEED).choice(len(docs), 1000, replace=False))
    train = np.ones(len(docs), dtype=bool)
    train[dev] = False
    counts = Counter()
    for start in range(0, len(dev), 32):
        batch = dev[start:start + 32]
        block = dense[batch] @ dense.T
        block[:, ~train] = -np.inf
        for row, scores in zip(batch, block, strict=True):
            top = np.argsort(-scores, kind='stable')[:K]
            groups = [labels[i] for i in top]
            for temperature in TEMPERATURES:
                counts[temperature] += vote(groups, softmax_weights(scores[top], temperature)) in labels[row]
    temperature = max(TEMPERATURES, key=lambda t: (counts[t], t))
    save('temperature_selection.json', {'development_cves': [json.loads(docs[i]['raw_data'])['identifier'] for i in dev],
                                        'correct_by_temperature': counts, 'selected': temperature})

    print('Indexing BM25', flush=True)
    index = bm25s.BM25()
    index.index(bm25s.tokenize(texts, stopwords='en', show_progress=False), show_progress=False)
    prompts = [r['prompt'] for r in ranking]
    tokens = bm25s.tokenize(prompts, stopwords='en', show_progress=False).ids
    # Use string tokens so the index applies its own vocabulary.
    token_strings = bm25s.tokenize(prompts, stopwords='en', show_progress=False, return_ids=False)
    assert len(tokens) == len(token_strings) == 1000
    bm25_predictions = []
    for token_list in token_strings:
        scores = index.get_scores(token_list)
        top = np.argsort(-scores, kind='stable')[:K]
        bm25_predictions.append(vote([labels[i] for i in top], np.ones(K)))
    del index

    print('Training supervised baselines', flush=True)
    x_rows = np.repeat(np.arange(len(labels)), [len(group) for group in labels])
    y_cols = np.array([class_idx[label] for group in labels for label in group])
    sample_weights = 1 / np.array([len(labels[i]) for i in x_rows])
    targets = np.zeros((len(labels), len(classes)), dtype=np.float32)
    targets[x_rows, y_cols] = sample_weights
    dense_model = Ridge(alpha=1).fit(dense, targets)
    dense_predictions = classes[dense_model.predict(query).argmax(axis=1)]
    del targets, dense, dense_model
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, dtype=np.float32)
    features = vectorizer.fit_transform(texts)
    text_model = MultinomialNB(alpha=1).fit(features[x_rows], y_cols, sample_weight=sample_weights)
    text_predictions = classes[text_model.predict(vectorizer.transform(prompts))]

    predictions = []
    for i, record in enumerate(ranking):
        neighbors = record['neighbors'][:K]
        groups = [r['cwe_ids'] for r in neighbors]
        scores = np.array([r['similarity'] for r in neighbors])
        predictions.append({
            'row_index': i, 'nearest_neighbor': record['predictions']['1'],
            'majority': vote(groups, np.ones(K)), 'cosine_sum': record['predictions']['20'],
            'inverse_distance': vote(groups, 1 / np.maximum(1 - scores, 1e-6)),
            'softmax': vote(groups, softmax_weights(scores, temperature)),
            'bm25': bm25_predictions[i], 'embedding_ridge': dense_predictions[i],
            'text_nb': text_predictions[i],
        })
    ledger = OUT / 'predictions.jsonl'
    with ledger.open('x') as stream:
        for row in predictions:
            stream.write(json.dumps(row) + '\n')
    save('prediction_freeze.json', {'sha256': sha(ledger), 'rows': len(predictions), 'gold_opened': False})

    with (ROOT / 'data/cti-bench/data/cti-rcm.tsv').open() as stream:
        gold = list(csv.DictReader(stream, delimiter='\t'))
    assert len(gold) == len(predictions) and [r['Prompt'] for r in gold] == prompts
    truth = ['CWE-' + str(int(r['GT'].split('-')[1])) for r in gold]
    report = {}
    for method in predictions[0].keys() - {'row_index'}:
        predicted = [row[method] for row in predictions]
        correct = sum(a == b for a, b in zip(truth, predicted, strict=True))
        report[method] = {'correct': correct, 'accuracy_percent': correct / 10,
                          'macro_f1_percent': 100 * f1_score(truth, predicted, average='macro', zero_division=0)}
    assert [report[m]['correct'] for m in ('nearest_neighbor', 'majority', 'cosine_sum')] == [686, 757, 763]
    save('report.json', {'temperature': temperature, 'methods': report})
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    with (OUT / 'run.lock').open('a') as lock, threadpool_limits(limits=4):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
