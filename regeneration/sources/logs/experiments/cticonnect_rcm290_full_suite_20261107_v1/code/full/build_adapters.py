"""Snapshot the existing method implementations for the CTIConnect dataset."""
import ast
from pathlib import Path
import re
from common import CODE, SUITE

SOURCE = SUITE / 'code'
QUERY = "R/'logs/experiments/rcm1000_source_priority_all_models_20261017_v1/shared/queries.jsonl'"
MASK = "R/'analysis/ctibench1000_final_group_filters_20261107/source_supported_exclusions.json'"
SHARED = "R/'logs/experiments/rcm1000_source_priority_all_models_20261017_v1/shared'"


def load(name):
    source = (SOURCE / name).read_text()
    source = source.replace(QUERY, 'QUERY_PATH').replace(MASK, 'MASK')
    source = re.sub(r'(?<![\w])1000(?![\w])', '290', source)
    source = re.sub(r'(?<![\w])120238(?![\w])', '119889', source)
    source = source.replace("str(q['row_index'])", "q['question_id']")
    source = source.replace("dataset='cti-rcm1000'", "dataset='cticonnect-rcm290'")
    parsed = ast.parse(source)
    first = parsed.body[0]
    header_end = first.end_lineno if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) else 0
    lines = source.splitlines(keepends=True)
    lines.insert(header_end, 'from common import CODE, OUT, QUERY_PATH, MASK, CONNECT, questions, score_native\n')
    return ''.join(lines)


def write(name, source):
    assert QUERY not in source and MASK not in source
    assert "cti-rcm.tsv" not in source
    compile(source, str(CODE / name), 'exec')
    (CODE / name).write_text(source)


source = load('generate_queries.py').replace('from swcv.api import ChatClient', 'from api import ChatClient')
source = source.replace("queries=[json.loads(l) for l in Q.open()]", 'queries=questions()')
write('generate_queries.py', source)

source = load('query_retrieval.py').replace(SHARED, 'OUT / "inputs"')
source = source.replace("queries=[json.loads(l) for l in (Q/'queries.jsonl').open()]", 'queries=questions()')
source = source.replace("'question_id':str(i)", "'question_id':queries[i]['question_id']")
source = source.replace("q['cve_id']", "q['source_cve_id']")
start = source.index('original=np.array(')
end = source.index('torch.backends.cuda.matmul.allow_tf32', start)
source = source[:start] + "original=np.load(CONNECT/'query_embeddings.npy',mmap_mode='r');assert original.shape==(290,2560)\n" + source[end:]
source = source.replace("key=lambda x:int(x['question_id'])", "key=lambda x:x['question_id']")
write('query_retrieval.py', source)

write('agent_rank.py', load('agent_rank.py'))
write('sparse_service.py', load('sparse_service.py'))

for name in ['ircot_run.py', 'agent_run_storage_safe_20261108.py']:
    source = load(name)
    source = source.replace("queries=[json.loads(l) for l in (QUERY_PATH).open()]", 'queries=questions()')
    source = source.replace("'content':answer", "'prediction':answer")
    source = source.replace("'content':result.trace['raw_final_response']", "'prediction':result.trace['raw_final_response']")
    source = source.replace("'content':'','pipeline_failure'", "'prediction':'','pipeline_failure'")
    if name == 'ircot_run.py':
        source = source[:source.index("metrics=R/'analysis/cticonnect_style_rcm335_20261105")]
        source += 'score_native(out, path)\n'
        source = source.replace('import requests', 'import shutil\nimport requests', 1)
        source = source.replace("    response=requests.post('http://127.0.0.1:8480", "    if shutil.disk_usage(out).free<5*1024**3:raise SystemExit('Disk reserve reached')\n    response=requests.post('http://127.0.0.1:8480")
    else:
        source = source[:source.index("module=R/'analysis/cticonnect_style_rcm335_20261105")]
        source += 'score_native(O, path)\n'
        # The native aggregator reads usage from every call, including judge probes.
        source = source.replace("self.calls.append({'kind':'pairwise_logprob'", "usage=body.get('usage') or {}\n  self.calls.append({'prompt_tokens':usage.get('prompt_tokens',0),'completion_tokens':usage.get('completion_tokens',0),'kind':'pairwise_logprob'")
    write('agent_run.py' if name.startswith('agent') else name, source)

source = load('prepare_crag.py')
source = source.replace("(O/'naive_rag20_prepared.jsonl')", "(OUT/'inputs/full_naive_prepared.jsonl')")
source = source.replace("O=Path(__file__).resolve().parents[1]", 'O=OUT')
source = source.replace("key=lambda x:int(x['question_id'])", "key=lambda x:x['question_id']")
write('prepare_crag.py', source)

source = load('prepare_linear_native.py').replace(SHARED, "OUT/'inputs'")
source = source.replace("M=Path(__file__).resolve().parents[1]", 'M=OUT')
source = source.replace('/dev/shm/swcv_linearrag_filtered_20261107', '/dev/shm/swcv_linearrag_cticonnect_20261108')
source = source.replace("str(M/'code/linear_ner_worker.py')", "str(CODE/'linear_ner_worker.py')")
source = source.replace("q['cve_id']", "q['source_cve_id']")
source = source.replace("entities=sorted({e for values in pe.values() for e in values}|{e for values in qentities.values() for e in values});sentences=sorted(se)",
    "corpus_entities={e for values in pe.values() for e in values};assert corpus_entities=={e for values in se.values() for e in values}\nentities=sorted(corpus_entities|{e for values in qentities.values() for e in values});sentences=sorted(se)")
# Keep query-only embeddings for lookup but keep them out of the graph seed index.
source = source.replace("part=texts[b:b+256];writer.write_table", "part=texts[b:b+256];writer.write_table")
source = source.replace("schema=schema))\n if kind=='entity':", "schema=schema).filter(pa.array([kind!='entity' or t in corpus_entities for t in part])))\n if kind=='entity':")
source = source.replace("def question_ner(self,text):return set(qentities[text])", "def question_ner(self,text):return sorted(set(qentities[text]))")
source = source.replace("rag.index(passages);assert", "rag.index(passages);assert set(rag.entity_embedding_store.hash_ids)==set(rag.entity_hash_id_to_sentence_hash_ids);assert")
write('prepare_linear.py', source)
write('linear_ner_worker.py', load('linear_ner_worker.py'))
