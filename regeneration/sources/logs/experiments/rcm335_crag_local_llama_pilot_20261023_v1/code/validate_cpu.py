"""Read-only strict checkpoint and full-question token-fit admission check."""
from pathlib import Path
import json,time
from transformers import AutoTokenizer,T5ForSequenceClassification
P=Path(__file__).resolve().parent.parent
t=AutoTokenizer.from_pretrained(P/'evaluator',local_files_only=True)
m,info=T5ForSequenceClassification.from_pretrained(P/'evaluator',local_files_only=True,output_loading_info=True)
assert not info.get('missing_keys') and not info.get('unexpected_keys') and not info.get('mismatched_keys') and not info.get('error_msgs'),str(info)
assert m.config.num_labels==1
assert t('Synthetic evaluator input')['input_ids']==t('Synthetic evaluator input',add_special_tokens=False)['input_ids']+[t.eos_token_id]
assert hasattr(t,'pad')
audit=[dict(row_index=q['row_index'],full_question_separator_tokens=len(t(q['prompt']+' [SEP] ',add_special_tokens=False)['input_ids'])) for q in map(json.loads,(P/'shared/queries.jsonl').open())]
assert len(audit)==335
for row in audit:
 n=row['full_question_separator_tokens'];covered=set()
 for off in range(0,n,192):
  covered.update(range(off,min(off+256,n)))
  if off+256>=n:break
 assert covered==set(range(n))
result=dict(passed=True,at=time.time(),inference_performed=False,device='CPU checkpoint validation only',model_class=type(m).__name__,loading={k:list(v) if isinstance(v,set) else v for k,v in info.items()},tokenizer_class=type(t).__name__,parameter_count=sum(x.numel() for x in m.parameters()),token_audit=audit,evaluator_question_window_tokens=256,question_window_stride=192,no_question_tokens_dropped=True)
(P/'evaluator_cpu_validation.json').write_text(json.dumps(result,indent=2)+'\n');print('Strict checkpoint loading, tokenizer and all335 full-question evaluator-window checks passed; no inference.')
