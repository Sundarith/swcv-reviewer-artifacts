import os,json,hashlib,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,torch
from transformers import AutoTokenizer,AutoModel
OUT=Path(__file__).resolve().parent
MODEL=Path('__LOCAL_ASSET_ROOT__/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b')
def run(gpu):
 torch.cuda.set_device(gpu);torch.backends.cuda.matmul.allow_tf32=False
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side='left')
 model=AutoModel.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(f'cuda:{gpu}').eval()
 rows=[json.loads(l) for l in (OUT/'description_inputs.jsonl').open()];indices=list(range(gpu,len(rows),2));vectors=np.lib.format.open_memmap(OUT/f'vectors_gpu{gpu}.npy',mode='w+',dtype='float32',shape=(len(indices),2560))
 done=0;lock=threading.Lock()
 def worker(worker_id):
  nonlocal done
  for k in range(worker_id,len(indices),8):
   i=indices[k];text=rows[i]['description'];tokens=tokenizer(text,return_tensors='pt',truncation=False)
   length=tokens['input_ids'].shape[1]
   assert length<=8192,f'Input {i} has {length} tokens; no clipping allowed'
   tokens={key:value.to(f'cuda:{gpu}') for key,value in tokens.items()}
   with torch.inference_mode():
    hidden=model(**tokens).last_hidden_state[:,-1,:].float();vec=torch.nn.functional.normalize(hidden,dim=1).cpu().numpy()[0]
   assert np.isfinite(vec).all();vectors[k]=vec
   with lock:
    done+=1
    if done%200==0:
     temp=OUT/f'progress_gpu{gpu}.tmp';temp.write_text(json.dumps({'completed':done,'total':len(indices),'workers':8}));temp.replace(OUT/f'progress_gpu{gpu}.json')
 with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,range(8)))
 vectors.flush();(OUT/f'gpu{gpu}_complete.json').write_text(json.dumps({'count':len(indices),'workers':8}))
if __name__=='__main__':run(int(os.environ['DESCRIPTION_GPU']))
