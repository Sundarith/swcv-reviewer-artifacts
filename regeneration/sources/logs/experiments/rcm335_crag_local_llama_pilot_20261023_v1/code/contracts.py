"""Independent paper-based CRAG decisions; no upstream source distribution."""
import re

def action(scores,upper=.59,lower=-.99):
 if not scores:raise ValueError('No document scores')
 return 'correct' if max(scores)>=upper else ('incorrect' if max(scores)<lower else 'ambiguous')

def evidence_pool(decision,dense,lexical):
 if decision not in ('correct','incorrect','ambiguous'):raise ValueError(decision)
 candidates=dense if decision=='correct' else lexical if decision=='incorrect' else dense+lexical
 return list(dict.fromkeys(candidates))

def strips(text):
 """Group at most three complete sentences; token windows are applied separately."""
 sentences=[s for s in re.split(r'(?<=[.!?])\s+',text) if s.strip()]
 return [' '.join(sentences[i:i+3]) for i in range(0,len(sentences),3)]

def select_strips(scored,threshold=-.5,limit=5):
 # Stable original candidate/strip order breaks equal relevance scores.
 ranked=sorted(enumerate(scored),key=lambda x:(-x[1]['score'],x[0]))
 return [x for _,x in ranked if x['score']>threshold][:limit]
