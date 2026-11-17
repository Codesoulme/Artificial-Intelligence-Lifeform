"""Analyze stored development scores without selecting a deployment threshold."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'paired':ROOT/'runs/deepseek-memory-need-hard-context-20261117/result.json',
 'hard':ROOT/'runs/deepseek-memory-need-hard-train-20261117/result.json',
}
def summarize(rows):
 if not rows or any(not math.isfinite(r['score']) or r['need'] not in (0,1) for r in rows):
  raise ValueError('Invalid scores')
 neg=[r['score'] for r in rows if r['need']==0]; pos=[r['score'] for r in rows if r['need']==1]
 if not neg or not pos: raise ValueError('Both classes required')
 auc=sum((p>n)+.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))
 thresholds=[-math.inf]+sorted(set(r['score'] for r in rows))+[math.inf]
 curve=[{'threshold':t if math.isfinite(t) else None,'extreme':('-inf' if t<0 else '+inf') if not math.isfinite(t) else None,
         'tp':sum(p>=t for p in pos),'fn':sum(p<t for p in pos),'fp':sum(n>=t for n in neg),'tn':sum(n<t for n in neg)} for t in thresholds]
 groups={}
 for r in rows: groups.setdefault(r.get('pair',r.get('case')),[]).append(r)
 strict=0
 for g in groups.values():
  a=[r['score'] for r in g if not r['need']];b=[r['score'] for r in g if r['need']]
  if len(a)!=1 or len(b)!=3: raise ValueError('Expected four-way cases')
  strict+=all(a[0]<v for v in b)
 return {'positives':len(pos),'negatives':len(neg),'auc':auc,'cases':len(groups),'within_case_strict_order':strict,
         'best_error_count':min(c['fp']+c['fn'] for c in curve),
         'best_complete_separation':any(c['fp']==0 and c['fn']==0 for c in curve),'curve':curve}
def main():
 out={}
 for name,path in SOURCES.items():
  data=json.loads(path.read_text(encoding='utf8')); rows=data.get('rows',data.get('test'))
  out[name]=summarize(rows)
 print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
