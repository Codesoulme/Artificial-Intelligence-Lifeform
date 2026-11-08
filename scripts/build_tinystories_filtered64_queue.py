"""Build a pinned 64-story review queue from the local TinyStories train shard."""
import bisect,hashlib,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runs/p28-source-review/vendor'))
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/p28-source-review'
SOURCE=ROOT/'runs/p28-stories-material/train-00000-of-00005.parquet'
OUT=ROOT/'runs/p164-tinystories-filtered64-review-20261109'
EXPECTED='3c86b047b0063056f11b17a494c20a79ce8fda2f17bf4d06634a9d332b740302'
REV='7fb5f327b907318dd95877a366e288f1971b64c1'
SEED=20261109
MAJOR={x['row_index_0based'] for x in json.loads((BASE/'manual-review-32.json').read_text(encoding='utf8'))['annotations'] if x['severity']=='major'}
OLD={x['row_index_0based'] for x in json.loads((BASE/'train-samples-32.json').read_text(encoding='utf8'))['samples']}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mechanical(text):
 compact=''.join(text.split())
 return {'empty':not compact,'replacement':'\ufffd' in text,'literal_backslash_n':'\\n' in text,
         'too_long':False,'has_terminal':text.rstrip().endswith(('。','！','？','.','!','?')),
         'normalized_sha256':hashlib.sha256(''.join(text.split()).encode()).hexdigest()}
def run():
 if OUT.exists():raise FileExistsError(OUT)
 if sha(SOURCE)!=EXPECTED:raise ValueError('source changed')
 pf=pq.ParquetFile(SOURCE,pre_buffer=False,buffer_size=0);ends=[];total=0
 for i in range(pf.metadata.num_row_groups):total+=pf.metadata.row_group(i).num_rows;ends.append(total)
 candidates=[]
 for index in random.Random(SEED).sample(range(total), min(total,2000)):
  if index in MAJOR or index in OLD:continue
  candidates.append(index)
  if len(candidates)>=64:break
 if len(candidates)!=64:raise ValueError('not enough candidates')
 grouped={}
 for index in candidates:grouped.setdefault(bisect.bisect_right(ends,index),[]).append(index)
 rows=[];normalized=set();collateral=0
 for group,indices in grouped.items():
  table=pf.read_row_group(group,columns=['story'],use_threads=False);collateral+=table.num_rows;base=ends[group-1] if group else 0
  for index in indices:
   text=table.column(0)[index-base].as_py();checks=mechanical(text);checks['too_long']=len(text)>5120
   if checks['empty'] or checks['replacement'] or checks['literal_backslash_n'] or checks['too_long']:raise ValueError('mechanical defect in selected row')
   if checks['normalized_sha256'] in normalized:raise ValueError('duplicate selected row')
   normalized.add(checks['normalized_sha256']);rows.append({'queue_index':len(rows),'repository':'Gabrui/multilingual_TinyStories','revision':REV,'config':'chinese','split':'train','path':'chinese/train-00000-of-00005.parquet','row_index_0based':index,'row_group':group,'text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'mechanical':checks,'review_status':'pending_independent_review'})
  del table
 pf.close();rows.sort(key=lambda x:x['queue_index']);OUT.mkdir(parents=False);q=OUT/'queue.jsonl';q.write_text('\n'.join(json.dumps(x,ensure_ascii=False,separators=(',',':')) for x in rows)+'\n',encoding='utf8')
 result={'status':'pending_filtered64_story_review_not_training','repository':'Gabrui/multilingual_TinyStories','revision':REV,'source_sha256':EXPECTED,'records':len(rows),'excluded_known_major':len(MAJOR),'excluded_previous_sample':len(OLD),'unique_normalized':len(normalized),'queue_sha256':sha(q),'decoded_collateral_rows':collateral,'training_started':False,'training_approval':False,'test_split_read':False,'limitations':['Only the first train shard was sampled.','Excluding known defects does not establish whole-corpus quality.','Review is pending and no training artifact is produced.']}
 (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');return result
if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
