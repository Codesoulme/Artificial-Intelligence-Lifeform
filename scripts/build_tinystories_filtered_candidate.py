"""Build a 57-story filtered candidate with complete provenance; never trains."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
QUEUE=ROOT/'runs/p164-tinystories-filtered64-review-20261109/queue.jsonl'
REVIEW=ROOT/'runs/p164-tinystories-filtered64-review-20261109/manual-review.json'
SOURCE=ROOT/'runs/p28-stories-material/train-00000-of-00005.parquet'
TOKENIZER=ROOT/'runs/conversation-data-v5/tokenizer.json'
OUT=ROOT/'runs/p166-tinystories-filtered-candidate-20261109'
QUEUE_SHA='48269b2f93c2bbf4fce66c59b335b29ccbffb69badaae99c24011ed16318d2d8'
REVIEW_SHA=''
SOURCE_SHA='3c86b047b0063056f11b17a494c20a79ce8fda2f17bf4d06634a9d332b740302'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run():
 if OUT.exists():raise FileExistsError(OUT)
 if sha(QUEUE)!=QUEUE_SHA or sha(SOURCE)!=SOURCE_SHA:raise ValueError('input binding changed')
 review=json.loads(REVIEW.read_text(encoding='utf8'));excluded={x['queue_index'] for x in review['reviews'] if x['status']=='major'}
 rows=[json.loads(x) for x in QUEUE.read_text(encoding='utf8').splitlines() if json.loads(x)['queue_index'] not in excluded]
 if len(rows)!=57:raise ValueError('unexpected filtered count')
 OUT.mkdir(parents=False);candidate=OUT/'candidate.jsonl'
 with candidate.open('w',encoding='utf8',newline='\n') as f:
  for i,row in enumerate(rows):
   f.write(json.dumps({'record_id':f'tinystories_train_shard0_row_{row["row_index_0based"]}','source_repository':row['repository'],'source_revision':row['revision'],'source_path':row['path'],'source_row_index_0based':row['row_index_0based'],'source_text_sha256':row['text_sha256'],'review_status':row['review_status'],'text':row['text']},ensure_ascii=False,separators=(',',':'))+'\n')
 manifest={'status':'candidate_not_approved_for_training','repository':'Gabrui/multilingual_TinyStories','revision':rows[0]['revision'],'config':'chinese','split':'train','source_shard_sha256':SOURCE_SHA,'queue_sha256':QUEUE_SHA,'review_sha256':sha(REVIEW),'tokenizer_sha256':sha(TOKENIZER),'records':len(rows),'excluded_major_records':len(excluded),'candidate_sha256':sha(candidate),'license_declared':'CDLA-Sharing-1.0','attribution_required':['roneneldan/TinyStories','52AI/TinyStoriesZh','Gabrui/multilingual_TinyStories'],'training_started':False,'training_approval':False,'limitations':['Only one source shard and 57 stories.','Single development-agent review; no independent human or blind review.','Not enough scale for general Chinese pretraining or chat.']}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8');return manifest
if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
