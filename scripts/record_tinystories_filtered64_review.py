"""Record the bounded P164 qualitative review without copying story text."""
import hashlib,json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
QUEUE=ROOT/'runs/p164-tinystories-filtered64-review-20261109/queue.jsonl'
OUT=ROOT/'runs/p164-tinystories-filtered64-review-20261109/manual-review.json'
QUEUE_SHA='48269b2f93c2bbf4fce66c59b335b29ccbffb69badaae99c24011ed16318d2d8'
STATUS={0:'minor',1:'clean',2:'clean',3:'clean',4:'major',5:'clean',6:'minor',7:'clean',8:'clean',9:'clean',10:'minor',11:'minor',12:'minor',13:'major',14:'minor',15:'clean',16:'clean',17:'clean',18:'minor',19:'minor',20:'major',21:'clean',22:'minor',23:'clean',24:'minor',25:'clean',26:'minor',27:'clean',28:'clean',29:'clean',30:'clean',31:'clean',32:'clean',33:'minor',34:'major',35:'clean',36:'clean',37:'clean',38:'minor',39:'minor',40:'minor',41:'clean',42:'clean',43:'clean',44:'minor',45:'minor',46:'clean',47:'clean',48:'clean',49:'clean',50:'minor',51:'major',52:'minor',53:'clean',54:'clean',55:'clean',56:'major',57:'clean',58:'clean',59:'minor',60:'major',61:'clean',62:'clean',63:'clean'}
REASONS={4:'儿童用药情节需要安全边界复核',13:'球的实体和归还方向出现意义错误',20:'儿童分药给玩具熊，存在不宜保留的用药示范',34:'洗手场景和辣味/用水表述明显损坏',51:'鸟用羊毛治疗伤腿，存在不宜模仿的健康暗示',56:'鸟与起重机实体混淆，救援动作语义受损',60:'悬崖采摘场景存在明显安全语境风险'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run():
 if OUT.exists():raise FileExistsError(OUT)
 if sha(QUEUE)!=QUEUE_SHA:raise ValueError('queue changed')
 rows=[json.loads(x) for x in QUEUE.read_text(encoding='utf8').splitlines()]
 if len(rows)!=64:raise ValueError('queue count')
 reviews=[]
 for row in rows:
  i=row['queue_index'];status=STATUS[i];reviews.append({'queue_index':i,'row_index_0based':row['row_index_0based'],'text_sha256':row['text_sha256'],'status':status,'issue':REASONS.get(i,'')})
 counts=Counter(x['status'] for x in reviews)
 result={'status':'conditional_story_pilot_review_not_training','queue_sha256':QUEUE_SHA,'reviewer':'single development-agent qualitative review; not human or blind independent evaluation','counts':dict(counts),'clean_or_minor':counts['clean']+counts['minor'],'major_limit':8,'suggested_gate_met':counts['clean']+counts['minor']>=56 and counts['major']<=8,'reviews':reviews,'training_started':False,'training_approval':False,'limitations':['One reviewer and one fixed first-shard queue cannot certify the corpus.','Conditional gate is for bounded pilot preparation only.','No model training or final evaluation was performed.']}
 OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');return result
if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
