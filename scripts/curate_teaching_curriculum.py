"""Remove visibly incomplete/ill-suited teacher answers before training."""
import collections,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tokenizers import Tokenizer
from scripts.prepare_conversation_v2 import ROOT,sha,encode_dialogue

def main():
    source=ROOT/'runs/teaching-curriculum-v2';out=ROOT/'runs/teaching-curriculum-v3';out.mkdir(exist_ok=False)
    manifest=json.loads((source/'manifest.json').read_text('utf-8'))
    for name,digest in manifest['files'].items():
        if sha(source/name)!=digest:raise ValueError('Source changed')
    tok=Tokenizer.from_file(str(ROOT/'runs/conversation-data-v4/tokenizer.json'));stats=collections.Counter()
    if sha(ROOT/'runs/conversation-data-v4/tokenizer.json')!=manifest['tokenizer_sha256']:raise ValueError('Tokenizer changed')
    for name in ('foundations.jsonl','foundations-tokens.jsonl'):(out/name).write_bytes((source/name).read_bytes())
    with (source/'teacher.jsonl').open(encoding='utf-8') as f,(out/'teacher.jsonl').open('x',encoding='utf-8') as raw,(out/'teacher-tokens.jsonl').open('x',encoding='utf-8') as encoded:
        for line in f:
            row=json.loads(line);answer=row['messages'][-1]['content'];prompt=row['messages'][0]['content'];why=None
            if not re.search(r'[。！？!?；;」』”）)\]】.]$',answer.rstrip()):why='unfinished_ending'
            elif len(answer)>350:why='long_answer'
            elif any(answer.count(a)!=answer.count(b) for a,b in [('（','）'),('“','”'),('「','」'),('【','】')]):why='unbalanced_delimiter'
            elif re.search(r'诊断|治疗|药物|用药|剂量|处方|诉讼|股票|股价|投资|汇率|房价|财务|疾病|病人',prompt):why='advanced_or_high_stakes_topic'
            if why:stats['removed_'+why]+=1;continue
            tokens,mask=encode_dialogue(tok,row['messages'])
            raw.write(line);encoded.write(json.dumps(dict(tokens=tokens,mask=mask,group=row['group']))+'\n')
            stats['teacher_dialogues']+=1;stats['teacher_targets']+=sum(mask)
    result=dict(parent_manifest_sha256=sha(source/'manifest.json'),tokenizer_sha256=manifest['tokenizer_sha256'],
        authored_sha256=manifest['authored_sha256'],source_sha256=sha(Path(__file__)),stats=stats,
        foundations=manifest['stats']['foundations_dialogues'],scope='Subset of isolated training-only material; punctuation filter cannot prove factual correctness. Original rejected files retained.',
        source_provenance=manifest['source_provenance'],files={p.name:sha(p) for p in out.iterdir()})
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(stats),flush=True)

if __name__=='__main__':main()
