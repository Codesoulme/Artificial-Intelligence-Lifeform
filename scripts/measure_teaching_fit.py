"""Training-example diagnostic, explicitly NOT a held-out chat evaluation."""
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer
from spora.growing import load
from spora.conversation import restore_conversation,generate
from spora.language_training import frozen_digest
from scripts.converse_research import prefix_tokens
from scripts.prepare_conversation_v2 import ROOT,sha

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--plan',default='configs/p23d-conversation.json');args=p.parse_args();out=ROOT/args.output
    if out.exists():raise FileExistsError(out)
    plan=json.loads((ROOT/args.plan).read_text('utf-8'));path=ROOT/plan['curriculum']/'foundations.jsonl';vocab=ROOT/plan['data']/'tokenizer.json'
    if sha(ROOT/plan['parent'])!=plan['parent_sha256']:raise ValueError('Individual changed')
    torch.set_num_threads(2);core,state,_=load(ROOT/plan['parent'],'cuda');before=frozen_digest(core,state)
    payload=torch.load(ROOT/args.checkpoint,map_location='cpu',weights_only=True);model=restore_conversation(core,payload,plan['parent_sha256'],sha(vocab),'cuda')
    tok=Tokenizer.from_file(str(vocab));eos=tok.token_to_id('<|eos|>');forbidden=tuple(tok.token_to_id(x) for x in ('<|pad|>','<|bos|>','<|user|>','<|assistant|>'))
    examples=[json.loads(line) for line in path.open(encoding='utf-8')];rows=[]
    for index in random.Random(231401).sample(range(len(examples)),12):
        messages=examples[index]['messages']
        for position in sorted(set((1,len(messages)-1))):
            history=[(m['role'],m['content']) for m in messages[:position]];prefix=prefix_tokens(tok,history)
            answer=generate(model,state,prefix,eos,96,forbidden);text=tok.decode(answer['tokens'],skip_special_tokens=False);expected=messages[position]['content']
            rows.append(dict(training_example=index,assistant_position=position,prompt=messages[position-1]['content'],text=text,expected=expected,
                exact=text==expected,**answer))
    result=dict(checkpoint_sha256=sha(ROOT/args.checkpoint),source_step=payload['steps'],training_file_sha256=sha(path),
        exact=sum(x['exact'] for x in rows),count=len(rows),old_individual_exact=before==frozen_digest(core,state),rows=rows,
        scope='Known TRAINING examples with gold prior messages. Measures memorization/conditional generation only; absolutely not normal chat or independent generalization.')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
