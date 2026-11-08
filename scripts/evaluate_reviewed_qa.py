"""Read-only likelihood and generation for eight held-out reviewed QA rows."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from scripts.converse_research import prefix_tokens
from scripts.prepare_conversation_v2 import sha
from spora.conversation import generate, restore_conversation
from spora.growing import load
from spora.language_training import frozen_digest


def canonical(text):
    return re.sub(r'[\s，,。.!！?？:：;；]', '', text)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--course', default='runs/reviewed-aya-curriculum-p83-v2')
    args = p.parse_args()
    out = ROOT/args.output
    if out.exists():
        raise FileExistsError(out)
    course = ROOT/args.course
    manifest = json.loads((course/'manifest.json').read_text(encoding='utf8'))
    source = course/'aya-development.jsonl'
    if sha(source) != manifest['files'][source.name]:
        raise ValueError('Reviewed development changed')
    plan = json.loads((ROOT/args.plan).read_text(encoding='utf8'))
    parent, checkpoint = ROOT/plan['parent'], ROOT/args.checkpoint
    vocab = ROOT/plan['data']/'tokenizer.json'
    if sha(parent) != plan['parent_sha256'] or sha(vocab) != manifest['tokenizer_sha256']:
        raise ValueError('Parent or vocabulary changed')
    checkpoint_sha = sha(checkpoint)
    torch.set_num_threads(2)
    organism, state, _ = load(parent,'cuda')
    before = frozen_digest(organism,state)
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
    model = restore_conversation(organism,payload,plan['parent_sha256'],sha(vocab),'cuda')
    tokenizer = Tokenizer.from_file(str(vocab))
    eos = tokenizer.token_to_id('<|eos|>')
    forbidden = tuple(tokenizer.token_to_id(x) for x in ('<|pad|>','<|bos|>','<|user|>','<|assistant|>'))
    results, nll_sum, target_count = [], 0.0, 0
    with source.open(encoding='utf8') as f:
        for line in f:
            row = json.loads(line)
            question, answer = row['messages']
            if question['role'] != 'user' or answer['role'] != 'assistant':
                raise ValueError('Unexpected roles')
            prefix = prefix_tokens(tokenizer,[('user',question['content'])])
            target = tokenizer.encode(answer['content']).ids + [eos]
            if len(prefix)+len(target)-1 > model.config.context:
                raise ValueError('Evaluation example exceeds context')
            seq = torch.tensor([prefix+target],device='cuda')
            logits = model(seq[:,:-1],state)[0,len(prefix)-1:]
            loss = F.cross_entropy(logits,torch.tensor(target,device='cuda'),reduction='sum')
            nll_sum += float(loss)
            target_count += len(target)
            output = generate(model,state,prefix,eos,budget=96,forbidden=forbidden)
            text = tokenizer.decode(output['tokens'],skip_special_tokens=False)
            results.append(dict(group=row['group'],question=question['content'],reference=answer['content'],
                                output=text,exact=canonical(text)==canonical(answer['content']),
                                ended=output['ended_with_eos']))
    if not results or not target_count:
        raise ValueError('Empty development set')
    if before != frozen_digest(organism,state) or checkpoint_sha != sha(checkpoint):
        raise AssertionError('Protected state changed')
    report = dict(scope='Eight reviewed single-turn development records; exact match is a narrow string diagnostic, not a semantic or chat score. Prior-corpus semantic overlap is not ruled out.',
                  checkpoint_sha256=checkpoint_sha, source_sha256=sha(source),manifest_sha256=sha(course/'manifest.json'),
                  evaluator_sha256=sha(Path(__file__)),mean_target_nll=nll_sum/target_count,
                  targets=target_count,total=len(results),exact=sum(x['exact'] for x in results),rows=results,
                  individual_unchanged=True,checkpoint_unchanged=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('total','targets','mean_target_nll','exact')}))


if __name__=='__main__':
    main()
