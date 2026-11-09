"""Compare P223 and P228 under identical P223-generated development histories."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from spora.conversation import restore_conversation
from spora.growing import load
from spora.language_training import frozen_digest
from scripts.converse_research import prefix_tokens

ROOT = Path(__file__).resolve().parents[1]
INDIVIDUAL = ROOT/'checkpoints/individual-prechinese-v3.pt'
TOKENIZER = ROOT/'runs/conversation-data-v5/tokenizer.json'
PARENT = ROOT/'runs/p223-reviewed-mixture-pilot-20261109/checkpoint.pt'
CHILD = ROOT/'runs/p228-guarded-continuation-20261109/checkpoint.pt'
BASELINE = ROOT/'runs/p228-guarded-continuation-20261109/multiturn-000000.json'
EXPECTED = {
    'individual': '6269d7749148c95ed4653928f61642c8d93098defed649baab3f807c71494db9',
    'tokenizer': 'b3629001784896437c4aa9f279f6b4cdc5a534be191b537009e7293f971fda20',
    'parent': 'e5df3ba8c17c8fd2c5e0d50d687cab69a72136bb9fe0ca5bf7bda973298d9cd8',
    'child': 'ccedec48b44a8859707edc41c24cb1b5f97ab9bca39eecca6a6416a8b3d58371',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def score_logits(logits, targets, eos, forbidden, ended):
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.numel() or not targets.numel():
        raise ValueError('Mismatched scores and targets')
    if eos in forbidden or (ended and int(targets[-1]) != eos):
        raise ValueError('Invalid termination target')
    scores = logits.float().clone()
    scores[:, list(forbidden)] = -torch.inf
    if not torch.isfinite(scores[:, eos]).all() or not torch.isfinite(scores.gather(1, targets[:, None])).all():
        raise ValueError('Invalid target score')
    loss = F.cross_entropy(scores, targets, reduction='none')
    argmax = scores.argmax(-1)
    result = {'targets': targets.numel(), 'loss_sum': float(loss.sum()),
              'target_greedy_matches': int((argmax == targets).sum())}
    if ended:
        log_probs = F.log_softmax(scores[-1], dim=-1)
        result['eos_nll'] = float(-log_probs[eos])
        result['eos_rank'] = int((scores[-1] > scores[-1, eos]).sum())+1
        result['eos_greedy'] = bool(argmax[-1] == eos)
    return result


@torch.no_grad()
def run(output, device='cuda', child=CHILD, child_sha=EXPECTED['child']):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    child = Path(child)
    expected = {**EXPECTED, 'child': child_sha}
    inputs = {'individual': INDIVIDUAL, 'tokenizer': TOKENIZER, 'parent': PARENT, 'child': child}
    if any(sha(path) != expected[name] for name, path in inputs.items()):
        raise ValueError('Checkpoint or tokenizer digest mismatch')
    baseline_sha = sha(BASELINE)
    baseline = json.loads(BASELINE.read_text(encoding='utf8'))
    rows = baseline['rows']
    if len(rows) != 36 or len({(r['scenario'], r['turn']) for r in rows}) != 36:
        raise ValueError('Unexpected development baseline')
    torch.set_num_threads(2)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    eos = tokenizer.token_to_id('<|eos|>')
    forbidden = tuple(tokenizer.token_to_id(t) for t in ('<|pad|>', '<|bos|>', '<|user|>', '<|assistant|>'))
    organism, state, _ = load(INDIVIDUAL, device)
    identity = frozen_digest(organism, state)
    models = [restore_conversation(organism, torch.load(path, map_location='cpu', weights_only=True),
                                   expected['individual'], expected['tokenizer'], device)
              for path in (PARENT, child)]
    result, history, scenario = [], [], None
    for row in rows:
        if row['scenario'] != scenario:
            scenario, history = row['scenario'], []
        history.append(('user', row['prompt']))
        while len(prefix_tokens(tokenizer, history))+96 > models[0].config.context and len(history) > 1:
            history = history[2:]
        prefix = prefix_tokens(tokenizer, history)
        if len(prefix) != row['input_tokens']:
            raise ValueError('Reconstructed prefix differs in length')
        generated = row['tokens']
        if tokenizer.decode(generated, skip_special_tokens=False) != row['text']:
            raise ValueError('Baseline text and tokens differ')
        target = generated + ([eos] if row['ended_with_eos'] else [])
        if not target:
            raise ValueError('Empty nonterminating response')
        sequence = torch.tensor([prefix+target], device=device, dtype=torch.long)
        targets = torch.tensor(target, device=device, dtype=torch.long)
        measurements = []
        for model in models:
            logits = model(sequence[:, :-1], state)[0, len(prefix)-1:]
            measurements.append(score_logits(logits, targets, eos, forbidden, row['ended_with_eos']))
        result.append({'scenario': scenario, 'turn': row['turn'], 'ended': row['ended_with_eos'],
                       'parent': measurements[0], 'child': measurements[1]})
        history.append(('assistant', row['text']))
    if any(sha(path) != expected[name] for name, path in inputs.items()) or sha(BASELINE) != baseline_sha or frozen_digest(organism, state) != identity:
        raise AssertionError('Protected input or individual changed')
    terminated = [r for r in result if r['ended']]
    total = sum(r['parent']['targets'] for r in result)
    summary = {'turns': len(result), 'target_positions': total, 'terminated_turns': len(terminated),
               'parent_nll': sum(r['parent']['loss_sum'] for r in result)/total,
               'child_nll': sum(r['child']['loss_sum'] for r in result)/total,
               'parent_eos_nll': sum(r['parent']['eos_nll'] for r in terminated)/len(terminated),
               'child_eos_nll': sum(r['child']['eos_nll'] for r in terminated)/len(terminated),
               'parent_eos_top1': sum(r['parent']['eos_greedy'] for r in terminated),
               'child_eos_top1': sum(r['child']['eos_greedy'] for r in terminated)}
    document = {'status': 'development_diagnostic_not_acceptance', 'inputs': expected,
                'baseline_sha256': baseline_sha, 'summary': summary, 'rows': result,
                'limitations': ['Parent-generated histories and targets are not independent reference answers.',
                                'Teacher-forced scores describe fixed paths, not free-generation quality.',
                                'Cases are repeatedly used for development and cannot establish chat ability.']}
    with output.open('x', encoding='utf8') as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--child', default=str(CHILD))
    parser.add_argument('--child-sha', default=EXPECTED['child'])
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.device, args.child, args.child_sha), ensure_ascii=False))


if __name__ == '__main__':
    main()
