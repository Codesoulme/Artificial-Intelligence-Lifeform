"""Verify and save a widened local language candidate, never promote it."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spora.conversation import restore_conversation, generate
from spora.conversation_growth import widen_conversation
from spora.growing import load
from spora.language_training import frozen_digest, interface_state, atomic_save
from scripts.prepare_conversation_v2 import ROOT, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    args = p.parse_args()
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError(out)
    plan = json.loads((ROOT / args.plan).read_text('utf-8'))
    path = ROOT / args.checkpoint
    original_hash = sha(path)
    if sha(ROOT / plan['parent']) != plan['parent_sha256']:
        raise ValueError('Individual changed')
    torch.set_num_threads(2); torch.manual_seed(231301)
    organism, state, _ = load(ROOT / plan['parent'], args.device)
    before = frozen_digest(organism, state)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    vocab = ROOT / plan['data'] / 'tokenizer.json'
    source = restore_conversation(organism, payload, plan['parent_sha256'], sha(vocab), args.device)
    grown = widen_conversation(source, seed=231301, noise_std=.001)
    tokenizer = Tokenizer.from_file(str(vocab))
    pad, bos, eos, user, assistant = (tokenizer.token_to_id(s) for s in
                                    ('<|pad|>', '<|bos|>', '<|eos|>', '<|user|>', '<|assistant|>'))
    prompts = ['你好。', '请介绍一种常见蔬菜。', '我想去散步，要带什么？',
               '请把今天要做的事分成两步。', '为什么热水会变凉？', '请解释诚实的含义。']
    probes = []
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            prefix = [bos, user] + tokenizer.encode(prompt).ids + [eos, assistant]
            tokens = torch.tensor([prefix], device=args.device)
            a = source(tokens, state); b = grown(tokens, state)
            left = generate(source, state, prefix, eos, budget=24, forbidden=(pad, bos, user, assistant))
            right = generate(grown, state, prefix, eos, budget=24, forbidden=(pad, bos, user, assistant))
            probes.append(dict(probe=i, max_logit_difference=float((a-b).abs().max()),
                               logits_close=bool(torch.allclose(a, b, atol=5e-5, rtol=5e-5)),
                               argmax_equal=bool(torch.equal(a.argmax(-1), b.argmax(-1))),
                               generation_equal=left == right))
        for length in (16, 128, 512):
            tokens = torch.randint(source.config.vocabulary, (1, length), device=args.device)
            a = source(tokens, state); b = grown(tokens, state)
            probes.append(dict(random_length=length, max_logit_difference=float((a-b).abs().max()),
                               logits_close=bool(torch.allclose(a, b, atol=5e-5, rtol=5e-5)),
                               argmax_equal=bool(torch.equal(a.argmax(-1), b.argmax(-1)))))
    preserved = before == frozen_digest(organism, state)
    unchanged = original_hash == sha(path)
    report = dict(source_sha256=original_hash, source_steps=payload['steps'], source_targets=payload['targets'],
                  source_parameters=sum(p.numel() for p in source.trainable_parameters()),
                  grown_parameters=sum(p.numel() for p in grown.trainable_parameters()),
                  config=asdict(grown.config), device=args.device, probes=probes,
                  individual_exact=preserved, source_unchanged=unchanged,
                  source_code_sha256={s: sha(ROOT / s) for s in ('spora/conversation_growth.py', 'scripts/grow_conversation.py', 'spora/conversation.py')},
                  scope='Finite numerical preservation probes, not proof of improved learning or chat ability. No optimizer inherited. No training updates.')
    passed = preserved and unchanged and all(r['logits_close'] and r['argmax_equal'] and r.get('generation_equal', True) for r in probes)
    report['passed'] = passed
    out.mkdir(parents=True)
    if passed:
        snapshot = dict(format='spora-conversation-v1', organism_sha256=plan['parent_sha256'],
                        tokenizer_sha256=sha(vocab), config=asdict(grown.config), state=interface_state(grown),
                        steps=payload['steps'], targets=payload['targets'], growth_parent_sha256=original_hash,
                        growth_recipe=dict(width_factor=2, seed=231301, noise_std=.001))
        atomic_save(snapshot, out / 'language-grown.pt')
        report['grown_sha256'] = sha(out / 'language-grown.pt')
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    if not passed:
        raise SystemExit('Growth validation failed; no candidate saved')


if __name__ == '__main__':
    main()
