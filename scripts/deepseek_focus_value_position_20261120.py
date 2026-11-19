"""Measure paired value likelihoods at two positions in the frozen decoder."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from scripts.deepseek_focus_transfer_20261118 import OUT as SOURCE_RUN, check
from scripts.deepseek_event_focus_generation_20261118 import MODEL, read, write, sha, prefix
from scripts.spora_event_focus_audit_20261118 import SOURCE, fingerprint
from spora.event_focus import focused_memory_args
from spora.growing import load
from spora.integrated_checkpoint import load_integrated

OUT = ROOT / 'runs/deepseek-focus-value-position-20261120'
SETTINGS = ('off', 'half', 'full')
SENTENCE_PREFIX = '在那次记录中，这个人喜欢喝'


def encode_candidate(tokenizer, start, word):
    target = tokenizer.encode(word, add_special_tokens=False)
    decode = lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    if not target or decode(start + target) != decode(start) + word:
        raise ValueError('Candidate does not decode as a literal continuation: ' + word)
    return target


def score_candidates(decoder, tokenizer, start, words, controls, last_only=False):
    outputs = []
    for word in words:
        target_ids = encode_candidate(tokenizer, start, word)
        if last_only:
            logp = []
            for index, token in enumerate(target_ids):
                ids = torch.tensor([start + target_ids[:index]], device='cuda', dtype=torch.long)
                output = decoder(input_ids=ids, use_cache=False, **controls)
                logp.append(float(output.logits[0, -1].float().log_softmax(dim=-1)[token]))
        else:
            ids = torch.tensor([start + target_ids[:-1]], device='cuda', dtype=torch.long)
            output = decoder(input_ids=ids, use_cache=False, **controls)
            scores = output.logits[0, len(start) - 1:].float().log_softmax(dim=-1)
            target = torch.tensor(target_ids, device='cuda', dtype=torch.long)
            logp = scores.gather(1, target[:, None])[:, 0].tolist()
        outputs.append({'word': word, 'token_ids': target_ids, 'token_logp': logp,
                        'sum_logp': sum(logp), 'mean_logp': sum(logp) / len(logp)})
    return outputs


def summarize(rows, positions, settings):
    summary = {}
    for position in positions:
        for setting in settings:
            subset = [r for r in rows if (r['position'], r['setting']) == (position, setting)]
            pairs = {(r['world'], r['key']) for r in subset}
            margins = [r['scores'][0]['mean_logp'] - r['scores'][1]['mean_logp'] for r in subset]
            sums = [r['scores'][0]['sum_logp'] - r['scores'][1]['sum_logp'] for r in subset]
            summary[f'{position}/{setting}'] = {
                'count': len(subset), 'mean_token_wins': sum(m > 0 for m in margins),
                'sequence_wins': sum(m > 0 for m in sums),
                'mean_margin': sum(margins) / len(margins),
                'pairs_both_mean_token_correct': sum(
                    all(m > 0 for r, m in zip(subset, margins) if (r['world'], r['key']) == pair)
                    for pair in pairs), 'pairs': len(pairs),
            }
    return summary


@torch.inference_mode()
def main(out, crossed=False, last_token=False):
    OUT = out
    if OUT.exists():
        raise FileExistsError(OUT)
    plan = check()
    transfer = read(SOURCE_RUN / 'result.json')
    if transfer['plan_sha256'] != sha(SOURCE_RUN / 'plan.json') or not transfer['complete']:
        raise ValueError('Incomplete transfer run')
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    free, _ = torch.cuda.mem_get_info()
    if free < 3.5 * 1024**3:
        raise RuntimeError('Insufficient GPU memory for the frozen model')
    started = time.perf_counter()
    settings = ('off', 'full', 'early', 'late') if crossed else SETTINGS
    if last_token:
        settings += ('last', 'last_late')
    questions = {q['name']: q['text'] for q in plan['questions']}
    positions = {'answer_start': ('previous_word', ''), 'sentence_value': ('sentence', SENTENCE_PREFIX)}
    if crossed:
        positions.update(short_with_prefix=('previous_word', SENTENCE_PREFIX),
                         sentence_without_prefix=('sentence', ''))
    OUT.mkdir()
    write(OUT / 'plan.json', {
        'transfer_plan_sha256': sha(SOURCE_RUN / 'plan.json'),
        'script_sha256': sha(Path(__file__)), 'settings': settings,
        'sentence_prefix': SENTENCE_PREFIX,
        'score': 'Compare standalone candidate tokens appended to a fixed prefix. Exact decoded text checked.',
        'positions': positions,
        'scope': 'Transfer development cases. Teacher-forced scoring only.'})
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.float16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    print('Base loaded', flush=True)
    decoder, parent, old = load_integrated(SOURCE, base, MODEL, individual_device='cuda')
    del parent, old
    decoder.eval().requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()
    original_strength = decoder.reader.strength
    print('Integrated model loaded', flush=True)

    fixtures = {(f['world'], f['key'], f['version']): f for f in plan['fixtures']}
    prompts = {name: prefix(tokenizer, questions[question]) + continuation
               for name, (question, continuation) in positions.items()}
    encodings = {name: tokenizer.encode(prompt, add_special_tokens=False) for name, prompt in prompts.items()}
    for start in encodings.values():
        for word in {f['expected_word'] for f in plan['fixtures']}:
            encode_candidate(tokenizer, start, word)
    rows = []
    original_forward = decoder.reader.forward
    active_setting = 'full'

    def forward_at_selected_site(hidden, matrices, site, address=None):
        if (active_setting == 'early' and site != 0) or (active_setting in ('late', 'last_late') and site != 1):
            return hidden
        if active_setting in ('last', 'last_late'):
            selected = original_forward(hidden[:, -1:], matrices, site, address)
            return torch.cat((hidden[:, :-1], selected), dim=1)
        return original_forward(hidden, matrices, site, address)

    with patch.object(decoder.reader, 'forward', forward_at_selected_site), \
            (OUT / 'progress.jsonl').open('x', encoding='utf-8') as progress:
        for fixture in plan['fixtures']:
            model, state, _ = load(SOURCE_RUN / (fixture['name'] + '.pt'), 'cuda')
            if fingerprint(model, state) != fixture['state_digest']:
                raise ValueError('Fixture restore mismatch')
            controls, selection = focused_memory_args(model, state)
            if (selection['key'], selection['event_time']) != (fixture['key'], fixture['event']):
                raise ValueError('Selected a different event')
            other = fixtures[fixture['world'], fixture['key'], 1 - fixture['version']]
            words = (fixture['expected_word'], other['expected_word'])
            for position, start in encodings.items():
                for setting in settings:
                    active_setting = setting
                    decoder.reader.strength = original_strength * (0.5 if setting == 'half' else 1)
                    kwargs = {} if setting == 'off' else {'memory_state': state, **controls}
                    scores = score_candidates(decoder, tokenizer, start, words, kwargs,
                                              last_only=active_setting in ('last', 'last_late'))
                    row = {'fixture': fixture['name'], 'world': fixture['world'], 'key': fixture['key'],
                           'version': fixture['version'], 'position': position, 'setting': setting,
                           'scores': scores}
                    rows.append(row)
                    progress.write(json.dumps(row, ensure_ascii=False) + '\n')
                progress.flush()
            total = len(fixtures) * len(positions) * len(settings)
            print(f'Scored {fixture["name"]} ({len(rows)}/{total})', flush=True)
            if fingerprint(model, state) != fixture['state_digest']:
                raise ValueError('Scoring changed the memory state')
            del model, state
    decoder.reader.strength = original_strength
    for position in prompts:
        for world, key, version in fixtures:
            if version == 1:
                continue
            pair = [r for r in rows if (r['position'], r['setting'], r['world'], r['key']) ==
                    (position, 'off', world, key)]
            if len(pair) != 2 or pair[0]['scores'] != list(reversed(pair[1]['scores'])):
                raise ValueError('Disabled scores depend on fixture')
    summary = summarize(rows, positions, settings)
    write(OUT / 'result.json', {'rows': rows, 'summary': summary, 'original_strength': original_strength,
                               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                               'seconds': time.perf_counter() - started})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--crossed', action='store_true')
    parser.add_argument('--last-token', action='store_true')
    args = parser.parse_args()
    main(args.out, args.crossed, args.last_token)
