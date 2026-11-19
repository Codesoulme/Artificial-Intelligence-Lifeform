"""Test a frozen memory score difference at a supplied sentence position."""

import argparse
import gc
import json
import time
from pathlib import Path

from deepseek_focus_value_position_20261120 import (
    MODEL, ROOT, SENTENCE_PREFIX, SOURCE, SOURCE_RUN, check, fingerprint, focused_memory_args,
    load, load_integrated, prefix, sha, torch, write,
)


@torch.inference_mode()
def main(out, transfer=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

    if out.exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient GPU memory for the frozen model')
    plan = check()
    torch.set_num_threads(2)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.float16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, old_state = load_integrated(SOURCE, base, MODEL, individual_device='cuda')
    del parent, old_state
    decoder.eval().requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()
    questions = {question['name']: question['text'] for question in plan['questions']}

    def ids(text):
        return tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids.to('cuda')

    probe = ids(prefix(tokenizer, questions['previous_word']))
    baseline = decoder(input_ids=probe, use_cache=False).logits[0, -1].float()
    if not torch.isfinite(baseline).all():
        raise ValueError('Non-finite baseline logits')

    class FirstTokenEvidence(LogitsProcessor):
        def __init__(self, delta, position, scale):
            self.delta = delta * scale
            self.position = position

        def __call__(self, input_ids, scores):
            if input_ids.shape[-1] == self.position:
                return scores + self.delta.to(scores)
            return scores

    settings = ('reader_only', 'evidence_0.5', 'evidence_1.0')
    openings = (SENTENCE_PREFIX,)
    if transfer:
        settings = ('reader_only', 'evidence_1.0', 'swapped_evidence')
        openings = ('这次经历中，此人最偏爱的饮品是', '记录里提到这位当事人爱喝', '按照那次的记录，他更喜欢')
    out.mkdir()
    write(out / 'plan.json', {
        'settings': settings, 'probes': [questions['previous_word'], questions['sentence']],
        'sentence_prefixes': openings, 'max_new_tokens': 16,
        'transfer_plan_sha256': sha(SOURCE_RUN / 'plan.json'),
        'script_sha256': sha(Path(__file__)),
        'method': 'Freeze the full-vocabulary short-query memory delta and apply it only at the first continuation token.',
        'scope': 'Sentence continuation with a supplied answer opening on development fixtures.',
    })
    paired = {(f['world'], f['key'], f['version']): f for f in plan['fixtures']}
    deltas = {}
    sentences = {opening: ids(prefix(tokenizer, questions['sentence']) + opening) for opening in openings}
    if transfer:
        for fixture in plan['fixtures']:
            organism, state, _ = load(SOURCE_RUN / (fixture['name'] + '.pt'), 'cuda')
            if fingerprint(organism, state) != fixture['state_digest']:
                raise ValueError('Fixture restore mismatch')
            controls, _ = focused_memory_args(organism, state)
            selected = decoder(input_ids=probe, use_cache=False, memory_state=state, **controls)
            deltas[fixture['name']] = selected.logits[0, -1].float() - baseline
            del selected, organism, state
    results = []
    with (out / 'progress.jsonl').open('x', encoding='utf-8') as progress:
        for fixture in plan['fixtures']:
            model, state, _ = load(SOURCE_RUN / (fixture['name'] + '.pt'), 'cuda')
            if fingerprint(model, state) != fixture['state_digest']:
                raise ValueError('Fixture restore mismatch')
            read_controls, event = focused_memory_args(model, state)
            if (event['key'], event['event_time']) != (fixture['key'], fixture['event']):
                raise ValueError('Selected a different event')
            controls = dict(memory_state=state, **read_controls)
            selected = decoder(input_ids=probe, use_cache=False, **controls).logits[0, -1].float()
            delta = selected - baseline
            if not torch.isfinite(delta).all():
                raise ValueError('Non-finite memory score difference')
            other = paired[fixture['world'], fixture['key'], 1 - fixture['version']]
            for opening, sentence in sentences.items():
                length = sentence.shape[-1]
                for setting in settings:
                    processors = []
                    if setting == 'swapped_evidence':
                        processors.append(FirstTokenEvidence(deltas[other['name']], length, 1.0))
                    elif setting != 'reader_only':
                        scale = float(setting.split('_')[-1])
                        processors.append(FirstTokenEvidence(delta, length, scale))
                    output = decoder.generate(input_ids=sentence, attention_mask=torch.ones_like(sentence),
                                              use_cache=False, do_sample=False, max_new_tokens=16,
                                              pad_token_id=tokenizer.eos_token_id, **controls,
                                              logits_processor=processors)
                    generated = output[0, length:].tolist()
                    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
                    expected = fixture['expected_word']
                    value = response.rstrip('。！!．. ')
                    row = dict(fixture=fixture['name'], opening=opening, setting=setting, expected=expected,
                               other_expected=other['expected_word'], response=response, generated_ids=generated,
                               exact_value=value == expected, exact_other=value == other['expected_word'],
                               ended=tokenizer.eos_token_id in generated)
                    results.append(row)
                    progress.write(json.dumps(row, ensure_ascii=False) + '\n')
            progress.flush()
            total = len(plan['fixtures']) * len(settings) * len(openings)
            print(f'Generated continuations for {fixture["name"]} ({len(results)}/{total})', flush=True)
            if fingerprint(model, state) != fixture['state_digest']:
                raise ValueError('Generation changed state')
    summary = {}
    for setting in settings:
        rows = [row for row in results if row['setting'] == setting]
        summary[setting] = {'exact': sum(row['exact_value'] for row in rows), 'total': len(rows),
                            'ended': sum(row['ended'] for row in rows),
                            'exact_other': sum(row['exact_other'] for row in rows)}
    write(out / 'result.json', dict(rows=results, summary=summary, seconds=time.perf_counter() - started,
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   script_sha256=sha(Path(__file__)), plan_sha256=sha(out / 'plan.json')))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/deepseek-focus-evidence-shift-20261120')
    parser.add_argument('--transfer', action='store_true')
    args = parser.parse_args()
    main(args.out, args.transfer)
