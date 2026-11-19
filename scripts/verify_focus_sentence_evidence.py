"""Check frozen sentence generation with matched and exchanged event evidence."""

import argparse
import gc
import json
from pathlib import Path

from deepseek_focus_sentence_bootstrap import OUT as BOOTSTRAP, answer
from deepseek_focus_rollout_gate import screen
from deepseek_focus_timing_gate import encode
from deepseek_focus_value_position_20261120 import (
    MODEL, ROOT, SOURCE, SOURCE_RUN, check, fingerprint, focused_memory_args,
    load, load_integrated, prefix, read, sha, torch, write,
)


@torch.inference_mode()
def main(out):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out.exists():
        raise FileExistsError(out)
    if torch.cuda.mem_get_info()[0] < 3.7 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory for the frozen model')
    torch.set_num_threads(2)
    manifest = check()
    reference = read(BOOTSTRAP / 'result.json')
    plan = read(BOOTSTRAP / 'plan.json')
    adapter_path = BOOTSTRAP / 'adapter.pt'
    if (sha(BOOTSTRAP / 'plan.json') != reference['plan_sha256']
            or sha(adapter_path) != reference['adapter_sha256']
            or sha(SOURCE_RUN / 'plan.json') != plan['source_plan_sha256']):
        raise ValueError('Source artifact digest mismatch')
    fixtures = {f['name']: f for f in manifest['fixtures'] if f['name'] in plan['held_fixtures']}
    comparisons = {(r['fixture'], r['question']): r for r in reference['rows'] if r['mode'] == 'rollout_gate'}
    if set(comparisons) != {(f, q) for f in fixtures for q in plan['held_questions']}:
        raise ValueError('Incomplete reference coverage')
    words = {fixture['expected_word'] for fixture in manifest['fixtures']}
    out.mkdir()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, dtype=torch.float16,
        attn_implementation='sdpa', device_map={'': 'cuda'}, low_cpu_mem_usage=True).eval()
    decoder, parent, old = load_integrated(SOURCE, base, MODEL, individual_device='cuda')
    del parent, old
    decoder.eval().requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()
    adapter = torch.load(adapter_path, weights_only=True)
    columns = torch.tensor(adapter['columns'], device='cuda')
    hidden_size = base.config.hidden_size
    style = torch.nn.Sequential(torch.nn.Linear(hidden_size, 24), torch.nn.Tanh(),
                                torch.nn.Linear(24, len(columns)))
    gate = torch.nn.Linear(hidden_size, 1)
    style.load_state_dict(adapter['style'])
    gate.load_state_dict(adapter['gate'])
    style.eval().to('cuda')
    gate.eval().to('cuda')
    probe_question = next(row['text'] for row in manifest['questions'] if row['name'] == 'previous_word')
    probe = torch.tensor([encode(tokenizer, prefix(tokenizer, probe_question))], device='cuda')
    baseline = decoder(input_ids=probe, use_cache=False).logits[0, -1].float()

    def controls(fixture):
        organism, state, _ = load(SOURCE_RUN / (fixture['name'] + '.pt'), 'cuda')
        if fingerprint(organism, state) != fixture['state_digest']:
            raise ValueError('Fixture restore mismatch')
        options, selected = focused_memory_args(organism, state)
        if (selected['key'], selected['event_time']) != (fixture['key'], fixture['event']):
            raise ValueError('Event selection mismatch')
        return organism, state, dict(memory_state=state, **options)

    evidence = {}
    for name, fixture in fixtures.items():
        _, _, options = controls(fixture)
        evidence[name] = decoder(input_ids=probe, use_cache=False, **options).logits[0, -1].float() - baseline
        if not torch.isfinite(evidence[name]).all():
            raise ValueError('Non-finite event evidence')
    results = []
    with (out / 'progress.jsonl').open('x', encoding='utf8') as stream:
        for name, fixture in fixtures.items():
            other = next(f for f in fixtures.values() if f['key'] == fixture['key']
                         and f['world'] == fixture['world'] and f['version'] != fixture['version'])
            organism, state, options = controls(fixture)
            for question in plan['held_questions']:
                prompt = encode(tokenizer, prefix(tokenizer, question))
                normal = comparisons[name, question]
                replay, steps = answer(decoder, tokenizer, prompt, options, columns, style, gate, evidence[name])
                if replay != normal['response'] or [s['token'] for s in steps] != [s['token'] for s in normal['steps']]:
                    raise ValueError('Independent matched replay differs')
                swapped, swapped_steps = answer(decoder, tokenizer, prompt, options, columns, style,
                                               gate, evidence[other['name']])
                trigger = next((i for i, step in enumerate(steps) if step['applied']), None)
                swapped_trigger = next((i for i, step in enumerate(swapped_steps) if step['applied']), None)
                if trigger != swapped_trigger or steps[:trigger] != swapped_steps[:swapped_trigger]:
                    raise ValueError('Swapped evidence changed the pre-trigger trajectory')
                row = {'fixture': name, 'other_fixture': other['name'], 'question': question,
                       'current_value': fixture['expected_word'], 'other_value': other['expected_word'],
                       'matched': replay, 'swapped': swapped, 'trigger': trigger,
                       'matched_current': screen(replay, fixture['expected_word'], words),
                       'swapped_current': screen(swapped, fixture['expected_word'], words),
                       'swapped_other': screen(swapped, other['expected_word'], words),
                       'swapped_tokens': [step['token'] for step in swapped_steps]}
                results.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            if fingerprint(organism, state) != fixture['state_digest']:
                raise ValueError('Inference changed fixture')
            print('Verified', name, len(results), flush=True)
    summary = {'rows': len(results), 'matched_replay_exact': len(results),
               'triggered': sum(row['trigger'] is not None for row in results),
               'matched_current': sum(row['matched_current'] for row in results),
               'swapped_current': sum(row['swapped_current'] for row in results),
               'swapped_other': sum(row['swapped_other'] for row in results),
               'correct_both_directions': sum(row['matched_current'] and row['swapped_other'] for row in results)}
    write(out / 'result.json', {'summary': summary, 'rows': results, 'adapter_sha256': sha(adapter_path),
                               'source_result_sha256': sha(BOOTSTRAP / 'result.json'),
                               'script_sha256': sha(Path(__file__))})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/deepseek-focus-sentence-evidence')
    main(parser.parse_args().out)
