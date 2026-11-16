"""Test if shared-joint candidate answers follow targeted state interventions.

Fact boundaries are oracle-provided. The closed-set counterfactual does not
establish open-world memory or autonomous writing.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch

from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_ttt_joint_writer_reader_pilot import fact_cache
from scripts.deepseek_spora_ttt_shared_joint_pilot import write_sequence, exact
from spora.research_ttt_ntp import TTTNTPWriteTargetAdapter, attach_read_adapters, install

SOURCE = ROOT / 'runs/deepseek-spora-ttt-shared-joint-random-prompts-20261116'
OUT = ROOT / 'runs/deepseek-spora-shared-joint-counterfactual-20261116.json'
PEOPLE = ('小林', '阿宁', '小雨', '阿哲')
BASE_VALUES = ('酸奶', '豆浆', '咖啡', '果汁')
CHANGED_VALUE = '豆浆'


def make_assignments():
    """Hold three facts fixed while changing only the first person's value."""
    baseline = tuple(zip(PEOPLE, BASE_VALUES))
    changed = ((PEOPLE[0], CHANGED_VALUE),) + baseline[1:]
    return baseline, changed


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    paths = {'base': base.MODEL / 'model.safetensors', 'candidate': SOURCE / 'candidate.pt',
             'report': SOURCE / 'result.json'}
    hashes = {key: base.sha(path) for key, path in paths.items()}
    report = json.loads(paths['report'].read_text(encoding='utf-8'))
    payload = torch.load(paths['candidate'], map_location='cpu', weights_only=True)
    if (hashes['base'] != base.WEIGHT_SHA or hashes['candidate'] != report['candidate_sha256']
            or payload['base_sha256'] != hashes['base']):
        raise ValueError('Candidate provenance mismatch')
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    model = load_full_streamed().eval()
    wrappers = install(model, tuple(payload['layers']))
    readers = attach_read_adapters(wrappers, rank=payload['read_rank'])
    writers = {layer: TTTNTPWriteTargetAdapter(wrappers[layer].base.out_features,
                                                rank=payload['write_rank']).to('cuda')
               for layer in wrappers}
    for layer in wrappers:
        readers[layer].load_state_dict(payload['readers'][str(layer)], strict=True)
        writers[layer].load_state_dict(payload['writers'][str(layer)], strict=True)
    model.requires_grad_(False)
    for writer in writers.values():
        writer.requires_grad_(False)
    baseline, changed = make_assignments()
    caches = {item: fact_cache(model, tokenizer, wrappers,
                               f'{item[0]}喜欢喝{item[1]}。', item[1])
              for item in set(baseline + changed)}
    states = {label: write_sequence(wrappers, [caches[item] for item in records], writers,
                                    differentiable=False)
              for label, records in (('baseline', baseline), ('changed', changed))}
    rows = []
    for index, name in enumerate(PEOPLE):
        question = f'请只回答{name}喜欢喝什么饮料。'
        outputs = {label: pilot.generate(model, tokenizer, question, state)
                   for label, state in states.items()}
        outputs['memory_off'] = pilot.generate(model, tokenizer, question)
        row = {'name': name, 'target': index == 0, 'expected_baseline': baseline[index][1],
               'expected_changed': changed[index][1], 'outputs': outputs,
               'baseline_exact': exact(outputs['baseline']['answer'], baseline[index][1]),
               'changed_exact': exact(outputs['changed']['answer'], changed[index][1]),
               'same_tokens': outputs['baseline']['token_ids'] == outputs['changed']['token_ids']}
        rows.append(row)
    ordinary = []
    for question in ('你好，请简单介绍自己。', '请写一句鼓励学习的话。'):
        off = pilot.generate(model, tokenizer, question)
        on = pilot.generate(model, tokenizer, question, states['baseline'])
        ordinary.append({'question': question, 'off': off, 'on': on,
                         'same_tokens': off['token_ids'] == on['token_ids']})
    if any(base.sha(path) != hashes[key] for key, path in paths.items()):
        raise ValueError('Protected input changed')
    summary = {'target_both_correct': rows[0]['baseline_exact'] and rows[0]['changed_exact'],
               'target_changed': not rows[0]['same_tokens'],
               'nontarget_unchanged': sum(row['same_tokens'] for row in rows[1:]),
               'ordinary_same_as_off': sum(row['same_tokens'] for row in ordinary)}
    result = {'summary': summary, 'rows': rows, 'ordinary': ordinary, 'source_sha256': hashes,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Previously used four synthetic people, values and fixed phrasing.',
                              'Fact boundary and target intervention supplied by the experiment.',
                              'One change, one seed, no sequential correction or unknown test.',
                              'Separate-process candidate load; fast state is reconstructed, not reloaded.',
                              'Two ordinary prompts are not an open-dialogue regression suite.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
