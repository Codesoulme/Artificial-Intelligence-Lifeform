"""Jointly compare host language and causal memory across layer variants."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_spora_sentence_lora_pilot import make_state

OUT = ROOT / 'runs/deepseek-spora-paired-language-memory-20261116.json'
ENTITIES = (0, 3)
VALUES = (1, 6)  # Explicit numeric values; language strings are evaluation labels.
TEMPLATES = ('{name}现在最喜欢喝什么饮料？', '请只回答{name}记忆里的饮品名称。')
ORDINARY = ('小明有三本书，又买了两本，送给朋友一本。他现在有几本书？',
            '你知道我今天早餐吃了什么吗？', '请只回复两个字：你好。',
            '我有点累，请给一个简单的休息建议。')


def generate(chat, question, args=None):
    prompt = closed_think_inputs(chat.tokenizer, question, 'cuda')
    options = dict(args or {})
    options['use_cache'] = False
    with torch.inference_mode():
        output = chat.decoder.generate(**prompt, **options,
                    max_new_tokens=64, do_sample=False,
                    pad_token_id=chat.tokenizer.pad_token_id, eos_token_id=chat.tokenizer.eos_token_id)
    tail = output[0, prompt['input_ids'].shape[1]:].tolist()
    return {'text': chat.tokenizer.decode(tail, skip_special_tokens=True).strip(),
            'token_ids': tail, 'ended': chat.tokenizer.eos_token_id in tail}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.manual_seed(68214)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base_sha, source_sha = p.sha(p.MODEL / 'model.safetensors'), p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    rows, ordinary = [], []
    with tempfile.TemporaryDirectory(prefix='spora-paired-language-memory-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        base = p.load_full_streamed().eval()
        originals = {i: base.model.layers[i] for i in (6, 12)}
        chat = p.runtime.load_chat(base_loader=lambda: base)
        replacements = {i: base.model.layers[i] for i in originals}
        source_digest = p.frozen_digest(chat.organism, chat.state)
        states = {(entity, value): make_state(chat, KEYS[entity], value)
                  for entity in ENTITIES for value in VALUES}
        state_digests = {k: p.frozen_digest(chat.organism, state) for k, state in states.items()}
        for variant, active in (('original_layers', ()), ('spora_layers', (6, 12))):
            for i in originals:
                base.model.layers[i] = replacements[i] if i in active else originals[i]
            for entity in ENTITIES:
                for template_index, template in enumerate(TEMPLATES):
                    question = template.format(name=NAMES[entity])
                    off = generate(chat, question)
                    for value in VALUES:
                        args, native = p.runtime._prepare_memory_read(chat.organism, states[entity, value],
                                                                      KEYS[entity], 'current', None)
                        if int(native.argmax(-1)[0]) != value:
                            raise ValueError('Native controlled memory read failed')
                        output = generate(chat, question, args)
                        expected = p.WORDS[value]
                        rows.append({'variant': variant, 'entity': entity, 'name': NAMES[entity],
                                     'template': template_index, 'question': question, 'value': expected,
                                     'memory': output, 'off': off,
                                     'exact': output['text'].strip('。.!！ ') == expected,
                                     'contains': expected in output['text']})
                print(json.dumps({'variant': variant, 'completed_entity': entity}), flush=True)
            args, _ = p.runtime._prepare_memory_read(chat.organism, states[ENTITIES[0], VALUES[0]],
                                                     KEYS[ENTITIES[0]], 'current', None)
            for question in ORDINARY:
                ordinary.append({'variant': variant, 'question': question,
                                 'off': generate(chat, question), 'forced_unrelated_memory': generate(chat, question, args)})
            print(json.dumps({'variant': variant, 'ordinary_complete': len(ORDINARY)}), flush=True)
        for i in originals:
            base.model.layers[i] = replacements[i]
        if p.frozen_digest(chat.organism, chat.state) != source_digest or any(
                p.frozen_digest(chat.organism, state) != state_digests[key] for key, state in states.items()):
            raise ValueError('Protected individual or controlled memory changed')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != source_sha:
        raise ValueError('Protected source changed')
    summary = {}
    for variant in ('original_layers', 'spora_layers'):
        group = [r for r in rows if r['variant'] == variant]
        paired = []
        for entity in ENTITIES:
            for template_index in range(len(TEMPLATES)):
                pair = [r for r in group if r['entity'] == entity and r['template'] == template_index]
                if len(pair) != len(VALUES):
                    raise ValueError('Missing counterfactual pair')
                paired.append(all(r['exact'] for r in pair))
        orows = [r for r in ordinary if r['variant'] == variant]
        summary[variant] = {'memory_exact': sum(r['exact'] for r in group), 'memory_cases': len(group),
                            'memory_contains': sum(r['contains'] for r in group),
                            'strict_pairs': sum(paired), 'pair_count': len(paired),
                            'forced_ordinary_changed': sum(r['off']['token_ids'] != r['forced_unrelated_memory']['token_ids'] for r in orows),
                            'ordinary_cases': len(orows)}
    report = {'summary': summary, 'memory_rows': rows, 'ordinary_rows': ordinary,
              'base_sha256': base_sha, 'checkpoint_sha256': source_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Finite numeric values and caller-selected addresses, not open text memory.',
                              'Same templates and values are prior development materials.',
                              'Forced unrelated memory is an intervention, not a measured routing error rate.',
                              'Exact short-answer metric can reject a correct sentence; raw outputs are saved.',
                              'No natural write, identity selection, multi-turn evaluation or independent restore.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
