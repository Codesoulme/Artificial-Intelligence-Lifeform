"""Measure native memory-reader strength against counterfactual and ordinary output."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_spora_paired_language_memory_20261116 import generate
from scripts.deepseek_spora_sentence_lora_pilot import make_state

OUT = ROOT / 'runs/deepseek-spora-reader-strength-tradeoff-20261116.json'
STRENGTHS = (0.0, 0.25, 0.5, 1.0)
ENTITIES = (0, 3)
VALUES = (1, 6)
TEMPLATES = ('{name}现在最喜欢喝什么饮料？', '请只回答{name}记忆里的饮品名称。')
ORDINARY = ('小明有三本书，又买了两本，送给朋友一本。他现在有几本书？',
            '你知道我今天早餐吃了什么吗？', '请只回复两个字：你好。')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.manual_seed(68214)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base_sha, checkpoint_sha = p.sha(p.MODEL / 'model.safetensors'), p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    rows, ordinary = [], []
    with tempfile.TemporaryDirectory(prefix='spora-strength-tradeoff-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        base = p.load_full_streamed().eval()
        originals = {i: base.model.layers[i] for i in (6, 12)}
        chat = p.runtime.load_chat(base_loader=lambda: base)
        replacements = {i: base.model.layers[i] for i in originals}
        original_strength = chat.decoder.reader.strength
        if not isinstance(original_strength, (float, int)):
            raise ValueError('Reader has no explicit strength')
        for i in originals:
            base.model.layers[i] = originals[i]
        source_digest = p.frozen_digest(chat.organism, chat.state)
        states = {(entity, value): make_state(chat, KEYS[entity], value)
                  for entity in ENTITIES for value in VALUES}
        state_digests = {k: p.frozen_digest(chat.organism, state) for k, state in states.items()}
        forced_args, _ = p.runtime._prepare_memory_read(chat.organism, states[ENTITIES[0], VALUES[0]],
                                                        KEYS[ENTITIES[0]], 'current', None)
        off_outputs = {question: generate(chat, question) for question in ORDINARY}
        for strength in STRENGTHS:
            chat.decoder.reader.strength = strength
            for entity in ENTITIES:
                for ti, template in enumerate(TEMPLATES):
                    question = template.format(name=NAMES[entity])
                    for value in VALUES:
                        args, native = p.runtime._prepare_memory_read(chat.organism, states[entity, value],
                                                                      KEYS[entity], 'current', None)
                        if int(native.argmax(-1)[0]) != value:
                            raise ValueError('Native controlled read failed')
                        result = generate(chat, question, args)
                        expected = p.WORDS[value]
                        rows.append({'strength': strength, 'entity': entity, 'template': ti,
                                     'question': question, 'expected': expected, 'output': result,
                                     'exact': result['text'].strip('。.!！ ') == expected})
            for question in ORDINARY:
                output = generate(chat, question, forced_args)
                off = off_outputs[question]
                ordinary.append({'strength': strength, 'question': question, 'off': off, 'forced': output,
                                 'changed': output['token_ids'] != off['token_ids']})
                if strength == 0.0 and output['token_ids'] != off['token_ids']:
                    raise ValueError('Zero strength failed ordinary output parity')
            print(json.dumps({'strength': strength, 'memory_cases': len(ENTITIES)*len(VALUES)*len(TEMPLATES),
                              'ordinary_cases': len(ORDINARY)}), flush=True)
        chat.decoder.reader.strength = original_strength
        for i in originals:
            base.model.layers[i] = replacements[i]
        if p.frozen_digest(chat.organism, chat.state) != source_digest or any(
                p.frozen_digest(chat.organism, state) != state_digests[key] for key, state in states.items()):
            raise ValueError('Individual or controlled memory changed')
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected source changed')
    summary = {}
    for strength in STRENGTHS:
        group = [r for r in rows if r['strength'] == strength]
        pairs = [all(r['exact'] for r in group if r['entity'] == e and r['template'] == t)
                 for e in ENTITIES for t in range(len(TEMPLATES))]
        ordinary_group = [r for r in ordinary if r['strength'] == strength]
        summary[str(strength)] = {'memory_exact': sum(r['exact'] for r in group), 'memory_cases': len(group),
                                  'strict_pairs': sum(pairs), 'pair_count': len(pairs),
                                  'ordinary_changed': sum(r['changed'] for r in ordinary_group),
                                  'ordinary_cases': len(ordinary_group)}
    report = {'summary': summary, 'memory_rows': rows, 'ordinary_rows': ordinary,
              'original_reader_strength': original_strength, 'base_sha256': base_sha,
              'checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Prior development questions and finite numeric facts; direct oracle address.',
                              'Only original language layers; no learned context-dependent gate.',
                              'Forced unrelated memory is an intervention, not actual routing-error frequency.',
                              'Token output change is not equivalent to a quality or safety measurement.',
                              'No natural writes, identity routing, historical recall or long-term recovery.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
