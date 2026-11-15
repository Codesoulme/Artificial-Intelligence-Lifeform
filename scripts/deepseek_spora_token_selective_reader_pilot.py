"""Train per-token internal read gates and audit isolated free generation."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch.nn import functional as F
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_spora_relation_hard_negative_pilot import make_rows
from scripts.deepseek_spora_paired_language_memory_20261116 import generate
from scripts.deepseek_spora_sentence_lora_pilot import make_state
from spora.research_selective_reader import SelectiveMemoryReader

OUT = ROOT / 'runs/deepseek-spora-token-selective-reader-20261116'
SEED = 85317
TEMPLATES = ('{name}现在最喜欢喝什么饮料？', '请只回答{name}记忆里的饮品名称。')
ENTITIES, VALUES = (0, 3), (1, 6)
ORDINARY = ('小明有三本书，又买了两本，送给朋友一本。他现在有几本书？',
            '你知道我今天早餐吃了什么吗？', '请只回复两个字：你好。',
            '小林喜欢的饮料是什么味道？', '不要告诉我小林喜欢的饮料。')


def collect(reader, decoder, tokenizer, questions, memory_args):
    captured = {}

    def capture(module, args):
        hidden, _, site = args[:3]
        if site in captured:
            raise ValueError('Duplicate reader call for site')
        captured[site] = hidden[0, -1].detach().float().cpu()

    result = []
    hook = reader.register_forward_pre_hook(capture)
    try:
        for index, question in enumerate(questions):
            captured.clear()
            prompt = closed_think_inputs(tokenizer, question, 'cuda')
            options = dict(memory_args)
            options['use_cache'] = False
            with torch.inference_mode():
                decoder(**prompt, **options)
            if set(captured) != {0, 1}:
                raise ValueError('Expected exactly two reader sites')
            result.append(torch.stack((captured[0], captured[1])))
            if (index + 1) % 32 == 0:
                print(json.dumps({'features': index + 1, 'total': len(questions)}), flush=True)
    finally:
        hook.remove()
    return torch.stack(result)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base_sha, checkpoint_sha = p.sha(p.MODEL / 'model.safetensors'), p.sha(p.runtime.CHECKPOINT)
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    examples = make_rows()
    with tempfile.TemporaryDirectory(prefix='spora-selective-reader-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        base = p.load_full_streamed().eval()
        originals = {i: base.model.layers[i] for i in (6, 12)}
        chat = p.runtime.load_chat(base_loader=lambda: base)
        replacements = {i: base.model.layers[i] for i in originals}
        for i in originals:
            base.model.layers[i] = originals[i]
        original_digest = p.frozen_digest(chat.organism, chat.state)
        states = {(entity, value): make_state(chat, KEYS[entity], value)
                  for entity in ENTITIES for value in VALUES}
        state_digests = {key: p.frozen_digest(chat.organism, state) for key, state in states.items()}
        args, _ = p.runtime._prepare_memory_read(chat.organism, states[ENTITIES[0], VALUES[0]],
                                                  KEYS[ENTITIES[0]], 'current', None)
        reader = chat.decoder.reader
        vectors = collect(reader, chat.decoder, chat.tokenizer, [r['question'] for r in examples], args)
        labels = torch.tensor([r['label'] == 'drink' for r in examples], dtype=torch.float32)
        train = torch.tensor([r['group'] in ('control_train', 'hard_train') for r in examples])
        wrapper = SelectiveMemoryReader(reader, vectors.shape[-1], threshold=0.5)
        features = F.layer_norm(vectors, (vectors.shape[-1],))
        gate_losses = []
        for site in (0, 1):
            gate = wrapper.gates[site]
            optimizer = torch.optim.AdamW(gate.parameters(), lr=0.0003, weight_decay=0.02)
            positive = labels[train].sum()
            pos_weight = (train.sum() - positive) / positive
            for _ in range(300):
                optimizer.zero_grad(set_to_none=True)
                logits = gate(features[train, site]).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, labels[train], pos_weight=pos_weight)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite gate loss')
                loss.backward()
                optimizer.step()
            gate_losses.append(float(loss.detach()))
        with torch.no_grad():
            for row_index, row in enumerate(examples):
                row['gate_scores'] = [float(wrapper.gates[site](features[row_index, site]).sigmoid()) for site in (0, 1)]
                row['both_allow'] = all(score >= wrapper.threshold for score in row['gate_scores'])
        wrapper = wrapper.to('cuda').eval()
        decoder_layers = [chat.decoder.base.model.layers[index] for index in chat.decoder.layer_indices]
        if any(layer._reader is not reader for layer in decoder_layers):
            raise ValueError('Unexpected reader installation')
        for layer in decoder_layers:
            object.__setattr__(layer, '_reader', wrapper)
        # The wrappers retain the same memory context; only their reader is changed.
        memory_rows, ordinary_rows = [], []
        for policy in ('baseline', 'selective'):
            wrapper.enabled = policy == 'selective'
            for entity in ENTITIES:
                for template_index, template in enumerate(TEMPLATES):
                    question = template.format(name=NAMES[entity])
                    for value in VALUES:
                        state = states[entity, value]
                        memory_args, native = p.runtime._prepare_memory_read(chat.organism, state,
                                                                            KEYS[entity], 'current', None)
                        if int(native.argmax(-1)[0]) != value:
                            raise ValueError('Controlled native memory mismatch')
                        output = generate(chat, question, memory_args)
                        expected = p.WORDS[value]
                        memory_rows.append({'policy': policy, 'entity': entity, 'template': template_index,
                                            'question': question, 'expected': expected, 'output': output,
                                            'exact': output['text'].strip('。.!！ ') == expected})
            for question in ORDINARY:
                off = generate(chat, question)
                forced = generate(chat, question, args)
                ordinary_rows.append({'policy': policy, 'question': question, 'off': off, 'forced': forced,
                                      'changed': off['token_ids'] != forced['token_ids']})
            print(json.dumps({'policy': policy, 'memory_cases': len(ENTITIES)*len(VALUES)*len(TEMPLATES),
                              'ordinary_cases': len(ORDINARY)}), flush=True)
        for layer in decoder_layers:
            object.__setattr__(layer, '_reader', reader)
        for i in originals:
            base.model.layers[i] = replacements[i]
        if p.frozen_digest(chat.organism, chat.state) != original_digest or any(
                p.frozen_digest(chat.organism, state) != state_digests[key] for key, state in states.items()):
            raise ValueError('Individual or memory changed')
        gate_state = [{k: v.detach().cpu().clone() for k, v in gate.state_dict().items()} for gate in wrapper.gates]
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected source changed')
    summaries = {}
    for group in ('control_train', 'hard_train', 'dev', 'test', 'held'):
        selected = [row for row in examples if row['group'] == group]
        positives = [r for r in selected if r['label'] == 'drink']
        negatives = [r for r in selected if r['label'] != 'drink']
        summaries[group] = {'positives': len(positives), 'positive_both_allow': sum(r['both_allow'] for r in positives),
                            'negatives': len(negatives), 'negative_both_allow': sum(r['both_allow'] for r in negatives)}
    generation = {}
    for policy in ('baseline', 'selective'):
        mem = [r for r in memory_rows if r['policy'] == policy]
        ordinary = [r for r in ordinary_rows if r['policy'] == policy]
        pairs = [all(r['exact'] for r in mem if r['entity'] == entity and r['template'] == ti)
                 for entity in ENTITIES for ti in range(len(TEMPLATES))]
        generation[policy] = {'memory_correct': sum(r['exact'] for r in mem), 'memory_cases': len(mem),
                              'strict_pairs': sum(pairs), 'pair_count': len(pairs),
                              'ordinary_changed': sum(r['changed'] for r in ordinary),
                              'ordinary_cases': len(ordinary)}
    OUT.mkdir()
    torch.save({'format': 'spora-token-selective-reader-research-v1', 'gates': gate_state,
                'threshold': 0.5, 'hidden_width': int(vectors.shape[-1]), 'base_sha256': base_sha,
                'checkpoint_sha256': checkpoint_sha}, OUT / 'gates.pt')
    report = {'feature_summary': summaries, 'generation': generation, 'examples': examples,
              'memory_rows': memory_rows, 'ordinary_rows': ordinary_rows, 'gate_losses': gate_losses,
              'gate_sha256': p.sha(OUT / 'gates.pt'), 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Last prompt token is supervised; the same gate is applied to every token.',
                              'Second-site training features use the ungated first site, creating distribution shift.',
                              'Finite drink labels and authored examples, explicit oracle address, forced memory.',
                              'No identity, unknown memory, general relations, causal write or independent restore.',
                              'Prior held groups have been explored; hard threshold is not calibrated.',
                              'This is an internal read-output gate, not an access-control or safety guarantee.']}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'feature_summary': summaries, 'generation': generation}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
