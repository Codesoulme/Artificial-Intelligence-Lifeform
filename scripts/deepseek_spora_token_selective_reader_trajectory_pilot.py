"""Train a token-trajectory memory gate with teacher-forced position labels."""
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
from scripts.deepseek_spora_paired_language_memory_20261116 import generate
from scripts.deepseek_spora_sentence_lora_pilot import make_state
from spora.research_selective_reader import SelectiveMemoryReader

OUT = ROOT / 'runs/deepseek-spora-token-selective-reader-trajectory-20261116'
SEED = 85318
ENTITIES, VALUES = (0, 3), (1, 6)
TEMPLATES = ('{name}现在最喜欢喝什么饮料？', '请只回答{name}记忆里的饮品名称。')
NEGATIVE = ('小林喜欢的饮料是什么味道？', '不要告诉我小林喜欢的饮料。',
            '小明有三本书，又买了两本，送给朋友一本。他现在有几本书？',
            '你知道我今天早餐吃了什么吗？', '请只回复两个字：你好。')


def collect_sequence(decoder, reader, tokenizer, question, target_text, memory_args, positive):
    prompt = closed_think_inputs(tokenizer, question, 'cuda')
    target_ids = tokenizer.encode(target_text, add_special_tokens=False) if target_text else []
    if target_ids:
        target_ids = target_ids + [tokenizer.eos_token_id]
    target = torch.tensor([target_ids], dtype=torch.long, device='cuda')
    ids = torch.cat((prompt['input_ids'], target), dim=1) if target_ids else prompt['input_ids']
    hidden_by_site = {}

    def capture(module, args):
        hidden, _, site = args[:3]
        hidden_by_site[site] = hidden[0].detach().float().cpu()

    hook = reader.register_forward_pre_hook(capture)
    try:
        options = dict(memory_args)
        options['use_cache'] = False
        with torch.inference_mode():
            decoder(input_ids=ids, attention_mask=torch.ones_like(ids), **options)
    finally:
        hook.remove()
    if set(hidden_by_site) != {0, 1}:
        raise ValueError('Expected both reader sites')
    labels = torch.zeros(ids.shape[1], dtype=torch.float32)
    if positive and target_ids:
        start = max(0, prompt['input_ids'].shape[1] - 1)
        labels[start:start + len(target_ids)] = 1.
    features = torch.stack((hidden_by_site[0], hidden_by_site[1]))
    return features, labels


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
    with tempfile.TemporaryDirectory(prefix='spora-token-trajectory-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        base = p.load_full_streamed().eval()
        originals = {i: base.model.layers[i] for i in (6, 12)}
        chat = p.runtime.load_chat(base_loader=lambda: base)
        for i in originals:
            base.model.layers[i] = originals[i]
        source_digest = p.frozen_digest(chat.organism, chat.state)
        states = {(entity, value): make_state(chat, KEYS[entity], value)
                  for entity in ENTITIES for value in VALUES}
        state_digests = {key: p.frozen_digest(chat.organism, state) for key, state in states.items()}
        reader = chat.decoder.reader
        training_features, training_labels = [], []
        for entity in ENTITIES:
            for value in VALUES:
                state = states[entity, value]
                args, native = p.runtime._prepare_memory_read(chat.organism, state, KEYS[entity], 'current', None)
                if int(native.argmax(-1)[0]) != value:
                    raise ValueError('Controlled native read failed')
                for template in TEMPLATES:
                    question = template.format(name=NAMES[entity])
                    target = f'{NAMES[entity]}喜欢喝{p.WORDS[value]}。'
                    features, labels = collect_sequence(chat.decoder, reader, chat.tokenizer, question, target, args, True)
                    training_features.append(features)
                    training_labels.append(labels)
        negative_args, _ = p.runtime._prepare_memory_read(chat.organism, states[ENTITIES[0], VALUES[0]],
                                                          KEYS[ENTITIES[0]], 'current', None)
        for question in NEGATIVE:
            features, labels = collect_sequence(chat.decoder, reader, chat.tokenizer, question, '', negative_args, False)
            training_features.append(features)
            training_labels.append(labels)
        features = torch.cat(training_features, dim=1)
        labels = torch.cat(training_labels)
        wrapper = SelectiveMemoryReader(reader, features.shape[-1], threshold=0.5)
        losses = []
        for site in (0, 1):
            gate = wrapper.gates[site]
            optimizer = torch.optim.AdamW(gate.parameters(), lr=0.0005, weight_decay=0.02)
            site_features = F.layer_norm(features[site], (features.shape[-1],))
            positives = labels.sum()
            pos_weight = (labels.numel() - positives) / positives
            for _ in range(500):
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(gate(site_features).squeeze(-1), labels,
                                                          pos_weight=pos_weight)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach()))
        wrapper = wrapper.to('cuda').eval()
        decoder_layers = [chat.decoder.base.model.layers[index] for index in chat.decoder.layer_indices]
        for layer in decoder_layers:
            object.__setattr__(layer, '_reader', wrapper)
        memory_rows, ordinary_rows = [], []
        for policy in ('baseline', 'selective'):
            wrapper.enabled = policy == 'selective'
            for entity in ENTITIES:
                for template_index, template in enumerate(TEMPLATES):
                    question = template.format(name=NAMES[entity])
                    for value in VALUES:
                        args, native = p.runtime._prepare_memory_read(chat.organism, states[entity, value],
                                                                      KEYS[entity], 'current', None)
                        output = generate(chat, question, args)
                        expected = p.WORDS[value]
                        memory_rows.append({'policy': policy, 'entity': entity, 'template': template_index,
                                            'question': question, 'expected': expected, 'output': output,
                                            'exact': output['text'].strip('。.!！ ') == expected})
            for question in NEGATIVE:
                off = generate(chat, question)
                args, _ = p.runtime._prepare_memory_read(chat.organism, states[0, VALUES[0]], KEYS[0], 'current', None)
                forced = generate(chat, question, args)
                ordinary_rows.append({'policy': policy, 'question': question, 'off': off, 'forced': forced,
                                      'changed': off['token_ids'] != forced['token_ids']})
            print(json.dumps({'policy': policy}), flush=True)
        for layer in decoder_layers:
            object.__setattr__(layer, '_reader', reader)
        for i in originals:
            base.model.layers[i] = originals[i]
        if p.frozen_digest(chat.organism, chat.state) != source_digest or any(
                p.frozen_digest(chat.organism, memory_state) != state_digests[key]
                for key, memory_state in states.items()):
            raise ValueError('Source state changed')
        gate_state = [{k: v.detach().cpu().clone() for k, v in gate.state_dict().items()} for gate in wrapper.gates]
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected source changed')
    summary = {}
    for policy in ('baseline', 'selective'):
        memory = [r for r in memory_rows if r['policy'] == policy]
        ordinary = [r for r in ordinary_rows if r['policy'] == policy]
        pairs = [all(r['exact'] for r in memory if r['entity'] == entity and r['template'] == template)
                 for entity in ENTITIES for template in range(len(TEMPLATES))]
        summary[policy] = {'memory_exact': sum(r['exact'] for r in memory), 'memory_cases': len(memory),
                           'strict_pairs': sum(pairs), 'pair_count': len(pairs),
                           'ordinary_changed': sum(r['changed'] for r in ordinary), 'ordinary_cases': len(ordinary)}
    OUT.mkdir()
    torch.save({'format': 'spora-token-selective-reader-trajectory-v1', 'gates': gate_state,
                'threshold': 0.5, 'hidden_width': int(features.shape[-1]), 'base_sha256': base_sha,
                'checkpoint_sha256': checkpoint_sha}, OUT / 'gates.pt')
    report = {'summary': summary, 'memory_rows': memory_rows, 'ordinary_rows': ordinary_rows,
              'gate_losses': losses, 'training_positions': int(labels.numel()),
              'positive_positions': int(labels.sum()), 'gate_sha256': p.sha(OUT / 'gates.pt'),
              'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Teacher-forced answer positions supervise a greedy free-generation gate.',
                              'Finite numeric addresses, controlled facts, authored negatives and prior development wording.',
                              'No identity, unknown memory, natural write, open relation or independent restore.',
                              'Gate remains research-only and is not access control.']}
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
