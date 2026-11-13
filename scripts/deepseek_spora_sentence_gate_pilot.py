"""Research-only learned answer-position gate over native memory evidence."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_spora_online_lora_memory_pilot import OnlineDelta
from scripts.deepseek_spora_sentence_lora_pilot import TRAIN_TEMPLATES, HELD_TEMPLATE, make_state
from spora.memory_output import MemoryOutputAdapter

SENTENCE = ROOT / 'runs/deepseek-spora-sentence-lora-pilot-20261114'
ADAPTER = ROOT / 'runs/deepseek-spora-memory-output-adapter-v2-20261111'
OUT = ROOT / 'runs/deepseek-spora-sentence-gate-pilot-20261114'
SEED = 83701


def checked(folder, name, field):
    path = folder / name
    digest = p.sha(path)
    if digest != json.loads((folder / 'result.json').read_text(encoding='utf-8'))[field]:
        raise ValueError('Candidate checksum mismatch')
    return torch.load(path, map_location='cpu', weights_only=True), path, digest


def gold_inputs(tokenizer, name, value, question):
    prefix = closed_think_inputs(tokenizer, question, 'cuda')
    text = f'{name}喜欢喝{p.WORDS[value]}。'
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    start = text.index(p.WORDS[value])
    positions = [i for i, (left, right) in enumerate(encoded['offset_mapping'])
                 if left < start + len(p.WORDS[value]) and right > start]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError('Invalid fact token alignment')
    ids = torch.tensor([encoded['input_ids'] + [tokenizer.eos_token_id]], device='cuda')
    return prefix, ids, positions[0]


def features(chat, state, index, value, question):
    prefix, target, positive = gold_inputs(chat.tokenizer, NAMES[index], value, question)
    ids = torch.cat((prefix['input_ids'], target), dim=1)
    args, evidence = p.runtime._prepare_memory_read(chat.organism, state, KEYS[index], 'current', None)
    with torch.inference_mode():
        result = chat.decoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                              output_hidden_states=True, return_dict=True, **args)
    begin = prefix['input_ids'].shape[1] - 1
    vectors = result.hidden_states[-1][0, begin:begin + target.shape[1]].float().cpu()
    labels = torch.zeros(target.shape[1])
    labels[positive] = 1
    return F.layer_norm(vectors, (vectors.shape[-1],)), labels, int(evidence.argmax(-1)[0])


def greedy(chat, state, index, question, gate, adapter, policy):
    prefix = closed_think_inputs(chat.tokenizer, question, 'cuda')
    ids = prefix['input_ids']
    args, evidence = p.runtime._prepare_memory_read(chat.organism, state, KEYS[index], 'current', None)
    with torch.inference_mode():
        bias = adapter(evidence).float()
    fired, gate_scores = False, []
    for step in range(20):
        with torch.inference_mode():
            output = chat.decoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                                  output_hidden_states=True, return_dict=True, **args)
            logits = output.logits[:, -1].float()
            hidden = output.hidden_states[-1][:, -1].float().cpu()
            hidden = F.layer_norm(hidden, (hidden.shape[-1],))
            probability = float(gate(hidden).sigmoid()[0, 0])
            gate_scores.append(probability)
            use_bias = (policy == 'first' and step == 0) or (policy == 'gate' and not fired and probability >= 0.5)
            if use_bias:
                logits = logits + bias
                fired = True
            token = logits.argmax(-1, keepdim=True)
        ids = torch.cat((ids, token), dim=1)
        if int(token[0, 0]) == chat.tokenizer.eos_token_id:
            break
    tail = ids[0, prefix['input_ids'].shape[1]:].tolist()
    return {'answer': chat.tokenizer.decode(tail, skip_special_tokens=True).strip(), 'token_ids': tail,
            'fired': fired, 'gate_scores': gate_scores, 'ended': chat.tokenizer.eos_token_id in tail}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    checkpoint_sha = p.sha(p.runtime.CHECKPOINT)
    sentence, sentence_path, sentence_sha = checked(SENTENCE, 'adapter.pt', 'adapter_sha256')
    output_adapter, output_path, output_sha = checked(ADAPTER, 'adapter.pt', 'adapter_sha256')
    if base_sha != p.WEIGHT_SHA or sentence['base_sha256'] != base_sha or sentence['checkpoint_sha256'] != checkpoint_sha or output_adapter['base_sha256'] != base_sha:
        raise ValueError('Candidate provenance mismatch')
    for layer in (6, 12):
        path = ROOT / 'runs/deepseek-spora-diverse-memory-replay-20261111' / f'layer-{layer}.pt'
        if output_adapter['candidate_sha256'][str(layer)] != p.sha(path):
            raise ValueError('Output adapter layer mismatch')
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='spora-sentence-gate-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        original_digest = p.frozen_digest(chat.organism, chat.state)
        for layer, weights in zip(sentence['layers'], sentence['adapters']):
            mlp = chat.base.model.layers[layer].mlp
            module = OnlineDelta(mlp.down_proj, sentence['rank'])
            result = module.load_state_dict(weights, strict=False)
            expected_missing = {'base.weight'} | ({'base.bias'} if module.base.bias is not None else set())
            if set(result.missing_keys) != expected_missing or result.unexpected_keys:
                raise ValueError('Sentence adapter state does not match its projection')
            mlp.down_proj = module.eval()
        adapter = MemoryOutputAdapter(**output_adapter['config']).to('cuda').eval()
        adapter.load_state_dict(output_adapter['state'], strict=True)
        train = [(i, v, t.format(name=NAMES[i])) for i in (0, 1) for v in (1, 2, 3, 6) for t in TRAIN_TEMPLATES]
        held = [(i, v, HELD_TEMPLATE.format(name=NAMES[i])) for i in (6, 7) for v in (4, 7)]
        states = {(i, v): make_state(chat, KEYS[i], v) for i, v, _ in train + held}
        state_digests = {key: p.frozen_digest(chat.organism, state) for key, state in states.items()}
        train_x, train_y = [], []
        for i, v, q in train:
            x, y, native = features(chat, states[i, v], i, v, q)
            if native != v:
                raise ValueError('Controlled memory read failed')
            train_x.append(x)
            train_y.append(y)
        x, y = torch.cat(train_x), torch.cat(train_y).unsqueeze(1)
        gate = nn.Linear(x.shape[1], 1)
        optimizer = torch.optim.AdamW(gate.parameters(), lr=0.002, weight_decay=0.01)
        pos_weight = torch.tensor([(y.numel() - int(y.sum())) / int(y.sum())])
        for _ in range(200):
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(gate(x), y, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
        gate.eval()
        rows = []
        for i, v, q in held:
            x_held, labels, native = features(chat, states[i, v], i, v, q)
            with torch.no_grad():
                probabilities = gate(x_held).sigmoid().squeeze(-1).tolist()
            outputs = {policy: greedy(chat, states[i, v], i, q, gate, adapter, policy)
                       for policy in ('none', 'first', 'gate')}
            row = {'name': NAMES[i], 'question': q, 'expected': p.WORDS[v], 'native_correct': native == v,
                   'gold_gate_labels': labels.tolist(), 'gold_gate_probabilities': probabilities, 'outputs': outputs}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
        if p.frozen_digest(chat.organism, chat.state) != original_digest or any(
                p.frozen_digest(chat.organism, state) != state_digests[key] for key, state in states.items()):
            raise ValueError('Protected state changed')
        gate_state = {k: v.detach().clone() for k, v in gate.state_dict().items()}
    for path, digest in ((p.MODEL / 'model.safetensors', base_sha), (p.runtime.CHECKPOINT, checkpoint_sha),
                         (sentence_path, sentence_sha), (output_path, output_sha)):
        if p.sha(path) != digest:
            raise ValueError('Protected file changed')
    OUT.mkdir()
    gate_path = OUT / 'gate.pt'
    torch.save({'format': 'spora-sentence-gate-pilot-v1', 'state': gate_state, 'base_sha256': base_sha,
                'checkpoint_sha256': checkpoint_sha, 'sentence_sha256': sentence_sha, 'output_sha256': output_sha}, gate_path)
    result = {'rows': rows, 'training_loss': float(loss.detach()), 'training_positives': int(y.sum()),
              'training_positions': y.numel(), 'threshold': 0.5, 'seed': SEED,
              'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'sentence_sha256': sentence_sha, 'output_sha256': output_sha, 'gate_sha256': p.sha(gate_path),
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Gold answer spans and numeric addresses are supplied during training and teacher-forced evaluation.',
                              'Small synthetic finite-value set and four held rows; people and values appeared elsewhere.',
                              'Gate threshold and loss are development choices; no independent calibration.',
                              'Greedy generation recomputes the full prefix, with no natural routing or unknown-state test.',
                              'Only first token of value is targeted and no general open-world fact extraction is demonstrated.']}
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
