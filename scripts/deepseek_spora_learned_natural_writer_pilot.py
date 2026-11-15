"""Train a text-state writer and evaluate it through persistent phrase slots."""
import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch
from torch.nn import functional as F

from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_crossed_values_pilot as crossed
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.language_training import atomic_save
from spora.research_semantic_slot import install_sequence_scaled, use_slot_sequence
from spora.semantic_phrase_slot_runtime import NaturalSemanticPhraseMemory
from spora.semantic_phrase_slot_state import SemanticPhraseSlotState
from spora.semantic_text_writer import TextSemanticWriter, encode_value_sequence, value_positions


OUT = ROOT / 'runs/deepseek-spora-learned-natural-writer-20261116'
TRAIN_VALUES = ('酸奶', '豆浆', '咖啡', '果汁', '牛奶', '红茶', '绿茶', '啤酒', '椰汁', '柠檬水')
HELD_VALUES = ('奶茶', '橙汁', '矿泉水', '可乐')
TRAIN_NAMES = crossed.TRAIN_NAMES
HELD_NAMES = crossed.HELD_NAMES
STEPS = 512
LR = 3e-4
SEED = 91827
MAX_TOKENS = 16


def collect_pairs(model, tokenizer, embedding, names, values):
    features, targets = [], []
    for name in names:
        for value in values:
            fact = f'{name}喜欢喝{value}。'
            encoded, positions = value_positions(tokenizer, fact, value)
            device = next(model.parameters()).device
            inputs = {key: tensor.to(device) for key, tensor in encoded.items()}
            with torch.inference_mode():
                hidden = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state[0].float()
            token_ids = tokenizer.encode(value, add_special_tokens=False)
            feature = hidden.index_select(0, torch.tensor(positions, device=device))
            target = embedding.index_select(0, torch.tensor(token_ids, device=device)).float()
            features.append(feature)
            targets.append(target)
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def fit_writer(writer, features, targets):
    optimizer = torch.optim.AdamW(writer.parameters(), lr=LR, weight_decay=1e-5, foreach=False)
    trace = []
    for step in range(STEPS):
        prediction = writer(features)
        prediction_norm = F.normalize(prediction, dim=-1)
        target_norm = F.normalize(targets, dim=-1)
        cosine_loss = 1.0 - (prediction_norm * target_norm).sum(-1).mean()
        scale = targets.detach().norm(dim=-1).mean().clamp_min(1e-6)
        mse_loss = ((prediction - targets.detach()) / scale).square().mean()
        loss = cosine_loss + 0.25 * mse_loss
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite text writer loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(writer.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 64 == 0:
            trace.append({'step': step + 1, 'loss': float(loss.detach()),
                          'cosine_loss': float(cosine_loss.detach()),
                          'gradient_norm': float(gradient)})
            print(json.dumps(trace[-1]), flush=True)
    return trace


def writer_metrics(writer, features, targets):
    with torch.inference_mode():
        prediction = F.normalize(writer(features), dim=-1)
        target = F.normalize(targets, dim=-1)
        cosine = (prediction * target).sum(-1)
    return {'cases': int(cosine.numel()), 'cosine_mean': float(cosine.mean()),
            'cosine_min': float(cosine.min()), 'cosine_top90': int((cosine >= .9).sum())}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    if base_sha != p.WEIGHT_SHA:
        raise ValueError('Unexpected base weight')
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True,
                                                trust_remote_code=False)
    model = load_full_streamed().eval()
    model.requires_grad_(False)
    embedding = model.model.embed_tokens.weight.detach().float()
    train_features, train_targets = collect_pairs(model, tokenizer, embedding,
                                                  TRAIN_NAMES, TRAIN_VALUES)
    held_features, held_targets = collect_pairs(model, tokenizer, embedding,
                                                HELD_NAMES, HELD_VALUES)
    writer = TextSemanticWriter(embedding.shape[-1]).to('cuda')
    trace = fit_writer(writer, train_features, train_targets)
    writer_metrics_before = {'train': writer_metrics(writer, train_features, train_targets),
                             'held': writer_metrics(writer, held_features, held_targets)}

    OUT.mkdir()
    writer_path = OUT / 'writer.pt'
    writer_payload = {'format': 'spora-text-semantic-writer-v1', 'base_sha256': base_sha,
                      'width': embedding.shape[-1], 'seed': SEED, 'steps': STEPS,
                      'state': {key: value.detach().cpu() for key, value in writer.state_dict().items()}}
    atomic_save(writer_payload, writer_path)
    writer_sha = p.sha(writer_path)
    install_sequence_scaled(model, scale=32.)

    state = SemanticPhraseSlotState(embedding.shape[-1], 6, max_tokens=MAX_TOKENS)
    memory = NaturalSemanticPhraseMemory(state, base_sha256=base_sha, bridge_sha256=writer_sha)

    def learned_write(text):
        fact = text.rstrip('。！？!?')
        value = fact.split('喜欢喝', 1)[1]
        sequence = encode_value_sequence(model, tokenizer, writer, text, value)
        return memory.write_encoded_sequence(text, sequence, token_count=len(tokenizer.encode(value, add_special_tokens=False)))

    def generate(question, expected):
        route = memory.query_slot(question)
        if route['slot'] is None:
            return {'question': question, 'expected': expected, 'status': 'no_route'}
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
        expected_ids = tokenizer.encode(expected, add_special_tokens=False)
        kwargs = dict(**prompt, use_cache=True, max_new_tokens=len(expected_ids) + 3,
                      do_sample=False, pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id)
        with torch.inference_mode(), use_slot_sequence(route['value']):
            output = model.generate(**kwargs)
        tail = output[0, prompt['input_ids'].shape[1]:].tolist()
        return {'question': question, 'expected': expected, 'status': 'read',
                'route': {'subject': route['subject'], 'relation': route['relation'],
                          'slot': route['slot']}, 'expected_ids': expected_ids,
                'generated_ids': tail, 'exact_tokens': tail == expected_ids + [tokenizer.eos_token_id],
                'generated_text': tokenizer.decode(tail, skip_special_tokens=True)}

    initial_writes = [learned_write('小赵喜欢喝奶茶。'), learned_write('小郑喜欢喝橙汁。')]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'learned-slots.pt'
        memory.save(path, metadata={'writer_sha256': writer_sha, 'phase': 'initial'})
        restored = NaturalSemanticPhraseMemory.load(path, expected_base_sha256=base_sha,
                                                    expected_bridge_sha256=writer_sha, device='cuda')
        initial = [generate('小赵喜欢喝什么饮料？', '奶茶'),
                   generate('小郑喜欢喝什么饮料？', '橙汁'),
                   generate('小吴喜欢喝什么饮料？', '未知')]
        correction = restored.write_encoded_sequence(
            '小赵喜欢喝矿泉水。', encode_value_sequence(model, tokenizer, writer,
                                                        '小赵喜欢喝矿泉水。', '矿泉水'),
            token_count=len(tokenizer.encode('矿泉水', add_special_tokens=False)))
        restored.save(path, metadata={'writer_sha256': writer_sha, 'phase': 'corrected'})
        corrected = NaturalSemanticPhraseMemory.load(path, expected_base_sha256=base_sha,
                                                     expected_bridge_sha256=writer_sha, device='cuda')
        memory = corrected
        after = [generate('小赵喜欢喝什么饮料？', '矿泉水'),
                 generate('小郑喜欢喝什么饮料？', '橙汁'),
                 generate('小吴喜欢喝什么饮料？', '未知')]
    if p.sha(p.MODEL / 'model.safetensors') != base_sha:
        raise ValueError('Base model changed')
    initial_known = [row for row in initial if row['status'] == 'read']
    corrected_known = [row for row in after if row['status'] == 'read']
    result = {'status': 'learned_natural_writer_pilot', 'initial_writes': initial_writes,
              'correction': correction, 'initial': initial, 'corrected': after,
              'writer_metrics': writer_metrics_before, 'trace': trace,
              'summary': {'initial_full_exact': sum(row['exact_tokens'] for row in initial_known),
                          'initial_known_cases': len(initial_known),
                          'corrected_full_exact': sum(row['exact_tokens'] for row in corrected_known),
                          'corrected_known_cases': len(corrected_known),
                          'unknown_no_route_initial': sum(row['status'] == 'no_route' for row in initial),
                          'unknown_no_route_corrected': sum(row['status'] == 'no_route' for row in after)},
              'base_sha256': base_sha, 'writer_sha256': writer_sha,
              'train_values': TRAIN_VALUES, 'held_values': HELD_VALUES,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['The parser supplies the value span and relation pattern.',
                              'The writer trains on frozen hidden states and has no online self-supervised update.',
                              'Only four held values, two held names, one relation, and exact phrase output are tested.',
                              'No full open-domain dialogue or sleep consolidation.']}
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
