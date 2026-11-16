"""Research-only low-rank correction of a learned semantic writer."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_local_baseline as base
from scripts import deepseek_spora_text_writer_broad_generalization as prior
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import OUTPUT as PAIR, _load_candidate
from scripts.deepseek_spora_output_aware_writer import GAIN, hard_negative_ids, norm_vectors, sample_output_tokens
from scripts.deepseek_spora_writer_layer_control import collect
from spora.language_training import atomic_save
from spora.semantic_text_writer import TextSemanticWriter

SOURCE = ROOT / 'runs/deepseek-spora-output-aware-broad-20261116'
OUT = ROOT / 'runs/deepseek-spora-writer-correction-20261116'
SEED, STEPS, RANK, REPLAY = 23817, 384, 8, 64
SUBJECT, TARGET = '小郑', '红茶'


class LowRankWriter(nn.Module):
    def __init__(self, original, rank=RANK):
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)
        width = original.projection.in_features
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.normal_(self.down.weight, std=width ** -0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden):
        return self.original(hidden) + self.up(self.down(hidden.float()))

    def merged_weight(self):
        return self.original.projection.weight + self.up.weight @ self.down.weight


def replay_distillation(student, teacher, temperature=2.):
    return F.kl_div(F.log_softmax(student / temperature, dim=-1),
                    F.softmax(teacher.detach() / temperature, dim=-1),
                    reduction='batchmean') * temperature ** 2


def main(output=OUT, replay_kl=0.):
    if output.exists():
        raise FileExistsError(output)
    if not 0 <= replay_kl <= 100:
        raise ValueError('Replay KL weight must be between 0 and 100')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    report = json.loads((SOURCE / 'result.json').read_text(encoding='utf-8'))
    source = SOURCE / 'writer.pt'
    source_sha = base.sha(source)
    if report['base_sha256'] != base.WEIGHT_SHA or report['writer_sha256'] != source_sha:
        raise ValueError('Writer provenance mismatch')
    payload = torch.load(source, map_location='cpu', weights_only=True)
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    target_ids = tokenizer.encode(TARGET, add_special_tokens=False)
    if len(target_ids) != 1:
        raise ValueError('This pilot requires a single-token correction')
    replay_values = random.Random(SEED).sample(report['train_values'], REPLAY)
    train_ids = target_ids + [tokenizer.encode(value, add_special_tokens=False)[0] for value in replay_values]
    held_ids = [tokenizer.encode(value, add_special_tokens=False)[0] for value in report['held_values']]
    sampled_ids = sample_output_tokens(tokenizer.vocab_size, train_ids, held_ids, 4096, SEED)
    cases = [{'name': SUBJECT, 'value': TARGET}]
    cases += [{'name': person, 'value': value} for person in prior.TRAIN_NAMES for value in replay_values]
    paths = {i: PAIR / f'layer-{i}.pt' for i in (6, 12)}
    layer_hashes = {str(i): base.sha(path) for i, path in paths.items()}
    model = load_full_streamed(skip_replaced=(6, 12)).eval()
    for index, path in paths.items():
        model.model.layers[index] = _load_candidate(path, index, model).eval()
    if any(parameter.is_meta for parameter in model.parameters()):
        raise ValueError('Unmaterialized model parameter')
    features, _ = collect(model, tokenizer, cases)
    features = features['shallow_layer_1'].to('cuda')
    prompts = {}
    for person in (SUBJECT,) + prior.TRAIN_NAMES:
        question = pilot.closed_think_inputs(tokenizer, f'{person}喜欢喝什么饮料？', 'cuda')
        with torch.inference_mode():
            prompts[person] = model.model(**question, use_cache=False, return_dict=True).last_hidden_state[0, -1].float().cpu().clone()
    prompt_vectors = torch.stack([prompts[case['name']] for case in cases]).to('cuda')
    index_for_value = {value: i + 1 for i, value in enumerate(replay_values)}
    labels = torch.tensor([0] + [index_for_value[case['value']] for case in cases[1:]], device='cuda')
    original = TextSemanticWriter(model.config.hidden_size).to('cuda')
    original.load_state_dict(payload['state'], strict=True)
    writer = LowRankWriter(original).to('cuda').train()
    optimizer = torch.optim.AdamW([*writer.down.parameters(), *writer.up.parameters()], lr=2e-3,
                                  weight_decay=1e-4, foreach=False)
    head = model.lm_head.weight.detach()[sampled_ids].float()
    sampler = torch.Generator(device='cpu').manual_seed(SEED)
    trace = []
    for step in range(STEPS):
        replay = torch.randint(1, len(cases), (12,), generator=sampler)
        indexes = torch.cat((torch.zeros(4, dtype=torch.long), replay)).to('cuda')
        hidden = prompt_vectors.index_select(0, indexes) + GAIN * norm_vectors(
            writer(features.index_select(0, indexes)), payload['train_norm'])
        if step % 32 == 0:
            with torch.no_grad():
                logits = model.lm_head(hidden.detach().to(model.lm_head.weight.dtype)).float()
                extra = hard_negative_ids(logits, sampled_ids, held_ids, 8)
                if extra:
                    sampled_ids.extend(extra)
                    head = torch.cat((head, model.lm_head.weight.detach()[extra].float()))
        student_logits = F.linear(hidden, head)
        answer_loss = F.cross_entropy(student_logits, labels.index_select(0, indexes))
        with torch.no_grad():
            old_hidden = prompt_vectors.index_select(0, indexes[4:]) + GAIN * norm_vectors(
                original(features.index_select(0, indexes[4:])), payload['train_norm'])
            old_logits = F.linear(old_hidden, head)
        retention_loss = replay_distillation(student_logits[4:], old_logits)
        loss = answer_loss + replay_kl * retention_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_([*writer.down.parameters(), *writer.up.parameters()],
                                                   5., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 64 == 0:
            trace.append({'step': step + 1, 'loss': float(loss.detach()),
                          'answer_loss': float(answer_loss.detach()), 'replay_kl': float(retention_loss.detach()),
                          'gradient_norm': float(gradient), 'output_classes': len(sampled_ids)})
            print(json.dumps(trace[-1]), flush=True)
    with torch.no_grad():
        merged = writer.merged_weight().detach().cpu().clone()
        samples = features[:4]
        merged_output = F.linear(samples, merged.to('cuda'))
        max_difference = float((merged_output - writer(samples)).abs().max())
    if (base.sha(base.MODEL / 'model.safetensors') != base.WEIGHT_SHA or base.sha(source) != source_sha
            or any(base.sha(path) != layer_hashes[str(i)] for i, path in paths.items())):
        raise ValueError('Protected source changed')
    output.mkdir()
    target = output / 'writer.pt'
    atomic_save({'format': 'spora-output-aware-writer-v1', 'base_sha256': base.WEIGHT_SHA,
                 'state': {'projection.weight': merged}, 'train_norm': payload['train_norm'],
                 'source_writer_sha256': source_sha, 'correction': TARGET, 'rank': RANK}, target)
    result = {'base_sha256': base.WEIGHT_SHA, 'source_writer_sha256': source_sha,
              'writer_sha256': base.sha(target), 'layers': layer_hashes,
              'train_values': report['train_values'], 'held_values': report['held_values'],
              'replay_values': replay_values, 'target': TARGET, 'subject': SUBJECT,
              'steps': STEPS, 'replay_kl_weight': replay_kl,
              'trace': trace, 'merge_max_abs_difference': max_difference,
              'seconds': time.perf_counter() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Oracle fact span and target; one correction on one subject.',
                              'Replay is limited to 64 existing synthetic values.',
                              'No automatic migration of previously saved slot vectors.',
                              'Isolated research candidate; no formal individual modified.']}
    with (output / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'merge_max_abs_difference': max_difference}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--replay-kl', type=float, default=0.)
    args = parser.parse_args()
    main(args.output, args.replay_kl)
