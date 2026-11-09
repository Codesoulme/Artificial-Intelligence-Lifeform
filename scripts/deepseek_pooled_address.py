"""Train attention-pooled, finite SPORA addresses from frozen DeepSeek states."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.deepseek_full_answer import thinking_inputs
from scripts.deepseek_learned_address import EXPANDED_PROMPTS, HELD_PROMPTS, NOVEL_PROMPTS
from scripts.deepseek_local_baseline import MODEL, REVISION, WEIGHT_SHA
from scripts.open_base_baseline import INDIVIDUAL, INDIVIDUAL_SHA, sha
from scripts.open_base_real_memory_alignment import native_read
from scripts.open_base_trajectory_comparison import build_trajectories, numeric_correct
from spora.growing import load
from spora.language_training import frozen_digest
from spora.open_base_memory import OpenBaseMemory, memory_hooks


class PooledAddress(nn.Module):
    def __init__(self, hidden_width, memory_width):
        super().__init__()
        self.hidden_width = hidden_width
        self.score = nn.ModuleList(nn.Linear(hidden_width, 1, bias=False) for _ in range(2))
        self.query = nn.ModuleList(nn.Linear(hidden_width, memory_width, bias=False) for _ in range(2))
        for layer in self.score:
            nn.init.zeros_(layer.weight)

    def forward(self, features, mask):
        if (features.ndim != 4 or features.shape[1] != 2 or features.shape[-1] != self.hidden_width
                or mask.shape != (features.shape[0], features.shape[2]) or mask.dtype != torch.bool
                or not bool(mask.any(-1).all())):
            raise ValueError('Expected [batch, 2, tokens, hidden] features and nonempty boolean mask')
        features = features.float()
        result = []
        for site in range(2):
            x = features[:, site]
            logits = self.score[site](x).squeeze(-1).masked_fill(~mask, float('-inf'))
            pooled = (x * logits.softmax(-1).unsqueeze(-1)).sum(1)
            result.append(F.normalize(self.query[site](pooled), dim=-1))
        return result


def question_mask(offsets, attention_mask, texts, questions):
    if offsets.shape[:2] != attention_mask.shape or offsets.shape[-1] != 2 or len(texts) != len(questions):
        raise ValueError('Invalid token offsets')
    result = torch.zeros_like(attention_mask, dtype=torch.bool)
    for i, (text, question) in enumerate(zip(texts, questions)):
        start = text.find(question)
        if not question or start < 0 or text.find(question, start + 1) >= 0:
            raise ValueError('Question must occur exactly once in the chat template')
        end = start + len(question)
        result[i] = ((offsets[i, :, 0] < end) & (offsets[i, :, 1] > start)
                     & attention_mask[i].bool())
        if not bool(result[i].any()):
            raise ValueError('Question has no selectable tokens')
    return result


@torch.no_grad()
def extract(model, tokenizer, questions, batch_size=8, question_only=True, hidden_indices=(6, 18)):
    if (len(hidden_indices) != 2 or any(type(index) is not int or index < 1 for index in hidden_indices)
            or hidden_indices[0] == hidden_indices[1]):
        raise ValueError('Two distinct positive hidden-state indices are required')
    texts = [tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
                                           add_generation_prompt=True) for question in questions]
    if any('<think>' not in text or '</think>' in text for text in texts):
        raise ValueError('Unexpected reasoning template')
    encoded = tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                        return_offsets_mapping=True)
    offsets = encoded.pop('offset_mapping')
    masks = (question_mask(offsets, encoded['attention_mask'], texts, questions) if question_only else
             encoded['attention_mask'].bool())
    features = []
    for offset in range(0, len(texts), batch_size):
        inputs = {k: v[offset:offset + batch_size].to('cuda') for k, v in encoded.items()}
        output = model(**inputs, use_cache=False, output_hidden_states=True)
        if max(hidden_indices) >= len(output.hidden_states):
            raise ValueError('Hidden-state index outside model')
        batch = torch.stack([output.hidden_states[index] for index in hidden_indices], dim=1)
        if not bool(torch.isfinite(batch).all()):
            raise ValueError('Nonfinite model features')
        features.append(batch.cpu())
        if (offset // batch_size + 1) % 16 == 0:
            print(f'extracted {min(offset + batch_size, len(texts))}/{len(texts)}', flush=True)
    return torch.cat(features), masks


@torch.no_grad()
def stats(selector, features, masks, labels, bases, batch_size=64):
    predictions = []
    for offset in range(0, len(labels), batch_size):
        addresses = selector(features[offset:offset + batch_size].to('cuda'), masks[offset:offset + batch_size].to('cuda'))
        predictions.append(torch.stack([(addresses[i] @ bases[i].T).argmax(-1) for i in range(2)], dim=1).cpu())
    predicted = torch.cat(predictions)
    correct = predicted == labels[:, None]
    return {'examples': len(labels), 'site_correct': correct.sum(0).tolist(),
            'both_correct': int(correct.all(-1).sum()), 'predicted': predicted.tolist()}


def run(output, source, steps):
    output, source = Path(output).resolve(), Path(source).resolve()
    runs = (ROOT / 'runs').resolve()
    if (not output.is_relative_to(runs) or output == runs or output.exists()
            or not source.is_relative_to(runs) or type(steps) is not int or not 1 <= steps <= 10000):
        raise ValueError('Expected new output and source under runs, with valid steps')
    source_sha = sha(source)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if (checkpoint.get('format') != 'deepseek-spora-learned-finite-address-v1'
            or checkpoint.get('weight_sha256') != WEIGHT_SHA or checkpoint.get('individual_sha256') != INDIVIDUAL_SHA
            or checkpoint.get('layers') != [5, 17] or checkpoint.get('current_only') is not True):
        raise ValueError('Source adapter provenance mismatch')
    def verify():
        return (sha(MODEL / 'model.safetensors') == WEIGHT_SHA and sha(INDIVIDUAL) == INDIVIDUAL_SHA
                and sha(source) == source_sha)
    if not verify():
        raise ValueError('Protected source mismatch')
    torch.manual_seed(4608)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    organism, original, _ = load(INDIVIDUAL, 'cpu')
    before = frozen_digest(organism, original)
    keys = list(range(organism.config.key_count))
    if set(EXPANDED_PROMPTS) & (set(HELD_PROMPTS) | set(NOVEL_PROMPTS)):
        raise ValueError('Train and held templates overlap')
    datasets = {}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    adapter = OpenBaseMemory(model.config.hidden_size, organism.config.memory_width).to('cuda').eval()
    adapter.load_state_dict(checkpoint['state'], strict=True)
    adapter.requires_grad_(False)
    selector = PooledAddress(model.config.hidden_size, organism.config.memory_width).to('cuda')
    selector.query.load_state_dict(adapter.query.state_dict())
    for name, templates in (('train', EXPANDED_PROMPTS), ('held', HELD_PROMPTS), ('novel', NOVEL_PROMPTS)):
        questions = [template.format(key=key) for key in keys for template in templates]
        features, masks = extract(model, tokenizer, questions)
        labels = torch.tensor([key for key in keys for _ in templates], dtype=torch.long)
        datasets[name] = (features, masks, labels)
    bases = [F.normalize(site.current_basis.to('cuda').float(), dim=-1) for site in organism.memories]
    optimizer = torch.optim.AdamW(selector.parameters(), lr=0.001, weight_decay=0.)
    generator = torch.Generator().manual_seed(4608)
    train_features, train_masks, train_labels = datasets['train']
    log = []
    for step in range(steps):
        indices = torch.randint(len(train_labels), (64,), generator=generator)
        addresses = selector(train_features[indices].to('cuda'), train_masks[indices].to('cuda'))
        labels = train_labels[indices].to('cuda')
        loss = sum(F.cross_entropy(10 * addresses[i] @ bases[i].T, labels) for i in range(2)) / 2
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite address loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(selector.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0 or step + 1 == steps:
            entry = {'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)}
            log.append(entry)
            print(json.dumps(entry), flush=True)
    selector.eval()
    results = {name: stats(selector, *data, bases) for name, data in datasets.items()}
    if any(p.grad is not None for p in model.parameters()) or frozen_digest(organism, original) != before or not verify():
        raise ValueError('Protected inputs changed')
    output.mkdir(parents=False, exist_ok=False)
    snapshot = output / 'selector.pt'
    torch.save({'format': 'deepseek-spora-question-pooled-finite-address-v1',
                'selector': {k: v.detach().cpu() for k, v in selector.state_dict().items()},
                'readout': {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA,
                'source_adapter_sha256': source_sha, 'layers': [5, 17]}, snapshot)
    print('selector checkpoint saved; checking trajectories', flush=True)
    trajectories = build_trajectories(organism, original)
    held_features, held_masks, _ = datasets['held']
    rows = []
    with torch.inference_mode():
        for item in trajectories:
            for key in item['keys']:
                question = HELD_PROMPTS[0].format(key=key)
                inputs = thinking_inputs(tokenizer, question, 'cuda')
                index = key * len(HELD_PROMPTS)
                addresses = selector(held_features[index:index + 1].to('cuda'), held_masks[index:index + 1].to('cuda'))
                predicted = [int((addresses[i] @ bases[i].T).argmax(-1)) for i in range(2)]
                with memory_hooks([model.model.layers[5], model.model.layers[17]], adapter, item['state'],
                                  current_only=True, addresses=addresses):
                    generated = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                               pad_token_id=tokenizer.eos_token_id)
                raw = tokenizer.decode(generated[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                final = raw.split('</think>', 1)[1].strip() if '</think>' in raw else None
                expected = item['targets'][key]
                rows.append({'trajectory': item['id'], 'key': key, 'expected': expected,
                             'native': native_read(organism, item['state'], [key])[0],
                             'predicted_addresses': predicted, 'raw': raw, 'final_answer': final,
                             'correct': final is not None and numeric_correct(final, expected)})
            print(f"trajectory {item['id'] + 1}/4 complete", flush=True)
    if frozen_digest(organism, original) != before or not verify():
        raise ValueError('Protected input changed during evaluation')
    report = {'status': 'pooled_finite_supervised_address_development_only', 'revision': REVISION,
              'weight_sha256': WEIGHT_SHA, 'individual_sha256': INDIVIDUAL_SHA, 'source_adapter_sha256': source_sha,
              'selector_sha256': sha(snapshot), 'steps': steps, 'results': results,
              'trajectory': {'examples': len(rows), 'correct': sum(row['correct'] for row in rows), 'rows': rows},
              'log': log, 'seconds': time.perf_counter() - start, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['All 64 entity-to-address mappings are explicitly supervised.',
                              'Random address basis is individual-specific; no unseen-entity generalization.',
                              'Attention is restricted to the known user-question span, not an entity oracle.',
                              'Features are extracted without memory; predicted addresses remain fixed during generation.',
                              'P26 trained the original readout but is not invoked by this selector.',
                              'All trajectories share one original background and were used in earlier development.',
                              'Finite numbered questions only; unknown rejection and open semantics are untested.',
                              'Fixed short reasoning was supervised in source readout; no reasoning claim.']}
    with (output / 'result.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'results': {name: {k: v for k, v in value.items() if k != 'predicted'}
                                  for name, value in results.items()},
                      'trajectory': {'examples': len(rows), 'correct': sum(row['correct'] for row in rows)}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()
    run(args.output, args.source, args.steps)
