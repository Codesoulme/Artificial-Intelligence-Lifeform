"""Train a small selector for the currently confirmed Chinese time span."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from scripts.audit_spora_dialogue_working_state import FrozenTurnFeatures
from scripts.audit_spora_fast_weight_lexical_write import base
from scripts.audit_spora_fresh_dialogue import digest
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.spora_active_time_cases import held_texts, training
from scripts.train_spora_recurrent_counterfactual import store_json
from spora.language_training import atomic_save


SEED = 71502
DAY = r'(?:[上下]?(?:周|星期)[一二三四五六日天末]|今天|明天|后天|今日|明日|次日)'
PERIOD = r'(?:清早|清晨|早晨|早上|上午|中午|下午|傍晚|晚上|夜里|夜间)'
CLOCK = r'(?:[零〇一二两三四五六七八九十]{1,3}|[0-9]{1,2})(?:点(?:半|[0-9一二三四五六七八九十]{1,3}分)?|[:：][0-9]{2})'
TIME = re.compile(
    rf'{DAY}(?:{PERIOD}|[早晚](?={CLOCK}))?(?:{CLOCK})?|[今明后][早晚](?:{CLOCK})?|{PERIOD}(?:{CLOCK})?|{CLOCK}'
)


class ActiveTimeSelector(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, 1))
        self.absent = nn.Parameter(torch.zeros(()))

    def forward(self, features, valid):
        logits = self.score(F.layer_norm(features.float(), (features.shape[-1],))).squeeze(-1)
        return torch.cat((self.absent.expand(logits.shape[0], 1), logits.masked_fill(~valid, -torch.inf)), dim=1)


def candidates(text):
    return list(TIME.finditer(text))


def features_for(extract, texts):
    """Use the frozen decoder's final contextual feature for each marked candidate."""
    flattened, groups = [], []
    for text in texts:
        start = len(flattened)
        found = candidates(text)
        for span in found:
            begin, end = span.span()
            marked = text[:begin] + '【' + span.group() + '】' + text[end:]
            flattened.append(marked + '\n这一次要求采用【' + span.group() + '】作为实际时间吗？')
        groups.append((found, start))
    if not flattened:
        width = extract.model.config.hidden_size
        return [[] for _ in texts], torch.empty(len(texts), 0, width), torch.zeros(len(texts), 0, dtype=torch.bool)
    contextual = []
    for start in range(0, len(flattened), 24):
        batch = flattened[start:start + 24]
        hidden = extract(batch)
        contextual.extend(hidden[i, len(text) - 1].clone() for i, text in enumerate(batch))
    width = max(len(found) for found, _ in groups)
    values = contextual[0].new_zeros((len(groups), width, contextual[0].numel()))
    valid = torch.zeros((len(groups), width), dtype=torch.bool)
    for i, (found, start) in enumerate(groups):
        if found:
            values[i, :len(found)] = torch.stack(contextual[start:start + len(found)])
            valid[i, :len(found)] = True
    return [found for found, _ in groups], values, valid


def fit(head, cases, found, features, valid):
    targets = []
    for case, matches in zip(cases, found, strict=True):
        choices = [i + 1 for i, match in enumerate(matches) if match.group() == case.current]
        if case.current and not choices:
            raise ValueError(f'Training target is outside lexical candidates: {case.current} in {case.text}')
        targets.append(choices[-1] if choices else 0)
    targets = torch.tensor(targets)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.002)
    rng = torch.Generator().manual_seed(SEED)
    for epoch in range(12):
        losses = []
        for batch in torch.randperm(len(cases), generator=rng).split(24):
            logits = head(features[batch], valid[batch])
            loss = F.cross_entropy(logits, targets[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch in (0, 5, 11):
            print('selector', epoch + 1, sum(losses) / len(losses), flush=True)


@torch.inference_mode()
def evaluate(head, cases, found, features, valid):
    rows, totals = [], {}
    logits = head(features, valid)
    choices = logits.argmax(dim=1).tolist()
    probabilities = logits.softmax(dim=1)
    for index, (case, matches, choice) in enumerate(zip(cases, found, choices, strict=True)):
        matched = matches[choice - 1] if choice else None
        picked = matched.group() if matched is not None else None
        latest = matches[-1].group() if matches else None
        available = case.current is None or any(span.group() == case.current for span in matches)
        summary = totals.setdefault(case.family, Counter())
        summary.update(cases=1, candidate_available=available, correct=picked == case.current,
                       latest_correct=latest == case.current, picked=picked is not None,
                       false_accept=case.current is None and picked is not None)
        rows.append({'family': case.family, 'text': case.text, 'expected': case.current, 'selected': picked,
                     'offsets': list(matched.span()) if matched is not None else None,
                     'candidates': [span.group() for span in matches], 'latest': latest,
                     'confidence': round(float(probabilities[index, choice]), 7)})
    return rows, totals


def main(args):
    started = perf_counter()
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    if not args.verify:
        args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / ('replay.json' if args.verify else 'result.json')
    if target.exists():
        raise FileExistsError(target)
    model_path, artifact = base.MODEL / 'model.safetensors', args.output / 'selector.pt'
    paths = (model_path, base.MODEL / 'tokenizer.json', Path(__file__),
             Path(__file__).with_name('spora_active_time_cases.py'),
             Path(__file__).with_name('spora_memory_composition_cases.py'))
    sources = {str(path): digest(path) for path in paths}
    if sources[str(model_path)] != base.WEIGHT_SHA:
        raise ValueError('Expected frozen DeepSeek weights')
    model = load_full_streamed()
    tokenizer = base.AutoTokenizer.from_pretrained(base.MODEL, local_files_only=True, trust_remote_code=False)
    features = FrozenTurnFeatures(SimpleNamespace(base=model), tokenizer)
    torch.manual_seed(SEED)
    head = ActiveTimeSelector(model.config.hidden_size)
    train, held = list(dict.fromkeys(training())), list(held_texts())
    if {case.text for case in train} & {case.text for case in held}:
        raise ValueError('A held question occurs in training')
    if args.verify:
        head.load_state_dict(torch.load(artifact, map_location='cpu', weights_only=True))
    else:
        print('extracting training features', len(train), flush=True)
        fit(head, train, *features_for(features, [case.text for case in train]))
        atomic_save(head.state_dict(), artifact)
    rows, summary = evaluate(head.eval(), held, *features_for(features, [case.text for case in held]))
    if any(digest(Path(path)) != value for path, value in sources.items()):
        raise ValueError('A protected source changed')
    data = {'sources': sources, 'head_sha256': digest(artifact), 'train_count': len(train), 'rows': rows,
            'summary': summary, 'seconds': perf_counter() - started, 'peak_bytes': torch.cuda.max_memory_allocated()}
    if args.verify:
        prior = json.loads((args.output / 'result.json').read_text(encoding='utf-8'))
        if any(data[key] != prior[key] for key in ('sources', 'head_sha256', 'rows', 'summary')):
            raise ValueError('Reload changed time selections')
    store_json(target, data)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify', action='store_true')
    main(parser.parse_args())
