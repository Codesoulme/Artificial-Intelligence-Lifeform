"""Inspect the default locator's selected spans on the near-name audit."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_learned_name_span import single_span_mask
from scripts.deepseek_spora_default_near_name_audit import CASES

OUTPUT = ROOT / 'runs/deepseek-spora-default-near-name-spans-20261111.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        chat = runtime.load_chat()
        questions = [q for _, q in CASES]
        texts = [chat.tokenizer.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False,
                                                    add_generation_prompt=True) for q in questions]
        encoded = chat.tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                                 return_offsets_mapping=True)
        features, masks = runtime.extract(chat.base, chat.tokenizer, questions, batch_size=2)
        vectors = torch.cat([F.layer_norm(features[:, i].float(), (features.shape[-1],)) for i in range(2)], dim=-1)
        locator = nn.Linear(vectors.shape[-1], 1).eval()
        locator.load_state_dict(chat.router['locator_state'], strict=True)
        with torch.no_grad():
            scores = locator(vectors).squeeze(-1).sigmoid()
        rows = []
        for i, (group, question) in enumerate(CASES):
            selected, single = single_span_mask(scores[i], masks[i], chat.router['locator_threshold'])
            token_ids = selected.nonzero(as_tuple=True)[0].tolist()
            start = texts[i].find(question)
            chars = set()
            for token in token_ids:
                lo, hi = encoded['offset_mapping'][i, token].tolist()
                chars.update(range(max(0, lo - start), min(len(question), hi - start)))
            selected_text = ''.join(question[j] for j in sorted(chars))
            rows.append({'group': group, 'question': question, 'selected_text': selected_text,
                         'single': single, 'selected_token_indices': token_ids})
    report = {'status': 'read_only_default_locator_span_probe', 'rows': rows,
              'source_checkpoint_sha256': runtime.sha(runtime.CHECKPOINT),
              'limitations': ['Ten authored cases; token offset reconstruction may include boundary merges.',
                              'No projection ablation, persistent memory write, or user-state modification.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for row in rows:
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
