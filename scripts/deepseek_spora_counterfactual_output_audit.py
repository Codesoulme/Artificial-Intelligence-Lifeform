"""Compare output bias under matched, independently written memory values."""
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_boundary_runtime_pilot import ENROLLMENTS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_output_bias_ablation import OUTPUT as PRIOR
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import query
from spora.internal_memory import observe_sourced
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputLogitsProcessor

OUTPUT = ROOT / 'runs/deepseek-spora-counterfactual-output-audit-20261112.json'
PERSON_INDICES = (0, 3)
VALUES = (1, 3, 6)
TEMPLATES = ('{name}现在想喝哪种饮料？', '{name}眼下喜欢喝哪种饮料？')


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    if prior['source_checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Previous checkpoint mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        runtime.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: load_full_streamed()
        chat = runtime.load_chat()
        device = next(chat.organism.parameters()).device
        rows = []
        for person in PERSON_INDICES:
            key, name, _ = ENROLLMENTS[person]
            for value in VALUES:
                chat.state = fresh_controlled_state(chat.organism)
                for j, (write_key, _, default_value) in enumerate(ENROLLMENTS):
                    written = value if j == person else default_value
                    chat.state, decision = observe_sourced(chat.organism, chat.state, write_key, written, 'direct', 1.0)
                    if not decision['current_write']:
                        raise ValueError('Controlled fact initialization refused')
                before = frozen_digest(chat.organism, chat.state)
                key_tensor = torch.tensor([key], dtype=torch.long, device=device)
                zeros = torch.zeros_like(key_tensor)
                with torch.no_grad():
                    memory_logits = query(chat.organism, chat.state, key_tensor, zeros, zeros)
                if not bool(torch.isfinite(memory_logits).all()):
                    raise ValueError('Nonfinite memory result')
                memory_top = int(memory_logits.argmax(-1)[0])
                for template_index, template in enumerate(TEMPLATES):
                    question = template.format(name=name)
                    enabled = chat.answer(question)
                    with patch.object(MemoryOutputLogitsProcessor, '__call__', lambda self, ids, scores: scores):
                        disabled = chat.answer(question)
                    expected = WORDS[value]
                    row = {'person_index': person, 'template_index': template_index, 'value': value,
                           'expected_word': expected, 'memory_top_value': memory_top,
                           'memory_matches': memory_top == value, 'enabled': enabled, 'disabled': disabled,
                           'enabled_correct': bool(enabled['accepted'] and enabled.get('key') == key
                                                   and enabled.get('answer') == expected),
                           'disabled_correct': bool(disabled['accepted'] and disabled.get('key') == key
                                                    and disabled.get('answer') == expected)}
                    rows.append(row)
                    print(json.dumps({'person': person, 'value': value, 'template': template_index,
                                      'memory_matches': row['memory_matches'], 'enabled_correct': row['enabled_correct'],
                                      'disabled_correct': row['disabled_correct'],
                                      'enabled_reason': enabled['reason'], 'disabled_reason': disabled['reason']}), flush=True)
                if frozen_digest(chat.organism, chat.state) != before:
                    raise ValueError('Memory changed during inference')
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected checkpoint changed')
    summary = {'examples': len(rows), 'memory_matches': sum(r['memory_matches'] for r in rows),
               'enabled_correct': sum(r['enabled_correct'] for r in rows),
               'disabled_correct': sum(r['disabled_correct'] for r in rows),
               'enabled_accepted': sum(r['enabled']['accepted'] for r in rows),
               'disabled_accepted': sum(r['disabled']['accepted'] for r in rows)}
    for person in PERSON_INDICES:
        group = [r for r in rows if r['person_index'] == person]
        summary[f'person_{person}'] = {'examples': len(group), 'memory_matches': sum(r['memory_matches'] for r in group),
                                      'enabled_correct': sum(r['enabled_correct'] for r in group),
                                      'disabled_correct': sum(r['disabled_correct'] for r in group)}
    result = {'status': 'matched_counterfactual_output_audit', 'summary': summary, 'rows': rows,
              'source_checkpoint_sha256': checkpoint_sha, 'seconds': time.perf_counter() - started,
              'limitations': ['Twelve authored conditions: two known names, three values, two question templates.',
                              'Independently initialized temporary memory; no natural-language memory write.',
                              'Only output bias is switched; other reader hooks remain active in both conditions.',
                              'Exact single-word answer scoring may undercount semantically equivalent wording.',
                              'Fresh controlled individual is not a real user state; no training or weights changed.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
