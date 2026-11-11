"""Check whether a three-value output calibration damages other finite values."""
import copy
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
from scripts.deepseek_spora_counterfactual_output_audit import TEMPLATES
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_output_recalibration_pilot import OUTPUT as CANDIDATE, make_state
from spora.growing import query
from spora.language_training import frozen_digest
from spora.memory_output import MemoryOutputLogitsProcessor

OUTPUT = ROOT / 'runs/deepseek-spora-output-unseen-values-audit-20261112.json'
VALUES = (0, 2, 4, 5, 7)
PERSONS = (0, 3)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoint_sha = runtime.sha(runtime.CHECKPOINT)
    adapter_path = CANDIDATE / 'adapter.pt'
    adapter_sha = runtime.sha(adapter_path)
    prior = json.loads((CANDIDATE / 'result.json').read_text(encoding='utf-8'))
    data = torch.load(adapter_path, map_location='cpu', weights_only=True)
    if (prior['source_checkpoint_sha256'] != checkpoint_sha or prior['adapter_sha256'] != adapter_sha
            or data['source_checkpoint_sha256'] != checkpoint_sha):
        raise ValueError('Checkpoint or adapter provenance mismatch')
    torch.set_num_threads(2)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        runtime.WORKING_CHECKPOINT = Path(directory) / 'unused.pt'
        runtime.AutoModelForCausalLM.from_pretrained = lambda *args, **kwargs: load_full_streamed()
        chat = runtime.load_chat()
        original = chat.decoder.output_adapter
        if original.config_dict() != data['config']:
            raise ValueError('Output adapter configuration mismatch')
        adapter = copy.deepcopy(original)
        adapter.load_state_dict(data['state'], strict=True)
        adapter.eval()
        device = next(chat.organism.parameters()).device
        rows = []
        for person in PERSONS:
            key, name, _ = ENROLLMENTS[person]
            for value in VALUES:
                chat.state = make_state(chat, person, value)
                before = frozen_digest(chat.organism, chat.state)
                keys = torch.tensor([key], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = query(chat.organism, chat.state, keys, torch.zeros_like(keys), torch.zeros_like(keys))
                memory_top = int(logits.argmax(-1)[0])
                for template_index, template in enumerate(TEMPLATES):
                    question = template.format(name=name)
                    chat.decoder.output_adapter = original
                    parent = chat.answer(question)
                    chat.decoder.output_adapter = adapter
                    candidate = chat.answer(question)
                    with patch.object(MemoryOutputLogitsProcessor, '__call__', lambda self, ids, scores: scores):
                        no_bias = chat.answer(question)
                    expected = WORDS[value]
                    row = {'person': person, 'value': value, 'template': template_index, 'memory_top_value': memory_top,
                           'expected_word': expected, 'parent': parent, 'candidate': candidate, 'no_bias': no_bias}
                    for label in ('parent', 'candidate', 'no_bias'):
                        r = row[label]
                        row[label + '_correct'] = bool(r['accepted'] and r.get('key') == key and r.get('answer') == expected)
                    rows.append(row)
                    print(json.dumps({'person': person, 'value': value, 'template': template_index,
                                      'memory_matches': memory_top == value,
                                      **{name: row[name + '_correct'] for name in ('parent', 'candidate', 'no_bias')}}), flush=True)
                if frozen_digest(chat.organism, chat.state) != before:
                    raise ValueError('Evaluation modified memory')
        chat.decoder.output_adapter = original
    if runtime.sha(runtime.CHECKPOINT) != checkpoint_sha or runtime.sha(adapter_path) != adapter_sha:
        raise ValueError('Protected input changed')
    summary = {'examples': len(rows), 'memory_matches': sum(r['memory_top_value'] == r['value'] for r in rows),
               **{name + '_correct': sum(r[name + '_correct'] for r in rows) for name in ('parent', 'candidate', 'no_bias')}}
    report = {'status': 'held_value_development_audit', 'summary': summary, 'rows': rows,
              'candidate_sha256': adapter_sha, 'source_checkpoint_sha256': checkpoint_sha,
              'seconds': time.perf_counter() - started,
              'limitations': ['Twenty synthetic conditions across two known people and two recurring question templates.',
                              'Five values absent from this recalibration, but present in parent training.',
                              'Manual numerical writes and exact one-word output comparison.',
                              'One controlled memory per condition; no natural-language learning or real user state.',
                              'Same evaluation identities and prompts as prior development; not a final independent benchmark.']}
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
