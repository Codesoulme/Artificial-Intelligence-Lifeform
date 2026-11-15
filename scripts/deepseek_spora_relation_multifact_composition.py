"""Compose a frozen relation probe with isolated multi-fact structured storage.

Entity addresses are explicitly supplied by the test, and output is codec
decoding, not generated language. This experiment never edits the individual.
"""
import gc
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_observation_value_probe import pool
from scripts.deepseek_pooled_address import extract
from scripts.deepseek_spora_relation_selection_probe import LABELS
from scripts.deepseek_spora_structured_runtime_pilot import BOOTSTRAP, load_base, make_codec
from spora.integrated_durable import create_integrated_durable

HEAD = ROOT / 'runs/deepseek-spora-relation-selection-probe-20261116/head.pt'
OUT = ROOT / 'runs/deepseek-spora-relation-multifact-composition-20261116.json'
FACTS = ((0, '小林', 'drink', '茶'), (0, '小林', 'color', '红色'),
         (0, '小林', 'activity', '读书'), (1, '小周', 'drink', '咖啡'),
         (1, '小周', 'color', '蓝色'), (1, '小周', 'activity', '跑步'))
# Entity key is deliberately supplied as an oracle, never extracted from text.
CASES = (
    (0, '小林最喜欢喝的饮料是什么？', 'drink', '茶'),
    (0, '小林喜欢什么颜色？', 'color', '红色'),
    (0, '小林喜欢什么活动？', 'activity', '读书'),
    (1, '小周最喜欢喝的饮料是什么？', 'drink', '咖啡'),
    (1, '小周喜欢什么颜色？', 'color', '蓝色'),
    (1, '小周喜欢什么活动？', 'activity', '跑步'),
    (0, '小林喜欢的饮料是什么味道？', 'reject', None),
    (0, '请不要告诉我小林喜欢的饮品。', 'reject', None),
    (1, '小周喜欢的饮料和颜色是什么？', 'reject', None),
    (1, '小周喜欢什么电影？', 'reject', None),
    (2, '小赵喜欢什么颜色？', 'color', None),
)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.set_num_threads(2)
    base_sha = p.sha(p.MODEL / 'model.safetensors')
    source_sha = p.sha(p.runtime.CHECKPOINT)
    bootstrap_sha = p.sha(BOOTSTRAP)
    head_sha = p.sha(HEAD)
    head_data = torch.load(HEAD, map_location='cpu', weights_only=True)
    if (base_sha != p.WEIGHT_SHA or head_data['base_sha256'] != base_sha
            or head_data['checkpoint_sha256'] != source_sha or tuple(head_data['labels']) != LABELS):
        raise ValueError('Incompatible head/source')
    head = nn.Linear(head_data['state']['weight'].shape[1], len(LABELS))
    head.load_state_dict(head_data['state'], strict=True)
    with tempfile.TemporaryDirectory(prefix='spora-multifact-') as folder:
        root = Path(folder)
        p.runtime.WORKING_CHECKPOINT = root / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        individual_digest = p.frozen_digest(chat.organism, chat.state)
        features, masks = extract(chat.base, chat.tokenizer, [row[1] for row in CASES], batch_size=1)
        with torch.inference_mode():
            selected = head(pool(features, masks).float()).argmax(-1).tolist()
        if p.frozen_digest(chat.organism, chat.state) != individual_digest:
            raise ValueError('Source individual mutated')
        del chat, features, masks
        gc.collect()
        torch.cuda.empty_cache()
        base, tokenizer = load_base()
        with create_integrated_durable(root / 'isolated.sqlite', BOOTSTRAP, base, p.MODEL,
                                       individual_device='cuda') as runtime:
            codec = make_codec(runtime, tokenizer)
            for entity, name, relation, fact in FACTS:
                packet, meta = codec.encode(tokenizer.encode(name, add_special_tokens=False),
                                            tokenizer.encode(fact, add_special_tokens=False))
                runtime.write_structured_relation(entity, relation, packet, meta,
                                                  codec_config=codec.config_dict())

            def evaluate(phase):
                rows = []
                for (entity, question, expected_relation, expected_fact), index in zip(CASES, selected):
                    relation = LABELS[index]
                    record = None if relation == 'reject' else runtime.read_structured_relation(entity, relation)
                    decoded = None if record is None else codec.decode(record['packet'], record['metadata'])
                    value = None if decoded is None else tokenizer.decode(decoded['fact']['token_ids'], skip_special_tokens=True)
                    target = ('果汁' if phase == 'corrected' and entity == 0
                              and expected_relation == 'drink' else expected_fact)
                    rows.append({'phase': phase, 'question': question, 'oracle_entity': entity,
                                 'expected_relation': expected_relation, 'predicted_relation': relation,
                                 'expected_fact': target, 'decoded_fact': value,
                                 'correct': relation == expected_relation and value == target,
                                 'unsafe_read': expected_relation == 'reject' and record is not None})
                return rows

            before = evaluate('before')
            packet, meta = codec.encode(tokenizer.encode('小林', add_special_tokens=False),
                                        tokenizer.encode('果汁', add_special_tokens=False))
            runtime.write_structured_relation(0, 'drink', packet, meta, replace_existing=True)
            after = evaluate('corrected')
            missing = runtime.read_structured_relation(0, 'unknown') is None
    protected = ((HEAD, head_sha), (p.MODEL / 'model.safetensors', base_sha),
                 (p.runtime.CHECKPOINT, source_sha), (BOOTSTRAP, bootstrap_sha))
    if any(p.sha(path) != digest for path, digest in protected):
        raise ValueError('Protected source changed')
    report = {'before': before, 'after': after, 'unknown_relation_absent': missing,
              'head_sha256': head_sha, 'base_sha256': base_sha, 'bootstrap_sha256': bootstrap_sha,
              'limitations': ['Entity addresses are explicitly supplied by the test.',
                              'Relations/values are explicitly written and stored in external structured state.',
                              'Exact codec decode is not generation or a learned natural-language write.',
                              'No cross-process restore or calibrated rejection is evaluated.']}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'before_correct': sum(x['correct'] for x in before),
                      'after_correct': sum(x['correct'] for x in after),
                      'unsafe_reads': sum(x['unsafe_read'] for x in after),
                      'rows_per_phase': len(before), 'unknown_relation_absent': missing}), flush=True)


if __name__ == '__main__':
    main()
