"""Evaluate reading open-vocabulary fact scores from a persisted SPORA packet."""

import argparse
import json
from pathlib import Path

from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, sha, torch
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.integrated_durable import create_integrated_durable, load_integrated_durable
from spora.memory_output import MemorySequenceLogitsProcessor
from spora.structured_text_memory import StructuredTextMemoryCodec
from transformers import AutoTokenizer


FACTS = (
    ('许乔', '集合点', '北门', '银杏亭'),
    ('罗玥', '纪念品', '竹制书签', '铜铃'),
    ('江乐', '工具', '木夹', '迷你手电'),
    ('舒晨', '颜色', '藏青', '浅紫'),
    ('许乔', '零食', '马蹄糕', '桂花糖'),
    ('江乐', '地点', '霞光桥', '东侧展台'),
)
GAINS = (0., 1024., 8192.)


def codec(runtime, tokenizer):
    embedding = runtime.decoder.base.get_input_embeddings().weight.detach()
    return StructuredTextMemoryCodec(embedding, memory_width=runtime.organism.config.memory_width,
                                     identity_slots=3, fact_slots=5, identity_slot_width=15,
                                     fact_slot_width=9, eos_token_id=tokenizer.eos_token_id)


def write(runtime, encoder, tokenizer, revised):
    for name, label, original, changed in FACTS:
        entity = runtime.resolve_confirmed_name(name)['key']
        if entity is None:
            raise ValueError(f'Missing enrolled name: {name}')
        tokens = tokenizer.encode(changed if revised else original, add_special_tokens=False)
        packet, bounds = encoder.encode(tokenizer.encode(name, add_special_tokens=False), tokens)
        runtime.write_structured_relation(entity, label, packet.cpu(), bounds, codec_config=encoder.config_dict(),
                                          replace_existing=revised)


@torch.inference_mode()
def assess(runtime, encoder, tokenizer, revised):
    rows = []
    for name, label, original, changed in FACTS:
        entity = runtime.resolve_confirmed_name(name)['key']
        record = runtime.read_structured_relation(entity, label)
        target = tokenizer.encode(changed if revised else original, add_special_tokens=False) + [tokenizer.eos_token_id]
        scores = encoder.fact_scores(record['packet'], record['metadata'])
        top = scores.topk(2, dim=-1)
        inputs = closed_think_inputs(tokenizer, f'请核对{name}登记的{label}。只答记录内容。', 'cuda')
        length = inputs['input_ids'].shape[1]
        generated = {}
        for gain in GAINS:
            processors = [MemorySequenceLogitsProcessor(scores * gain, length)] if gain else None
            output = runtime.generate_ordinary(**inputs, use_cache=True, do_sample=False, top_p=None, top_k=None,
                                               temperature=None, max_new_tokens=12, logits_processor=processors,
                                               eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            ids = output[0, length:].tolist()
            generated[str(gain)] = {'tokens': ids, 'exact': ids == target,
                                    'answer': tokenizer.decode(ids, skip_special_tokens=True).strip()}
        rows.append({'name': name, 'relation': label, 'expected': changed if revised else original,
                     'packet_exact': top.indices[:, 0].tolist() == target,
                     'minimum_margin': float((top.values[:, 0] - top.values[:, 1]).min()), 'generated': generated})
    unknown = (runtime.read_structured_relation(0, '日期') is None
               and runtime.read_structured_relation(63, '集合点') is None)
    return {'rows': rows, 'unknown_absent': unknown,
            'counts': {str(gain): sum(row['generated'][str(gain)]['exact'] for row in rows) for gain in GAINS}}


def main(checkpoint, output, phase):
    report = output / f'{phase}.json'
    if report.exists():
        raise FileExistsError(report)
    torch.set_num_threads(2)
    digest = sha(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    open_runtime = create_integrated_durable if phase == 'write' else load_integrated_durable
    with open_runtime(output / 'individual.sqlite', checkpoint, base, MODEL) as runtime:
        encoder = codec(runtime, tokenizer)
        if phase == 'write':
            for key, name in enumerate(dict.fromkeys(row[0] for row in FACTS)):
                runtime.enroll_name(key, name)
            write(runtime, encoder, tokenizer, False)
            first = assess(runtime, encoder, tokenizer, False)
            write(runtime, encoder, tokenizer, True)
            result = {'before_correction': first, 'corrected': assess(runtime, encoder, tokenizer, True)}
        else:
            result = {'corrected': assess(runtime, encoder, tokenizer, True)}
            original = json.loads((output / 'write.json').read_text(encoding='utf-8'))
            if original['corrected'] != result['corrected'] or original['revision'] != runtime.store.revision:
                raise ValueError('Persisted packet generation changed after reload')
        result['revision'] = runtime.store.revision
    if sha(checkpoint) != digest:
        raise ValueError('Protected bootstrap changed')
    result['bootstrap_sha256'] = digest
    with report.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(phase, {name: record['counts'] for name, record in result.items() if isinstance(record, dict)}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('write', 'reopen'), required=True)
    main(**vars(parser.parse_args()))
