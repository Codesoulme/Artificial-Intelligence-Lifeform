"""Controlled out-of-distribution audit; direct packets, no user state writes."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'runs/open-base-deps'))

import torch
from scripts import deepseek_spora_runtime_chat as runtime
from spora.structured_text_memory import StructuredTextMemoryCodec
from spora.structured_text_output import StructuredFactSuffixDecoder

CHECKPOINT = ROOT / 'runs/deepseek-spora-memory-output-integrated-v5-20261111/individual-with-output-adapter.pt'
CANDIDATE = ROOT / 'runs/deepseek-spora-structured-fact-suffix-decoder-20261113/candidate.pt'
OUTPUT = ROOT / 'runs/deepseek-spora-suffix-generalization-audit-20261113.json'
CASES = (
    ('trained', '苹果'), ('trained', '香蕉'), ('trained', '西瓜'), ('trained', '葡萄'),
    ('unseen_fruit', '草莓'), ('unseen_fruit', '橘子'), ('unseen_fruit', '芒果'),
    ('other_domain', '咖啡'), ('other_domain', '读书'), ('other_domain', '蓝色'),
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    torch.set_num_threads(2)
    source_sha = runtime.sha(CHECKPOINT)
    candidate_sha = runtime.sha(CANDIDATE)
    with tempfile.TemporaryDirectory() as temp:
        runtime.CHECKPOINT = CHECKPOINT
        runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
        chat = runtime.load_chat()
        embedding = chat.base.model.embed_tokens.weight.detach()
        tokenizer = chat.tokenizer
        codec = StructuredTextMemoryCodec(
            embedding.cpu(), memory_width=90, identity_slots=3, fact_slots=5,
            identity_slot_width=15, fact_slot_width=9, seed=29071,
            eos_token_id=tokenizer.eos_token_id)
        decoder = StructuredFactSuffixDecoder.from_checkpoint(
            embedding, CANDIDATE, expected_source_sha256=source_sha,
            expected_base_sha256=runtime.WEIGHT_SHA,
            expected_codec=codec.config_dict()).to(embedding.device).eval()
        identity = tokenizer.encode('小赵', add_special_tokens=False)
        rows = []
        for group, fact in CASES:
            ids = tokenizer.encode(fact, add_special_tokens=False)
            packet, metadata = codec.encode(identity, ids)
            reconstructed = codec.decode(packet, metadata)['fact']['token_ids']
            with torch.inference_mode():
                output = decoder.generate(packet.to(embedding.device).unsqueeze(0),
                                          eos_id=tokenizer.eos_token_id, max_new_tokens=24)[0].tolist()
            answer = tokenizer.decode(output, skip_special_tokens=True).strip()
            rows.append({'group': group, 'fact': fact, 'fact_token_ids': ids,
                         'codec_exact': reconstructed == ids, 'answer': answer,
                         'answer_contains_fact': fact in answer, 'output_token_ids': output})
    report = {'status': 'controlled_suffix_generalization_audit', 'rows': rows,
              'source_sha256': source_sha, 'candidate_sha256': candidate_sha,
              'limitations': ['Direct packets with one known held identity; no natural-language routing.',
                              'Unseen values and other domains are small authored probes.',
                              'The decoder has no uncertainty or refusal head; absence of refusal is reported.']}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'cases': len(rows), 'codec_exact': sum(r['codec_exact'] for r in rows),
                      'trained_correct': sum(r['answer_contains_fact'] for r in rows if r['group'] == 'trained'),
                      'unseen_correct': sum(r['answer_contains_fact'] for r in rows if r['group'] != 'trained')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
