"""Upper-bound diagnostic: exact memory tokens added directly to lm-head logits."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
from torch import nn
from torch.nn import functional as F
from scripts import deepseek_spora_long_memory_generate_parity as p
from scripts.deepseek_named_binding_probe import KEYS, NAMES
from scripts.deepseek_spora_sentence_lora_pilot import make_state
from scripts.deepseek_spora_semantic_bridge_pilot import MemorySemanticLogitResidual
from spora.continuous_text_memory import ContinuousTextCodec

OUT = ROOT / 'runs/deepseek-spora-exact-logit-memory-upper-bound-20261116'
VALUES = (0, 1, 4, 7)
ENTITIES = (0, 3)
QUESTION = '请只回答{name}喜欢的饮品名称。'


class ExactEmbeddingBridge(nn.Module):
    """Research-only nearest codebook lookup returning the frozen embedding."""
    def __init__(self, codec, embedding):
        super().__init__()
        self.slot_count = codec.slot_count
        self.slot_width = codec.slot_width
        self.register_buffer('codebook', codec.codebook.detach().clone())
        self.register_buffer('embedding', embedding.detach().float().clone())

    def forward(self, codes):
        flat = F.normalize(codes.float().reshape(-1, self.slot_width), dim=-1)
        valid = codes.float().reshape(-1, self.slot_width).norm(dim=-1) > 1e-5
        ids = (flat @ self.codebook.float().T).argmax(-1)
        vectors = self.embedding.index_select(0, ids).reshape(codes.shape[0], codes.shape[1], -1)
        return vectors * valid.reshape(codes.shape[0], codes.shape[1], 1)


def answer(decoder, tokenizer, question, args):
    prompt = p.closed_think_inputs(tokenizer, question, 'cuda')
    options = dict(args)
    options['use_cache'] = False
    with torch.inference_mode():
        output = decoder.generate(**prompt, **options, max_new_tokens=24, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    tail = output[0, prompt['input_ids'].shape[1]:].tolist()
    return tokenizer.decode(tail, skip_special_tokens=True).strip(), tail


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.manual_seed(85102)
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base_sha, checkpoint_sha = p.sha(p.MODEL / 'model.safetensors'), p.sha(p.runtime.CHECKPOINT)
    with tempfile.TemporaryDirectory(prefix='spora-logit-upper-bound-') as folder:
        p.runtime.WORKING_CHECKPOINT = Path(folder) / 'unused.pt'
        chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
        decoder, model, tok = chat.decoder, chat.organism, chat.tokenizer
        embedding = chat.base.model.embed_tokens.weight.detach()
        codec = ContinuousTextCodec(embedding.cpu(), memory_width=model.config.memory_width,
                                    slot_count=5, seed=29071, eos_token_id=tok.eos_token_id).to('cuda')
        bridge = ExactEmbeddingBridge(codec, embedding).to('cuda').eval()
        original_head = decoder.base.lm_head
        adapter = MemorySemanticLogitResidual(original_head, decoder._memory_context, model.config.memory_width,
                                              bridge, embedding, condition_slots=4, rank=1).to('cuda').eval()
        with torch.no_grad():
            adapter.scale.fill_(1.0)
        states = {(entity, value): make_state(chat, KEYS[entity], value)
                  for entity in ENTITIES for value in VALUES}
        rows = []
        for mode, head in (('baseline', original_head), ('exact_logit', adapter)):
            decoder.base.lm_head = head
            for entity in ENTITIES:
                for value in VALUES:
                    state = states[entity, value]
                    args, native = p.runtime._prepare_memory_read(model, state, KEYS[entity], 'current', None)
                    if int(native.argmax(-1)[0]) != value:
                        raise ValueError('Native memory read mismatch')
                    question = QUESTION.format(name=NAMES[entity])
                    text, tokens = answer(decoder, tok, question, args)
                    rows.append({'mode': mode, 'entity': entity, 'value': p.WORDS[value],
                                 'question': question, 'answer': text, 'token_ids': tokens,
                                 'contains': p.WORDS[value] in text,
                                 'exact': text.strip('。.!！ ') == p.WORDS[value]})
            print(json.dumps({'mode': mode, 'rows': len([r for r in rows if r['mode'] == mode])}), flush=True)
        decoder.base.lm_head = original_head
    if p.sha(p.MODEL / 'model.safetensors') != base_sha or p.sha(p.runtime.CHECKPOINT) != checkpoint_sha:
        raise ValueError('Protected source changed')
    summary = {}
    for mode in ('baseline', 'exact_logit'):
        group = [r for r in rows if r['mode'] == mode]
        summary[mode] = {'contains': sum(r['contains'] for r in group), 'exact': sum(r['exact'] for r in group),
                         'cases': len(group)}
    report = {'summary': summary, 'rows': rows, 'base_sha256': base_sha, 'checkpoint_sha256': checkpoint_sha,
              'bridge': 'nearest_codebook_to_frozen_embedding_direct_lm_head_logit_residual',
              'limitations': ['This is an explicit vocabulary oracle, not a learned bridge.',
                              'Entity and memory address are supplied by the test.',
                              'It bypasses natural language routing and does not prove parameter-internal memory.',
                              'No ordinary chat, unknown state or independent reload evaluation.',
                              'A positive result would locate the bottleneck but would not be a deployable solution.'],
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    OUT.mkdir()
    (OUT / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
