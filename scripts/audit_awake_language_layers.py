"""Isolate the effect of the active replacement blocks on ordinary generation."""

import argparse
import time
from pathlib import Path

from audit_awake_ordinary_generation import QUESTIONS
from deepseek_output_aligned_memory import MODEL, SOURCE, load_integrated, sha, torch, write
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.structure_transfer import SPORALayerReplacement


REFERENCES = (
    '热水向较冷的环境传递热量，温差缩小以后，冷却就会变慢。',
    '可以试试不加糖的柠檬气泡水。',
    '一共六块冰。',
    'What does she like to drink most?',
    '可以分层穿衣，带一件保暖外套，根据室内外温差增减。',
    '可以做几分钟深呼吸，也可以读几页喜欢的书。',
)


def load_layer_pair():
    from transformers import AutoTokenizer

    torch.set_num_threads(2)
    payload = torch.load(SOURCE, map_location='cpu', weights_only=True)
    indices = tuple(spec['index'] for spec in payload['metadata']['replacements'])
    del payload
    base = load_full_streamed()
    originals = {i: base.model.layers[i] for i in indices}
    decoder, organism, state = load_integrated(SOURCE, base, MODEL)
    decoder.eval().requires_grad_(False)
    replacements = {i: base.model.layers[i] for i in indices}
    if not all(isinstance(block, SPORALayerReplacement) for block in replacements.values()):
        raise ValueError('Unexpected active replacement type')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    return decoder, tokenizer, originals, replacements, organism, state


@torch.inference_mode()
def main(output, limit):
    checkpoint_hash = sha(SOURCE)
    base_hash = sha(MODEL / 'model.safetensors')
    decoder, tokenizer, originals, replacements, _, _ = load_layer_pair()
    base = decoder.base
    indices = tuple(originals)
    cases = []
    for question, reference in zip(QUESTIONS, REFERENCES, strict=True):
        prompt = tokenizer.apply_chat_template([dict(role='user', content=question)], tokenize=False,
                                               add_generation_prompt=True) + '</think>\n\n'
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        target = tokenizer.encode(reference, add_special_tokens=False) + [tokenizer.eos_token_id]
        cases.append((question, ids, target))
    variants = (('base', ()), *((f'layer_{i}', (i,)) for i in indices), ('integrated', indices))
    results = {}
    for name, active in variants:
        for index in indices:
            base.model.layers[index] = replacements[index] if index in active else originals[index]
        rows = []
        for question, ids, target in cases:
            teacher_forced = torch.tensor([ids + target], device='cuda')
            output_forced = decoder(input_ids=teacher_forced, use_cache=False).logits[0]
            logits = output_forced[len(ids) - 1:-1].float()
            nll = torch.nn.functional.cross_entropy(logits, teacher_forced[0, len(ids):], reduction='sum').item()
            del output_forced, logits
            prompt = torch.tensor([ids], device='cuda')
            torch.cuda.synchronize()
            started = time.perf_counter()
            generated = decoder.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt), do_sample=False,
                                         use_cache=True, max_new_tokens=limit, temperature=None, top_p=None,
                                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            tokens = generated[0, len(ids):].tolist()
            row = dict(question=question, answer=tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                       eos=tokens[-1] == tokenizer.eos_token_id, nll=nll, reference_tokens=len(target),
                       seconds=round(time.perf_counter() - started, 3), tokens=tokens)
            rows.append(row)
            print(name, question, row['answer'], row['eos'], flush=True)
        results[name] = rows
    if sha(SOURCE) != checkpoint_hash or sha(MODEL / 'model.safetensors') != base_hash:
        raise ValueError('Source model or checkpoint changed during the audit')
    summary = {name: dict(eos=sum(row['eos'] for row in rows),
                          reference_nll=sum(row['nll'] for row in rows) / sum(row['reference_tokens'] for row in rows),
                          identical_to_base=sum(row['tokens'] == base_row['tokens']
                                                for row, base_row in zip(rows, results['base'], strict=True)))
               for name, rows in results.items()}
    write(output, dict(summary=summary, replacements=indices, limit=limit, source_sha256=checkpoint_hash,
                       model_sha256=base_hash, references=REFERENCES, results=results))
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.limit <= 256:
        parser.error('An unused output and a limit between 1 and 256 are required')
    main(args.output, args.limit)
