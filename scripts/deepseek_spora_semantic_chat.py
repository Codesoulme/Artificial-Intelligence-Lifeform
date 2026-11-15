"""Local chat entry for the persisted semantic phrase-slot prototype."""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]

import torch

from scripts import deepseek_local_baseline as p
from scripts import deepseek_spora_ttt_read_adapter_pilot as pilot
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from spora.research_semantic_slot import install_sequence_scaled, use_slot_sequence
from spora.semantic_phrase_slot_runtime import (NaturalSemanticPhraseMemory,
                                                 SEQUENCE_BRIDGE_BINDING)
from spora.semantic_phrase_slot_state import SemanticPhraseSlotState
from spora.semantic_slot_runtime import parse_fact, parse_question_route


def build_memory(path, base_sha, width, max_slots):
    path = Path(path)
    if path.exists():
        return NaturalSemanticPhraseMemory.load(
            path, expected_base_sha256=base_sha,
            expected_bridge_sha256=SEQUENCE_BRIDGE_BINDING, device='cpu')
    return NaturalSemanticPhraseMemory(
        SemanticPhraseSlotState(width, max_slots, max_tokens=16),
        base_sha256=base_sha, bridge_sha256=SEQUENCE_BRIDGE_BINDING)


def generate(model, tokenizer, question, *, slot=None, max_new_tokens=64):
    if not isinstance(question, str):
        return '这条输入无法解析。'
    try:
        prompt = pilot.closed_think_inputs(tokenizer, question, 'cuda')
    except (TypeError, ValueError):
        return '这条输入无法解析。'
    kwargs = dict(**prompt, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False,
                  pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    with torch.inference_mode():
        if slot is None:
            output = model.generate(**kwargs)
        else:
            with use_slot_sequence(slot.to('cuda')):
                output = model.generate(**kwargs)
    return tokenizer.decode(output[0, prompt['input_ids'].shape[1]:].tolist(),
                            skip_special_tokens=True).strip()


def persist_fact(memory, text, tokenizer, embedding, path):
    """Publish the candidate only after the state file was saved."""
    candidate = copy.deepcopy(memory)
    result = candidate.write(text, tokenizer, embedding)
    candidate.save(path, metadata={'interface': 'semantic-sequence-chat-v1'})
    return candidate, result


def main():
    for stream in (sys.stdin, sys.stdout):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is not None:
            reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description='SPORA semantic memory chat')
    parser.add_argument('--state', type=Path, default=ROOT / 'runs/semantic-chat-state.pt')
    parser.add_argument('--max-slots', type=int, default=128)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()
    if args.max_slots < 1 or args.max_slots > 4096:
        raise ValueError('--max-slots must be between 1 and 4096')
    if args.max_new_tokens < 1 or args.max_new_tokens > 512:
        raise ValueError('--max-new-tokens must be between 1 and 512')

    base_sha = p.sha(p.MODEL / 'model.safetensors')
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True,
                                                trust_remote_code=False)
    model = load_full_streamed().eval()
    install_sequence_scaled(model, scale=32.)
    model.requires_grad_(False)
    embedding = model.model.embed_tokens.weight.detach()
    memory = build_memory(args.state, base_sha, embedding.shape[-1], args.max_slots)
    args.state.parent.mkdir(parents=True, exist_ok=True)

    for raw in sys.stdin:
        text = raw.strip()
        if not text:
            continue
        try:
            fact = parse_fact(text)
        except ValueError:
            fact = None
        if fact is not None:
            try:
                memory, result = persist_fact(memory, text, tokenizer, embedding, args.state)
            except (OSError, ValueError, RuntimeError) as error:
                print(f'记录失败：{error}', flush=True)
                continue
            print(f"已记录：{result['subject']}的{result['relation']}事实已保存。", flush=True)
            continue
        try:
            route = memory.query_slot(text)
        except ValueError:
            print(generate(model, tokenizer, text, max_new_tokens=args.max_new_tokens), flush=True)
            continue
        if route['slot'] is None:
            print(f"我还没有记录{route['subject']}的{route['relation']}信息。", flush=True)
            continue
        print(generate(model, tokenizer, text, slot=route['value'],
                       max_new_tokens=args.max_new_tokens), flush=True)


if __name__ == '__main__':
    main()
