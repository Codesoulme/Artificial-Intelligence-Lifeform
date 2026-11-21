"""Train a zero-init, cache-safe SPORA residual inside one Qwen block."""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_qwen_compatible_raw_control_20261121 import evaluate
from audit_awake_multiturn_layers import encode
from deepseek_output_aligned_memory import MODEL, sha, torch, write
from scripts.deepseek_semantic_value_pilot import WORDS
from scripts.deepseek_spora_full_stream_loader import load_full_streamed
from scripts.deepseek_spora_memory_answer_distill import _memory_args
from scripts.spora_unknown_state_probe import fresh_controlled_state
from spora.growing import grow
from spora.integrated_checkpoint import load_integrated_runtime
from spora.internal_memory import observe_sourced
from spora.language_training import atomic_save
from spora.pretrained_attention import SPORAResidualBlock, SPORAPretrainedBlock
from train_awake_dialogue_recovery import dialogue_cases, memory_logits
from train_awake_feedforward_replay import paired_targets
from train_awake_native_outputs import memory_batches
from train_awake_partial_layer import corpus
from train_awake_replacement_replay import answer_logits, divergence


SOURCE = Path("runs/qwen-compatible-raw-v1/integrated.pt")
SEED = 20261121


def prose_cases(tokenizer):
    rows = []
    for text in corpus():
        ids = tokenizer.encode(text, add_special_tokens=False)[:192]
        if len(ids) < 10:
            continue
        split = max(4, min(len(ids) - 2, len(ids) // 2))
        tail = ids[split:min(len(ids), split + 48)]
        if len(tail) >= 2:
            rows.append((ids[:split], tail))
    return rows


def dialogue_replay(path, tokenizer, limit):
    rows = []
    seen = set()
    eos = tokenizer.eos_token_id
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            messages = json.loads(line)["messages"]
            if len(messages) < 4 or len(messages) % 2:
                raise ValueError("Expected at least two complete dialogue turns")
            for i, message in enumerate(messages):
                if message["role"] != ("assistant" if i % 2 else "user") or not message["content"].strip():
                    raise ValueError("Invalid dialogue message")
                if message["role"] != "assistant":
                    continue
                rendered = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
                prompt = tokenizer.encode(rendered + "</think>\n\n", add_special_tokens=False)
                target = tokenizer.encode(message["content"], add_special_tokens=False) + [eos]
                fingerprint = (tuple(prompt), tuple(target))
                if len(prompt) + len(target) <= limit and fingerprint not in seen:
                    rows.append((prompt, target))
                    seen.add(fingerprint)
    if not rows:
        raise ValueError("No eligible dialogue rows")
    return rows


def expanded_memory_cases(organism, tokenizer):
    names = ("杜桐", "蔡乔", "顾河", "方晗", "周芸", "吴鸣", "沈泉", "丁青",
             "胡南", "齐飞", "柯禾", "安秋", "程木", "许圆", "孟欢", "马平")
    current = ("请核对最新一条，{name}习惯点什么饮料？", "记着{name}现在想喝哪样吗？",
               "查一下{name}眼下的饮料选择。", "目前留档的是{name}偏爱喝什么？")
    past = ("改过那条之前，{name}通常选什么喝？", "哪种饮料是{name}以前那条偏好？",
            "能查一下更新之前{name}爱喝什么吗？", "回到更早的备注，{name}选的饮品是什么？")
    rows = []
    for offset in (0, 2):
        state = fresh_controlled_state(organism)
        pending = []
        for i, name in enumerate(names):
            key = 12 + i
            event = state.cursor
            earlier, latest = (i + offset) % len(WORDS), (i + offset + 3) % len(WORDS)
            for value in (earlier, latest):
                if state.cursor >= len(state.pages) * organism.config.time_count:
                    state = grow(organism, state)
                state, decision = observe_sourced(organism, state, key, value, "direct", 1.)
                if not decision["current_write"]:
                    raise ValueError("Expanded replay observation was rejected")
            pending.extend((
                (current[i % len(current)].format(name=name), key, "current", event, latest),
                (past[i % len(past)].format(name=name), key, "past", event, earlier),
            ))
        for question, key, mode, event, value in pending:
            prompt = encode(tokenizer, [{"role": "user", "content": question}])
            tail = tokenizer.encode(WORDS[value], add_special_tokens=False) + [tokenizer.eos_token_id]
            controls = _memory_args(organism, state, {"key": key, "intent": mode, "event_time": event})
            rows.append((prompt, tail, controls))
    return rows


def unadapted_logits(decoder, layer_index, block, prompt, target, controls=None):
    install_layer(decoder, layer_index, block.base)
    try:
        with torch.no_grad():
            logits = (answer_logits(decoder, prompt, target) if controls is None
                      else memory_logits(decoder, prompt, target, controls))
            return logits.log_softmax(-1)
    finally:
        install_layer(decoder, layer_index, block)


def train(decoder, layer_index, block, language, memory, steps, learning_rate, reference_weight,
          memory_weight, contrast_weight, *, dialogue=(), dialogue_kl_weight=1.):
    block.down.requires_grad_(True)
    block.up.requires_grad_(True)
    parameters = [block.down.weight, block.up.weight]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.)
    rng = random.Random(SEED)
    paired = memory_batches(memory, contrast_weight)
    memory_order = list(range(len(paired)))
    if contrast_weight:
        random.Random(SEED).shuffle(memory_order)
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if memory and step % 3 == 0:
            pair_index, side = divmod(step // 3, 2 if contrast_weight else 1)
            batch = paired[memory_order[pair_index % len(paired)]]
            sample, contrast = batch[side]
            prompt, target, controls = sample[:3]
            teacher = (sample[3] if len(sample) == 4 else
                       unadapted_logits(decoder, layer_index, block, prompt, target, controls))
            logits = memory_logits(decoder, prompt, target, controls)
            value_loss = torch.nn.functional.cross_entropy(logits, torch.tensor(target, device="cuda"))
            kl = divergence(logits, teacher)
            if contrast is None:
                loss = memory_weight * (kl + .25 * value_loss)
            else:
                position, expected, alternative = contrast
                margin = logits[position, expected] - logits[position, alternative]
                contrast_loss = torch.nn.functional.softplus(1. - margin)
                loss = memory_weight * (.25 * kl + .25 * value_loss + contrast_weight * contrast_loss)
            source = "memory"
        else:
            pool = dialogue if dialogue and step % 5 != 1 else language
            is_dialogue = pool is dialogue
            prompt, target = pool[rng.randrange(len(pool))]
            teacher = unadapted_logits(decoder, layer_index, block, prompt, target)
            logits = answer_logits(decoder, prompt, target)
            kl = divergence(logits, teacher)
            value_loss = torch.nn.functional.cross_entropy(logits, torch.tensor(target, device="cuda"))
            loss = (dialogue_kl_weight if is_dialogue else 1.) * kl + reference_weight * value_loss
            source = "dialogue" if is_dialogue else "language"
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite residual training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append({"source": source, "kl": float(kl.detach()),
                       "value_loss": float(value_loss.detach()),
                       "total": float(loss.detach()), "residual_norm": block.residual_norm()})
        del teacher, logits, value_loss, kl, loss
        if (step + 1) % 24 == 0:
            window = losses[-24:]
            print("Step", step + 1, {key: round(sum(row[key] for row in window) / len(window), 6)
                                      for key in ("kl", "value_loss", "total", "residual_norm")
                                      if key in window[0]}, flush=True)
    block.requires_grad_(False).eval()
    return losses


def install_layer(decoder, index, block):
    container = decoder.base.model.layers[index]
    if container.__class__.__name__ == "_MemoryDecoderLayer":
        container.layer = block
    else:
        decoder.base.model.layers[index] = block


def main(output, ceval, layer_index, rank, steps, learning_rate, reference_weight, memory_weight,
         contrast_weight, *, checkpoint=SOURCE, dialogue_path=None, dialogue_kl_weight=1.,
         expanded_memory=False):
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    protected = {
        checkpoint: sha(checkpoint),
        MODEL / "model.safetensors": sha(MODEL / "model.safetensors"),
        MODEL / "tokenizer.json": sha(MODEL / "tokenizer.json"),
    }
    if dialogue_path is not None:
        protected[dialogue_path] = sha(dialogue_path)
    started = time.perf_counter()
    torch.manual_seed(SEED)
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False)
    base = load_full_streamed()
    runtime = load_integrated_runtime(checkpoint, base, MODEL)
    decoder, organism = runtime.decoder, runtime.organism
    decoder.eval().requires_grad_(False)
    position_limit = min(getattr(layer, "max_length", base.config.max_position_embeddings)
                         for layer in decoder.base.model.layers)
    if decoder.position_table_max_position is not None:
        position_limit = min(position_limit, decoder.position_table_max_position)
    container = decoder.base.model.layers[layer_index]
    source = container.layer if container.__class__.__name__ == "_MemoryDecoderLayer" else container
    if not isinstance(source, SPORAPretrainedBlock):
        raise ValueError("Selected layer must be a compatible pretrained block")
    paired_history = contrast_weight > 0
    baseline = evaluate(decoder, tokenizer, ceval, position_limit, organism, paired_history=paired_history)
    write(output / "baseline.json", baseline)
    print("Baseline", baseline["summary"], flush=True)
    dialogues = dialogue_cases(tokenizer)
    prose = prose_cases(tokenizer)
    language = [(prompt, target) for _, prompt, target in dialogues] + prose
    dialogue = dialogue_replay(dialogue_path, tokenizer, min(256, position_limit)) if dialogue_path else ()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    memory = (expanded_memory_cases(organism, tokenizer) if expanded_memory
              else paired_targets(decoder, tokenizer, organism))
    if not language or not memory:
        raise ValueError("The residual curriculum is empty")
    torch.cuda.reset_peak_memory_stats()
    block = SPORAResidualBlock(source, rank=rank)
    install_layer(decoder, layer_index, block)
    losses = train(decoder, layer_index, block, language, memory, steps, learning_rate, reference_weight,
                   memory_weight, contrast_weight, dialogue=dialogue, dialogue_kl_weight=dialogue_kl_weight)
    path = output / f"layer-{layer_index}-residual.pt"
    atomic_save({
        "format": "spora-qwen-compatible-residual-block-v1",
        "source_checkpoint": str(checkpoint),
        "base_model": str(MODEL),
        "layer": layer_index,
        "rank": rank,
        "state": {"down": block.down.weight.detach().cpu(), "up": block.up.weight.detach().cpu()},
    }, path)
    integrated_path = output / "integrated.pt"
    runtime.save(integrated_path)
    print("Saved", integrated_path, flush=True)
    trained = evaluate(decoder, tokenizer, ceval, position_limit, organism, paired_history=paired_history)
    if any(sha(source_path) != digest for source_path, digest in protected.items()):
        raise ValueError("Protected source changed during residual training")
    write(output / "result.json", {
        "format": "spora-qwen-compatible-residual-block-diagnostic-v1",
        "source_checkpoint": str(checkpoint),
        "layer": layer_index,
        "rank": rank,
        "steps": steps,
        "learning_rate": learning_rate,
        "reference_weight": reference_weight,
        "memory_weight": memory_weight,
        "contrast_weight": contrast_weight,
        "baseline": baseline,
        "trained": trained,
        "language_sequences": len(language),
        "dialogue_sequences": len(dialogue),
        "dialogue_kl_weight": dialogue_kl_weight,
        "memory_sequences": len(memory),
        "expanded_memory": expanded_memory,
        "losses": losses,
        "candidate_sha256": sha(path),
        "integrated_checkpoint_sha256": sha(integrated_path),
        "sources": {str(source_path): digest for source_path, digest in protected.items()},
        "seconds": time.perf_counter() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    })
    print("Trained", trained["summary"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ceval", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--reference-weight", type=float, default=.25)
    parser.add_argument("--memory-weight", type=float, default=.5)
    parser.add_argument("--contrast-weight", type=float, default=0.)
    parser.add_argument("--checkpoint", type=Path, default=SOURCE)
    parser.add_argument("--dialogue-corpus", type=Path)
    parser.add_argument("--dialogue-kl-weight", type=float, default=1.)
    parser.add_argument("--expanded-memory", action="store_true")
    args = parser.parse_args()
    if args.layer < 0 or args.rank < 1 or args.steps < 1 or args.learning_rate <= 0:
        parser.error("Layer, rank, steps and learning-rate are invalid")
    if any(not 0 <= weight < float("inf") for weight in
           (args.reference_weight, args.memory_weight, args.contrast_weight, args.dialogue_kl_weight)):
        parser.error("Loss weights must be finite and nonnegative")
    if not args.ceval.is_dir():
        parser.error("C-Eval directory does not exist")
    main(args.output, args.ceval, args.layer, args.rank, args.steps,
         args.learning_rate, args.reference_weight, args.memory_weight, args.contrast_weight,
         checkpoint=args.checkpoint, dialogue_path=args.dialogue_corpus, dialogue_kl_weight=args.dialogue_kl_weight,
         expanded_memory=args.expanded_memory)
