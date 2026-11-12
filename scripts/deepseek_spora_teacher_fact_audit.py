"""Compare fact-conditioned answers from the original and integrated decoder."""
import gc
import json
import tempfile
import time
from pathlib import Path

from scripts import deepseek_spora_long_memory_generate_parity as p

OUT = p.ROOT / 'runs/continuous-reader-teacher-fact-audit-20261112.json'
CASES = (
    ('小林', '苹果', '香蕉', '请只说出{name}喜欢的水果名称。'),
    ('小林', '香蕉', '苹果', '请只说出{name}喜欢的水果名称。'),
    ('小周', '西瓜', '葡萄', '记录里{name}喜欢哪种水果？只写名称。'),
    ('小周', '葡萄', '西瓜', '记录里{name}喜欢哪种水果？只写名称。'),
    ('小陈', '苹果', '香蕉', '{name}爱吃什么水果？只回答名字。'),
    ('小陈', '香蕉', '苹果', '{name}爱吃什么水果？只回答名字。'),
)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch = p.torch
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    source_sha = p.sha(p.runtime.CHECKPOINT)
    tokenizer = p.AutoTokenizer.from_pretrained(p.MODEL, local_files_only=True, trust_remote_code=False)
    rows = []
    for variant in ('original', 'integrated'):
        with tempfile.TemporaryDirectory() as temp:
            p.runtime.WORKING_CHECKPOINT = Path(temp) / 'unused.pt'
            if variant == 'original':
                base = p.load_full_streamed()
                decoder, chat = base.eval(), None
            else:
                chat = p.runtime.load_chat(base_loader=lambda: p.load_full_streamed(skip_replaced=(6, 12)))
                base, decoder = chat.base.eval(), chat.decoder.eval()
                digest = p.frozen_digest(chat.organism, chat.state)
            if any(x.is_meta for x in list(base.parameters()) + list(base.buffers())):
                raise ValueError('Unmaterialized model')
            for name, word, alternate, template in CASES:
                question = template.format(name=name)
                prompt = f'已知{name}喜欢的水果是{word}。' + question
                inputs = p.closed_think_inputs(tokenizer, prompt, 'cuda')
                targets = tokenizer(word, add_special_tokens=False, return_tensors='pt').input_ids.to('cuda')
                full = torch.cat((inputs['input_ids'], targets), dim=1)
                with torch.inference_mode():
                    scores = decoder(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False).logits.float()
                    start = inputs['input_ids'].shape[1]
                    logits = scores[:, start-1:start-1+targets.shape[1]]
                    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                            targets.reshape(-1), reduction='none')
                    first_rank = int((logits[0, 0] > logits[0, 0, targets[0, 0]]).sum()) + 1
                    result = decoder.generate(**inputs, use_cache=True, max_new_tokens=32, do_sample=False,
                                              pad_token_id=tokenizer.pad_token_id,
                                              eos_token_id=tokenizer.eos_token_id)
                tail = result[0, start:].tolist()
                answer = tokenizer.decode(tail, skip_special_tokens=True).strip()
                row = {'variant': variant, 'prompt': prompt, 'expected': word, 'alternate': alternate,
                       'answer': answer, 'token_ids': tail, 'eos': tokenizer.eos_token_id in tail,
                       'exact': answer == word, 'contains_expected_only': word in answer and alternate not in answer,
                       'target_token_nll': nll.cpu().tolist(), 'target_first_rank': first_rank}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=True), flush=True)
            if chat is not None and p.frozen_digest(chat.organism, chat.state) != digest:
                raise ValueError('Original individual changed')
            del decoder, base, chat
            gc.collect()
            torch.cuda.empty_cache()
    if p.sha(p.runtime.CHECKPOINT) != source_sha or p.sha(p.MODEL / 'model.safetensors') != p.WEIGHT_SHA:
        raise ValueError('Protected inputs changed')
    result = {'rows': rows, 'source_sha256': source_sha, 'base_sha256': p.WEIGHT_SHA,
              'seconds': time.perf_counter()-started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Facts are explicit in each prompt; no neural memory is evaluated.',
                              'Six authored cases and one generation format per case; not a general language benchmark.',
                              'Greedy outputs limited to 32 tokens; unended output may be incomplete.',
                              'Original and modified models run sequentially on identical hardware and prompts.',
                              'Substring checks are heuristic and complete raw answers are retained.']}
    with OUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'seconds': result['seconds'], 'rows': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
