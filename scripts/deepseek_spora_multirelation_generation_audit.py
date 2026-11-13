"""Counterfactual language reads across three domains, without prompt facts."""
from dataclasses import replace
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
from scripts import deepseek_spora_bridge_reload_audit as audit
from spora.continuous_text_memory import ContinuousTextCodec

torch, p = audit.torch, audit.p
audit.SOURCE = ROOT / 'runs/continuous-reader-cross-attention-pilot-20261113'
audit.OUT = ROOT / 'runs/deepseek-spora-multirelation-generation-audit-20261113'
audit.EXTRA_ONLY = True
audit.PARITY_VALUE_INDICES = (0, 1, 3, 9, 16, 17)
DOMAINS = {
    'drink': ('喜欢喝', '请只回答小赵喜欢喝什么饮品。', ('茶', '咖啡', '牛奶', '豆浆')),
    'color': ('喜欢的颜色是', '请只回答小赵喜欢的颜色。', ('红色', '蓝色', '绿色', '黄色')),
    'hobby': ('喜欢的活动是', '请只回答小赵喜欢的活动。', ('读书', '跑步', '画画', '唱歌')),
}
done = False


def observe(*, decoder, tokenizer, spec, cross_modules, group, name, expected, question):
    global done
    if done or expected != '苹果':
        return
    embedding = decoder.base.model.embed_tokens.weight.detach()
    codec = ContinuousTextCodec(embedding.cpu(), memory_width=90, slot_count=5,
                                seed=29071, eos_token_id=tokenizer.eos_token_id).to('cuda')
    addresses = spec['memory_addresses']
    original = spec['memory_state']
    identity_id = tokenizer.encode(name, add_special_tokens=False)[-1]
    rows = []

    def generate(text, memory):
        prompt = p.closed_think_inputs(tokenizer, text, 'cuda')
        with torch.inference_mode():
            output = decoder.generate(**prompt, **memory, use_cache=True, max_new_tokens=24,
                                      do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                      eos_token_id=tokenizer.eos_token_id)
        tail = output[0, prompt['input_ids'].shape[1]:].tolist()
        return {'answer': tokenizer.decode(tail, skip_special_tokens=True).strip(), 'token_ids': tail}

    for domain, (relation, inquiry, values) in DOMAINS.items():
        off = generate(inquiry, {})
        for fact in values:
            fact_ids = tokenizer.encode(fact, add_special_tokens=False)
            if not 1 <= len(fact_ids) <= 3:
                raise ValueError(f'Fact exceeds packet capacity: {domain}')
            with torch.no_grad():
                packet = codec.encode(torch.tensor([identity_id, *fact_ids], device='cuda')).reshape(1, -1)
                matrices = []
                for matrix, address in zip(original.current, addresses):
                    key = torch.nn.functional.normalize(address.float(), dim=-1)
                    old = torch.bmm(matrix.float(), key.unsqueeze(-1)).squeeze(-1)
                    scale = old.reshape(1, 5, 18)[:, 0].norm(dim=-1, keepdim=True)
                    matrices.append(matrix.float() + (packet * scale - old).unsqueeze(-1) * key.unsqueeze(1))
                decoded = codec.decode(torch.stack([
                    torch.bmm(matrix, torch.nn.functional.normalize(address.float(), dim=-1).unsqueeze(-1)).squeeze(-1)
                    for matrix, address in zip(matrices, addresses)]).mean(0)[0])['token_ids']
            if decoded != [identity_id, *fact_ids, tokenizer.eos_token_id]:
                raise ValueError('Controlled packet failed to decode')
            state = replace(original, current=matrices)
            memory = {**spec, 'memory_state': state}
            memory.pop('use_cache', None)
            on = generate(inquiry, memory)
            oracle = generate(f'以下是虚构记录：{name}{relation}{fact}。\n{inquiry}', {})
            def correct(result):
                answer = result['answer']
                return answer in (fact, fact + '。', f'{name}{relation}{fact}。', f'{name}{relation}{fact}')
            row = {'domain': domain, 'question': inquiry, 'fact': fact, 'fact_ids': fact_ids,
                   'codec_verified': True, 'off': {**off, 'strict_correct': correct(off)},
                   'memory': {**on, 'strict_correct': correct(on)},
                   'explicit_fact': {**oracle, 'strict_correct': correct(oracle)}}
            rows.append(row)
            with (audit.OUT / 'multirelation_rows.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(json.dumps({'domain': domain, 'done': len(rows)}), flush=True)
    summary = {domain: {mode: sum(r[mode]['strict_correct'] for r in rows if r['domain'] == domain)
                        for mode in ('off', 'memory', 'explicit_fact')}
               for domain in DOMAINS}
    result = {'summary': summary, 'rows': rows,
              'limitations': ['One synthetic person, one inquiry per domain and four values per domain.',
                              'Entity, relation, slot boundaries and memory writing are explicitly supplied.',
                              'Oracle condition is fact text in the prompt, not an internal memory.',
                              'Strict scorer may reject correct paraphrases; raw answers are preserved.',
                              'No automatic relationship routing, persistence, new training or independent test set.']}
    (audit.OUT / 'multirelation-result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    done = True
    print(json.dumps(summary, ensure_ascii=False), flush=True)


audit.PROBE_OBSERVER = observe
if __name__ == '__main__':
    audit.main()
    if not done:
        raise ValueError('Multirelation probe did not run')
