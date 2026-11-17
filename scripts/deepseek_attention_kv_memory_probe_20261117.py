"""Offline DeepSeek attention KV reconstruction, no model modification."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from spora.research_positive_feature_memory import read as feature_read, write as feature_write
from spora.research_random_feature_attention import read as random_read, write as random_write, merge as random_merge
from spora.research_positive_feature_memory import features as positive_features

OUT = ROOT / 'runs/deepseek-attention-kv-chunked-feature-probe-20261117.json'
TEXTS = (
    '小林喜欢羽毛球，小周喜欢游泳。第二天他们在公园散步。后来小林的爱好改成踢足球。请说出小林现在喜欢做什么。',
    '小王住在成都，小赵住在杭州。周末两人都去图书馆。小王搬到了南京。请说出小王现在住在哪座城市。',
    '小陈喜欢喝绿茶，小孙喜欢喝咖啡。午餐后两人去工作。小陈后来改喝豆浆。请说出小陈现在喜欢喝什么。',
    '小何是老师，小李是医生。两人约定在周五见面。小何后来成为工程师。请说出小何现在的职业。',
    '阿雅将一本诗集送给小孟。两人讨论了旅行和摄影。后来小孟把诗集借给阿武。请问现在谁借走了诗集？',
    '周二上午会议在蓝色会议室召开，参加的人有三名。周三的会议改到了绿色会议室，时间为下午。请说出周三的会议地点。',
    '仓库原来存有十二个纸箱，上午运走四个。下午又入库五个纸箱。管理员随后检查货架。现在仓库有多少纸箱？',
    '老陈以前每周坐公交去上班。搬家之后他开始骑车，偶尔乘坐地铁。雨天他一般在家工作。请问搬家后他开始怎样上班？',
)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.perf_counter()
    source = MODEL / 'model.safetensors'
    if sha(source) != WEIGHT_SHA:
        raise ValueError('Base checksum mismatch')
    torch.set_num_threads(2)
    random_projections = {seed: torch.randn((512, 128), generator=torch.Generator(device='cpu').manual_seed(seed)).to('cuda')
                          for seed in (110617, 110618, 110619)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
             trust_remote_code=False, torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval()
    model.requires_grad_(False)
    captured = {}
    attention = model.model.layers[5].self_attn
    hooks = []
    for name in ('q', 'k', 'v'):
        hooks.append(getattr(attention, name + '_proj').register_forward_hook(
            lambda module, args, output, name=name: captured.__setitem__(name, output.detach())))
    rows = []
    try:
        for index, text in enumerate(TEXTS):
            ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')
            with torch.inference_mode():
                model(input_ids=ids, use_cache=False)
                n = ids.shape[1]
                split = n // 2
                q = captured['q'].view(1, n, 12, 128).transpose(1, 2)
                k = captured['k'].view(1, n, 2, 128).transpose(1, 2)
                v = captured['v'].view(1, n, 2, 128).transpose(1, 2)
                pos = torch.arange(n, device=ids.device).unsqueeze(0)
                cos, sin = model.model.rotary_emb(torch.zeros(1, n, 128, device=ids.device, dtype=q.dtype), pos)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                for kv_head in range(2):
                    keys = k[0, kv_head, :split].float()
                    values = v[0, kv_head, :split].float()
                    queries = q[0, kv_head * 6:(kv_head + 1) * 6, split:].float().reshape(-1, 128)
                    weights = torch.softmax(queries @ keys.T / (128 ** 0.5), dim=-1)
                    target = weights @ values
                    baseline = values.mean(0).expand_as(target)
                    # Self-supervised least-squares KV map; this is a probe,
                    # not the paper's learned nonlinear test-time updates.
                    ridge = 1.0
                    dual = torch.linalg.solve(keys @ keys.T + ridge * torch.eye(split, device=keys.device), values)
                    mapping = keys.T @ dual
                    prediction = queries @ mapping
                    positive = feature_read(queries, feature_write(keys, values))
                    random_results = []
                    for seed, random_projection in random_projections.items():
                        full_state = random_write(keys, values, random_projection)
                        state = None
                        for start in range(0, split, 4):
                            part = random_write(keys[start:start+4], values[start:start+4], random_projection)
                            state = part if state is None else random_merge(state, part)
                        random_prediction = random_read(queries, random_projection, state)
                        full_prediction = random_read(queries, random_projection, full_state)
                        random_results.append({'seed': seed,
                                               'max_chunk_full_difference': float((random_prediction-full_prediction).abs().max()),
                                               'mse': float((random_prediction-target).square().mean()),
                                               'cosine': float(F.cosine_similarity(random_prediction,target,dim=-1).mean())})
                    positive_weights = positive_features(queries) @ positive_features(keys).T
                    positive_weights = positive_weights / positive_weights.sum(-1, keepdim=True)
                    def entropy(prob):
                        return float(-(prob * prob.clamp_min(1e-30).log()).sum(-1).mean())
                    def metrics(pred, truth):
                        return {'mse': float((pred - truth).square().mean()),
                                'cosine': float(F.cosine_similarity(pred, truth, dim=-1).mean())}
                    rows.append({'text_index': index, 'kv_head': kv_head, 'tokens': n,
                                 'history_tokens': split, 'queries': int(queries.shape[0]),
                                 'stored_kv_reconstruction': metrics(keys @ mapping, values),
                                 'query_historical_attention': metrics(prediction, target),
                                 'positive_feature_memory': metrics(positive, target),
                                 'random_feature_memories': random_results,
                                 'attention_entropy': entropy(weights),
                                 'positive_feature_entropy': entropy(positive_weights),
                                 'uniform_entropy': float(torch.log(torch.tensor(float(split)))),
                                 'history_mean_baseline': metrics(baseline, target)})
    finally:
        for hook in hooks:
            hook.remove()
    del model
    if sha(source) != WEIGHT_SHA:
        raise ValueError('Base checksum changed')
    result = {'layer_index': 5, 'base_sha256': WEIGHT_SHA, 'rows': rows,
              'seconds': time.perf_counter() - started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'limitations': ['Eight authored examples, one layer, last four added after prior experiment.',
                              'Queries were computed with full history in earlier layers.',
                              'Historical attention output is diagnostic target, not a generated answer.',
                              'Ridge fit is not Titan-Llama nonlinear gradient-based memory.',
                              'Entire prefix stored in solve temporarily; no persistent state or efficiency claim.']}
    with OUT.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
