"""Isolated paired-value continuation; all source dialogue and outputs stay local."""
import json, random, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runs/open-base-deps')]
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from scripts.deepseek_addressed_read import closed_think_inputs
from scripts.deepseek_local_baseline import MODEL, WEIGHT_SHA, sha
from scripts.deepseek_kernel_question_format_20261127 import selected_cases, build_facts, DATA
from scripts.deepseek_residual_kv_writer_20261127 import ResidualWriter, OUT as PARENT

OUT = ROOT / 'runs/research/relation-coverage-20261127-v1'
SEED, STEPS, LR = 112734, 96, 0.0003
REL = {
    '运动': ('排球', '击剑', '滑雪'), '魔术师': ('刘谦', '大卫科波菲尔', '蔡威泽'),
    '酒': ('葡萄酒', '清酒', '黄酒'), '歌手': ('周杰伦', '王菲', '林俊杰'),
    '职业': ('园丁', '厨师', '护士'), '食物': ('饺子', '面条', '包子'),
    '音乐': ('爵士', '古典', '摇滚'), '乐队': ('五月天', '逃跑计划', '旅行团')}
NAMES = ('小陈', '小吴')

def statement(n, r, v):
    template = {'运动':'最喜欢的运动是{}', '魔术师':'最喜欢的魔术师是{}',
        '酒':'最喜欢的酒是{}', '歌手':'喜欢的歌手是{}', '职业':'的职业是{}',
        '食物':'最喜欢吃{}', '音乐':'喜欢听{}音乐', '乐队':'经常听的乐队是{}'}
    return n + template[r].format(v) + '。'

def question(n, r):
    template = {'运动':'最喜欢的运动是什么', '魔术师':'最喜欢的魔术师是谁',
        '酒':'最喜欢的酒是什么', '歌手':'喜欢的歌手是谁', '职业':'的职业是什么',
        '食物':'最喜欢吃什么', '音乐':'喜欢听什么音乐', '乐队':'经常听的乐队是什么'}
    return n + template[r] + '？请简洁回答。'

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    def save(name, data):
        temp = OUT / (name + '.tmp')
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
        temp.replace(OUT / name)
    try:
        random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(2)
        torch.cuda.reset_peak_memory_stats()
        paths = {'base': MODEL / 'model.safetensors', 'parent': PARENT / 'candidate.pt',
                 'parent_report': PARENT / 'result.json', 'dialogue': DATA}
        hashes = {k: sha(v) for k, v in paths.items()}
        parent_report = json.loads(paths['parent_report'].read_text(encoding='utf8'))
        if hashes['base'] != WEIGHT_SHA or hashes['parent'] != parent_report['candidate_sha256']:
            raise ValueError('Unexpected input digest')
        payload = torch.load(paths['parent'], map_location='cpu', weights_only=True)
        if payload['base_sha256'] != WEIGHT_SHA: raise ValueError('Incorrect parent base')
        _, values, questions = selected_cases(json.loads(DATA.read_text(encoding='utf8')))
        if set(values) & {v for vals in REL.values() for v in vals}:
            raise ValueError('Source value equals a training value')
        facts = build_facts(values)
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        writer = ResidualWriter(model.config.num_hidden_layers).to('cuda')
        writer.load_state_dict(payload['writer'], strict=True)
        def plain(q): return closed_think_inputs(tok, q, 'cuda')['input_ids']
        with torch.no_grad():
            header = plain('')[:, :2]
            header_cache = tuple((k.detach(), v.detach()) for k,v in model(input_ids=header, use_cache=True).past_key_values.to_legacy_cache())
        def state(fact):
            ids = tok(fact + '\n', return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')
            with torch.no_grad():
                return tuple((k.detach().float(),v.detach().float()) for k,v in model(input_ids=ids,use_cache=True).past_key_values.to_legacy_cache())
        cases = [(n,r,v) for n in NAMES for r,vs in REL.items() for v in vs]
        states = {c:state(statement(*c)) for c in cases}
        source_states = [state(f) for f in facts]
        def run(q, source, answer=None):
            ids = plain(q)
            if not torch.equal(ids[:, :2], header): raise ValueError('Header mismatch')
            suffix = ids[:, 2:]
            current = suffix if answer is None else torch.cat((suffix, answer),1)
            if source is None:
                return model(input_ids=torch.cat((header,current),1),use_cache=False).logits.float(), suffix.shape[1]+2
            memory = writer(source)
            pairs = tuple((torch.cat((hk,memory[i][0].to(hk.dtype)),2),torch.cat((hv,memory[i][1].to(hv.dtype)),2)) for i,(hk,hv) in enumerate(header_cache))
            p = pairs[0][0].shape[2]
            out = model(input_ids=current, past_key_values=DynamicCache.from_legacy_cache(pairs),
                attention_mask=torch.ones(1,p+current.shape[1],device='cuda',dtype=torch.long),
                cache_position=torch.arange(p,p+current.shape[1],device='cuda'),use_cache=False)
            return out.logits.float(), suffix.shape[1]
        def loss(q, source, answer):
            target = torch.tensor([tok.encode(answer,add_special_tokens=False)+[tok.eos_token_id]],device='cuda')
            logits,n = run(q,source,target)
            return F.cross_entropy(logits[:,n-1:-1].reshape(-1,logits.shape[-1]),target.reshape(-1))
        def generate(q, source):
            tokens=[]
            with torch.no_grad():
                for _ in range(24):
                    added = torch.tensor([tokens],device='cuda') if tokens else None
                    logits,_ = run(q,source,added); t = logits[:,-1].argmax(-1).item(); tokens.append(t)
                    if t == tok.eos_token_id: break
            return {'text':tok.decode(tokens,skip_special_tokens=True).strip(), 'tokens':tokens,
                    'ended':bool(tokens and tokens[-1]==tok.eos_token_id)}
        rows=[]
        def evaluate(phase):
            writer.eval()
            for i,(q,v) in enumerate(zip(questions,values)):
                for mode, prompt, source in [('correct',q,source_states[i]),
                    ('wrong_relation',q,source_states[(i+1)%len(facts)]),
                    ('off',q,None), ('fact_prompt',facts[i]+'\n'+q,None)]:
                    ans=generate(prompt,source)
                    rows.append({'phase':phase,'group':'source_dev','index':i,'mode':mode,
                        'target':v,'target_in_text':v in ans['text'],**ans})
                save('progress.json',{'rows':rows});print(json.dumps({'phase':phase,'source_case':i+1}),flush=True)
            for case in [('小陈','运动','排球'),('小吴','歌手','王菲'),('小陈','职业','护士'),('小吴','音乐','摇滚')]:
                n,r,v=case; swap=next(x for x in REL[r] if x!=v); other='职业' if r!='职业' else '运动'
                for mode, source, target in [('correct',states[case],v),('swap',states[(n,r,swap)],swap),
                    ('wrong_relation',states[(n,other,REL[other][0])],None)]:
                    ans=generate(question(n,r),source)
                    rows.append({'phase':phase,'group':'synthetic_train','case':case,'mode':mode,
                                 'target':target,'target_in_text':target in ans['text'] if target else None,**ans})
            for q in ('请计算三加四。','请说一句鼓励人的话。'):
                for mode, source in [('off',None),('forced_unrelated',states[cases[0]])]:
                    rows.append({'phase':phase,'group':'ordinary','mode':mode,'question':q,**generate(q,source)})
            save('progress.json',{'rows':rows})
        evaluate('before')
        writer.train(); opt=torch.optim.AdamW(writer.parameters(),lr=LR,weight_decay=0)
        rng=random.Random(SEED); negative_rng=random.Random(SEED+1); order=cases.copy(); trace=[]
        for step in range(STEPS):
            if step % len(order)==0: rng.shuffle(order)
            case=order[step%len(order)]; n,r,v=case
            replacement=REL[r][(REL[r].index(v)+1)%len(REL[r])]
            wrong=negative_rng.choice([x for x in REL if x != r])
            q=question(n,r); normal=('请计算一加二。','请解释什么是雨。')[step%2]
            opt.zero_grad(set_to_none=True)
            ce=loss(q,states[case],statement(*case)); (0.5*ce).backward()
            counter=loss(q,states[(n,r,replacement)],statement(n,r,replacement)); (0.5*counter).backward()
            reject=loss(question(n,wrong),states[case],'我不知道。'); reject.backward()
            with torch.no_grad(): teacher=run(normal,None)[0][:,-1].detach()
            student=run(normal,states[case])[0][:,-1]
            kl=F.kl_div(F.log_softmax(student,-1),F.softmax(teacher,-1),reduction='batchmean')
            (0.25*kl).backward()
            grad=float(torch.nn.utils.clip_grad_norm_(writer.parameters(),1.0,error_if_nonfinite=True));opt.step()
            rec={'step':step+1,'state_relation':r,'question_relation':wrong,'correct_ce':float(ce.detach()),'swap_ce':float(counter.detach()),
                 'wrong_ce':float(reject.detach()),'normal_kl':float(kl.detach()),'grad':grad}
            if not all(torch.isfinite(torch.tensor(x)) for k,x in rec.items() if k not in ('state_relation','question_relation')):raise ValueError('Nonfinite loss')
            trace.append(rec)
            if step%8==0 or step==STEPS-1:
                save('training-progress.json',{'trace':trace});print(json.dumps(rec),flush=True)
        torch.save({'writer':{k:v.detach().cpu() for k,v in writer.state_dict().items()},
                    'base_sha256':WEIGHT_SHA,'parent_sha256':hashes['parent'],'seed':SEED},OUT/'candidate.pt')
        evaluate('after')
        after={k:sha(v) for k,v in paths.items()}
        save('result.json',{'rows':rows,'trace':trace,'steps':STEPS,'lr':LR,'seed':SEED,
            'input_hashes':hashes,'input_hashes_after':after,'candidate_sha256':sha(OUT/'candidate.pt'),
            'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Source cases previously examined, not independent; prepared facts and manually chosen states.',
                'Synthetic training does not establish generalization, autonomous writing or durable memory.',
                'String containment is not a semantic correctness metric; ordinary KL uses one position.']})
        print(json.dumps({'complete':True,'unchanged':hashes==after,'seconds':time.perf_counter()-start}),flush=True)
    except BaseException:
        (OUT/'error.txt').write_text(traceback.format_exc(),encoding='utf8');raise

if __name__=='__main__':main()
