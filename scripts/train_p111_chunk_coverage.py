"""Independent chunk-pretraining entry with separate story and pretrain coverage."""
import argparse,json,math,random,time,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from dataclasses import asdict
from spora.conversation import ConversationBridge,ConversationConfig,restore_conversation,generate
from spora.conversation_data import DialogueRows
from spora.token_objective import backward_token_budget
from spora.training_mix import choose_source
from spora.document_stream import DocumentStream,trim_ignored_tail
from spora.uniform_document_stream import UniformDocumentStream
from spora.document_chunk_stream import DocumentChunkStream
from spora.supervision_coverage import SupervisionCoverage
from spora.document_stream_observation import accumulate_unchanged_rows
from spora.story_coverage_run import coverage_contract, restore_coverage, sample_story, record_story, validate_identity
from spora.source_review import verify_training_source_reviews
from spora.p115_story_input import validate_candidate
from spora.dialogue_candidate import validate_candidate as validate_dialogue_candidate
from spora.dialogue_replay_coverage import DialogueReplayCoverage
from spora.growing import load
from spora.language_training import interface_state,atomic_save,frozen_digest
from spora.training_guard import (assess_guard, validate_guard, assess_multiturn_guard,
                                  validate_multiturn_guard)
from scripts.evaluate_conversation import evaluate_cases
from chinese_curriculum_common import memory_answers
from types import SimpleNamespace
from prepare_conversation_v2 import sha,SPECIAL
from train_wording import dump,protected_manifest
ROOT=Path(__file__).resolve().parents[1]
SOURCES=('spora/conversation.py','spora/conversation_data.py','scripts/train_p111_chunk_coverage.py','spora/document_chunk_stream.py','spora/uniform_document_stream.py','spora/story_coverage_run.py','spora/document_stream_observation.py','spora/supervision_coverage.py','spora/token_objective.py','spora/conversation_state.py','spora/training_mix.py','spora/document_stream.py','spora/source_review.py','spora/p115_story_input.py','spora/state_codec.py','spora/growing.py','scripts/prepare_teaching_curriculum.py','scripts/curate_teaching_curriculum.py','scripts/prepare_conversation_v4.py','scripts/prepare_conversation_v2.py','scripts/curate_conversation.py','scripts/prepare_prose_pretrain_pilot.py','spora/language_training.py')
SOURCES += ('spora/training_guard.py','scripts/evaluate_conversation.py',
            'scripts/converse_research.py','spora/conversation_metrics.py','spora/conversation_sampling.py')
SOURCES += ('spora/dialogue_candidate.py','spora/dialogue_replay_coverage.py')
PROMPTS=('你好，请简单介绍你的能力。','我今天有点累，可以怎样放松？','请用一句话解释为什么会下雨。','把“明天我去图书馆看书”改成疑问句。','我不知道该选哪本书，你能先问我一个问题吗？','社会主义核心价值观的个人层面包括哪些内容？')

def load_rows(path,length):
    return DialogueRows(path,length)

def padded(rows,length,pad,device):
    tokens=[];labels=[]
    for row in rows:
        seq=row['tokens'];mask=row['mask']
        padded_seq=seq+[pad]*(length+1-len(seq))
        target=[t if valid else -100 for t,valid in zip(seq[1:],mask[1:])]+[-100]*(length+1-len(seq))
        tokens.append(padded_seq[:-1]);labels.append(target)
    return torch.tensor(tokens,device=device),torch.tensor(labels,device=device)

def text_batch(stream,rng,batch,length,device):
    starts=[rng.randrange(len(stream)-length) for _ in range(batch)]
    seq=np.stack([np.array(stream[i:i+length+1],dtype=np.int64) for i in starts])
    return torch.tensor(seq[:,:-1],device=device),torch.tensor(seq[:,1:],device=device)

def batch_loss(model,state,x,y):
    return F.cross_entropy(model(x,state).flatten(0,1),y.flatten(),ignore_index=-100)

@torch.no_grad()
def teaching_loss(model,state,dataset,pad,seed):
    rng=random.Random(seed);total=0.;count=0
    for _ in range(8):
        rows=[dataset[rng.randrange(len(dataset))] for _ in range(4)]
        length=max(len(row['tokens']) for row in rows)-1
        x,y=padded(rows,length,pad,next(model.parameters()).device);n=int((y!=-100).sum())
        total+=float(batch_loss(model,state,x,y))*n;count+=n
    return total/count

@torch.no_grad()
def monitor(model,state,tokenizer,development,dialogue_dev,length,pad):
    rng=random.Random(230101);device=next(model.parameters()).device
    total=0.;count=0
    for _ in range(8):
        x,y=text_batch(development,rng,4,length,device);loss=batch_loss(model,state,x,y)
        total+=float(loss)*y.numel();count+=y.numel()
    sft_total=0.;sft_count=0
    for start in range(0,min(32,len(dialogue_dev)),4):
        x,y=padded(dialogue_dev[start:start+4],length,pad,device);n=int((y!=-100).sum())
        sft_total+=float(batch_loss(model,state,x,y))*n;sft_count+=n
    generated=[];eos=tokenizer.token_to_id('<|eos|>');bos=tokenizer.token_to_id('<|bos|>')
    user=tokenizer.token_to_id('<|user|>');assistant=tokenizer.token_to_id('<|assistant|>')
    for prompt in PROMPTS:
        prefix=[bos,user]+tokenizer.encode(prompt).ids+[eos,assistant]
        result=generate(model,state,prefix,eos,budget=96,forbidden=(pad,bos,user,assistant))
        text=tokenizer.decode(result['tokens'],skip_special_tokens=False)
        result.update(prompt=prompt,text=text,utf8_valid='\ufffd' not in text)
        generated.append(result)
    return dict(pretrain_nll=total/count,dialogue_nll=sft_total/sft_count,samples=generated,
                scope='Fixed development samples, not a chat acceptance test. NLL only comparable within this tokenizer.')

def story_sampling_mode(plan):
    mode = plan.get('story_sampling_mode', 'legacy')
    if mode not in ('legacy', 'uniform'):
        raise ValueError('story_sampling_mode must be legacy or uniform')
    if mode == 'uniform' and plan.get('respect_story_document_boundaries') is not True:
        raise ValueError('Uniform story sampling requires document boundaries')
    return mode


def make_story_documents(stream, bos, eos, pad, plan):
    mode = story_sampling_mode(plan)
    if not plan.get('respect_story_document_boundaries', False):
        return None
    if mode == 'uniform':
        return UniformDocumentStream(stream, bos, eos, pad, plan['length'])
    return DocumentStream(stream, bos, eos, pad)


def chunk_contract(plan):
    if plan.get('pretrain_sampling_mode') != 'chunk':
        raise ValueError('pretrain_sampling_mode must be chunk')
    if plan.get('respect_document_boundaries') is not True:
        raise ValueError('Chunk sampling requires pretrain document boundaries')
    if plan.get('observe_pretrain_coverage') is not True:
        raise ValueError('Chunk sampling requires pretrain coverage')
    story = coverage_contract(plan)
    if not story['record_story']:
        raise ValueError('Story coverage and story document boundaries are required')
    if type(plan.get('length')) is not int or plan['length'] < 1:
        raise ValueError('length must be a positive integer')
    return {'version': 1, 'mode': 'chunk', 'window_length': plan['length'],
            'scope': 'pretrain_supervised_positions_only',
            'excluded_sources': ['focus_replacements']}


def restore_chunk_coverage(payload, contract, source, documents):
    if payload is None:
        return SupervisionCoverage()
    if (payload.get('chunk_coverage_entry') != 'spora-chunk-coverage-v1'
            or payload.get('chunk_coverage_contract') != contract
            or payload.get('pretrain_source_sha256') != source):
        raise ValueError('Incompatible chunk coverage identity')
    state = payload.get('chunk_coverage')
    if not isinstance(state, dict) or state.get('version') != 2:
        raise ValueError('Missing or unsupported chunk coverage state')
    coverage = SupervisionCoverage.from_dict(state)
    total = 0
    for (digest, document), summary in coverage.summaries().items():
        if digest != source.lower() or document >= len(documents.starts):
            raise ValueError('Chunk coverage source or document differs')
        length = int(documents.ends[document] - documents.starts[document])
        if any(start < 1 or end > length for start, end in summary.intervals):
            raise ValueError('Chunk coverage position outside document')
        total += summary.total_positions
    counts = payload.get('counts')
    if not isinstance(counts, dict):
        raise ValueError('Missing source counts')
    expected = counts.get('pretrain_source_targets', 0)
    if type(expected) is not int or expected < 0 or total != expected:
        raise ValueError('Coverage and supervised pretrain counts differ')
    return coverage


def sample_chunk(documents, rng, batch_size, length, device):
    x, y, windows = documents.batch_with_windows(rng, batch_size, length, device)
    return x, y, windows, y.clone()


def record_chunk(coverage, source, windows, sampled_labels, final_labels, replaced_rows):
    if windows is None:
        return 0
    return accumulate_unchanged_rows(coverage, source, windows, sampled_labels, final_labels, replaced_rows)


def record_dialogue(coverage, source, rows, replaced_rows):
    recorded = 0
    for index, row in enumerate(rows):
        if index not in replaced_rows:
            coverage.add(source, row['pair_id'])
            recorded += sum(row['mask'][1:])
    return recorded


def exclude_run_outputs(protected, root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    runs = (root / 'runs').resolve()
    if not output.is_relative_to(runs) or output == runs:
        raise ValueError('Training output must be a directory inside runs')
    return {name: digest for name, digest in protected.items()
            if not (root / name).resolve().is_relative_to(output)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--until',type=int,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();plan=json.loads((ROOT/args.plan).read_text('utf-8'));data=ROOT/plan['data'];base_data=ROOT/plan.get('base_data',plan['data']);out=ROOT/args.output
    if not 0<args.until<=plan['schedule_steps'] or args.until%plan['check_every']:raise ValueError('End at a planned checkpoint')
    guard_config=plan.get('regression_guard');validate_guard(guard_config)
    multi_config=plan.get('multiturn_guard');validate_multiturn_guard(multi_config)
    multi_cases=None
    if multi_config is not None:
        case_path=(ROOT/multi_config['cases']).resolve()
        if not case_path.is_relative_to((ROOT/'data').resolve()) or sha(case_path)!=multi_config['cases_sha256']:
            raise ValueError('Multi-turn development cases changed or outside data')
        multi_cases=json.loads(case_path.read_text('utf8'))
    mode=story_sampling_mode(plan)
    contract=coverage_contract(plan)
    chunk_spec=chunk_contract(plan)
    if sha(ROOT/plan['parent'])!=plan['parent_sha256']:raise ValueError('Individual file changed')
    if sha(ROOT/plan['corpus_review'])!=plan['corpus_review_sha256']:raise ValueError('Corpus review changed')
    review=json.loads((ROOT/plan['corpus_review']).read_text('utf-8'))
    if review.get('approved_for_pilot') is not True:raise ValueError('Content review has no pilot approval')
    curriculum=ROOT/plan['curriculum']
    review_bound = str(review.get('data_manifest_sha256', '')).lower() == sha(data/'manifest.json').lower()
    # A curriculum review may bind the exact curriculum directory rather than
    # the mixed base data directory. This keeps custom reviewed courses
    # traceable without weakening the artifact hash checks below.
    curriculum_sha = sha(curriculum/'manifest.json').lower()
    review_bound = review_bound or str(review.get('input_directory_manifest_sha256', '')).lower() == curriculum_sha
    review_bound = review_bound or str(review.get('manifest_sha256', '')).lower() == curriculum_sha
    if plan.get('review_source_manifest'):
        review_bound = review_bound or str(review.get('source_manifest_sha256', '')).lower() == sha(ROOT/plan['review_source_manifest']).lower()
    if not review_bound:raise ValueError('Content review covers different training data')
    manifest=json.loads((data/'manifest.json').read_text('utf-8'))
    candidate = plan.get('candidate_story')
    candidate_identity = None
    story_path = data/'pretrain-train.bin'
    if candidate is not None:
        if (not isinstance(candidate, dict) or not isinstance(candidate.get('path'), str)
                or not isinstance(candidate.get('summary_sha256'), str) or mode != 'uniform'):
            raise ValueError('Candidate story requires path, summary hash and uniform sampling')
        candidate_dir = ROOT/candidate['path']
        candidate_identity = validate_candidate(candidate_dir, candidate['summary_sha256'], manifest['tokenizer_sha256'], plan['length'], plan['model']['vocabulary'])
        story_path = candidate_dir/'stories.bin'
    dialogue_candidate = plan.get('candidate_dialogue')
    dialogue_identity = None
    dialogue_path = data/'dialogue-train-tokens.jsonl'
    if dialogue_candidate is not None:
        if (not isinstance(dialogue_candidate, dict) or set(dialogue_candidate) != {'path', 'manifest_sha256'}
                or not isinstance(dialogue_candidate['path'], str) or not isinstance(dialogue_candidate['manifest_sha256'], str)
                or 'dialogue' not in plan['target_shares']):
            raise ValueError('Candidate dialogue requires path, manifest hash and dialogue share')
        dialogue_dir = ROOT/dialogue_candidate['path']
        dialogue_identity = validate_dialogue_candidate(dialogue_dir, dialogue_candidate['manifest_sha256'],
            manifest['tokenizer_sha256'], plan['length'], plan['model']['vocabulary'])
        dialogue_path = dialogue_dir/'context.jsonl'
    source_reviews=verify_training_source_reviews(ROOT,plan, None if candidate_identity is None else
        {'path': story_path, 'summary_sha256': candidate_identity['summary_sha256']},
        None if dialogue_identity is None else {'path': dialogue_path, 'manifest_sha256': dialogue_identity['manifest_sha256']})
    if candidate_identity is not None and source_reviews.get('story', {}).get('artifact_sha256') != candidate_identity['story_sha256']:
        raise ValueError('Candidate story review does not match validated artifact')
    story_source = manifest['files']['pretrain-train.bin'] if candidate_identity is None else candidate_identity['story_sha256']
    audit_paths=plan['data_audits'];audits={}
    for name in audit_paths:
        report=json.loads((ROOT/name).read_text('utf-8'))
        if not report['passed']:raise ValueError('Data isolation audit failed: '+name)
        for artifact,digest in report.get('data_artifact_sha256',{}).items():
            if manifest['files'][artifact]!=digest:raise ValueError('Audit covers different data')
        lineage=report.get('manifest_sha256')
        if lineage is not None:lineage={key.replace('\\','/'):value for key,value in lineage.items()}
        if lineage is not None and lineage[plan['data']]!=sha(data/'manifest.json'):raise ValueError('Lineage audit is stale')
        audits[name]=sha(ROOT/name)
    # Verify every local input used for training and monitoring, not final test.
    inputs=('tokenizer.json','pretrain-train.bin','pretrain-development.bin','dialogue-train-tokens.jsonl','dialogue-development-tokens.jsonl','focus-tokens.json')
    for name in inputs:
        if sha(data/name)!=manifest['files'][name]:raise ValueError('Dataset artifact changed: '+name)
    if base_data != data:
        base_manifest=json.loads((base_data/'manifest.json').read_text('utf-8'))
        for name in ('tokenizer.json','pretrain-train.bin'):
            if sha(base_data/name)!=base_manifest['files'][name]:raise ValueError('Base dataset artifact changed: '+name)
        if base_manifest['tokenizer_sha256']!=manifest['tokenizer_sha256']:raise ValueError('Base vocabulary differs')
    else:
        base_manifest=manifest
    tokenizer=Tokenizer.from_file(str(data/'tokenizer.json'));pad=tokenizer.token_to_id('<|pad|>')
    story_stream=np.memmap(story_path,dtype='<u2',mode='r')
    story_documents = make_story_documents(story_stream, tokenizer.token_to_id('<|bos|>'), tokenizer.token_to_id('<|eos|>'), pad, plan)
    cm=json.loads((curriculum/'manifest.json').read_text('utf-8'))
    if cm['tokenizer_sha256']!=manifest['tokenizer_sha256']:raise ValueError('Teaching vocabulary differs')
    if sha(curriculum/'manifest.json')!=plan['curriculum_manifest_sha256']:raise ValueError('Teaching manifest changed')
    for name,digest in cm['files'].items():
        if sha(curriculum/name)!=digest:raise ValueError('Teaching artifact changed: '+name)
    protected=protected_manifest()
    for pattern in ('p27b-token-budget/**/*.pt','p27-coverage*/**/*.pt','p21*/**/*.pt','p22*/**/*.pt','p23a*/**/*.pt','p23b*/**/*.pt','p23c*/**/*.pt','p23d*/**/*.pt','p23e*/**/*.pt','p23f*/**/*.pt','p24*/**/*.pt','p26*/**/*.pt'):
        for path in (ROOT/'runs').glob(pattern):protected[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path)
    protected[plan['initial_language']]=plan['initial_language_sha256']
    protected=exclude_run_outputs(protected,ROOT,out)
    if plan['initial_language'] not in protected:
        raise ValueError('Training output cannot contain its initial checkpoint')
    fingerprints={name:sha(ROOT/name) for name in SOURCES}
    metadata=dict(entry='spora-p111-chunk-coverage-v1',story_sampling_mode=mode,pretrain_sampling_mode='chunk',chunk_coverage_contract=chunk_spec,coverage_contract=contract,plan=plan,plan_sha256=sha(ROOT/args.plan),manifest_sha256=sha(data/'manifest.json'),base_manifest_sha256=sha(base_data/'manifest.json'),candidate_story=candidate_identity,candidate_dialogue=dialogue_identity,data_audit_sha256=audits,source_reviews=source_reviews,source_code_sha256=fingerprints,protected=protected)
    torch.set_num_threads(2);torch.manual_seed(plan['seed']);rng=random.Random(plan['seed'])
    organism,state,_=load(ROOT/plan['parent'],'cuda');before=frozen_digest(organism,state)
    payload=None;step=0;targets=0;focus_cursor=0;counts={};logs=[];elapsed=0.;batch_trace='00'*32
    if args.resume:
        if json.loads((out/'config.json').read_text('utf-8'))!=metadata:raise ValueError('Resume requires exact implementation/data/plan/protected sources')
        payload=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
        if payload.get('regression_guard', {}).get('stop'):
            raise ValueError('Stopped regression checkpoint requires a new reviewed experiment')
        if payload.get('multiturn_guard', {}).get('stop'):
            raise ValueError('Stopped multi-turn checkpoint requires a new reviewed experiment')
        if guard_config is not None and not isinstance(payload.get('guard_baseline'), dict):
            raise ValueError('Missing regression baseline')
        if multi_config is not None and not isinstance(payload.get('multiturn_baseline'), dict):
            raise ValueError('Missing multi-turn baseline')
        validate_identity(payload,contract,fingerprints,story_source)
        if payload.get('story_sampling_mode') != mode:raise ValueError('Sampling mode differs from checkpoint')
        model=restore_conversation(organism,payload,plan['parent_sha256'],manifest['tokenizer_sha256'],'cuda')
        step=payload['steps'];targets=payload['targets'];focus_cursor=payload['focus_cursor'];counts=payload['counts'];elapsed=payload['seconds']
        if args.until<=step:raise ValueError('Resume must make progress')
        rng.setstate(payload['python_rng']);torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])
        logs=payload['development_logs'];batch_trace=payload['batch_trace_sha256']
    else:
        out.mkdir(exist_ok=False);dump(out/'config.json',metadata)
        initial=ROOT/plan['initial_language']
        if sha(initial)!=plan['initial_language_sha256']:raise ValueError('Language parent changed')
        language_payload=torch.load(initial,map_location='cpu',weights_only=True)
        model=restore_conversation(organism,language_payload,plan['parent_sha256'],manifest['tokenizer_sha256'],'cuda')
        if asdict(model.config)!={**plan['model'],'memory_layers':tuple(plan['model']['memory_layers'])}:raise ValueError('Architecture differs from language parent')
        # Explicit curriculum fork: retain learned tensors, reset optimizer and
        # schedule, keep ancestor steps/targets in plan (not silently resuming).
        if language_payload['steps']!=plan['ancestor_steps'] or language_payload['targets']!=plan['ancestor_targets']:raise ValueError('Ancestor accounting differs')
    if plan.get('memory_ablation', False):
        model.memory_gate.data.fill_(-10.)
        model.memory_gate.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=plan['learning_rate'],weight_decay=.01,betas=(.9,.95))
    if payload is not None:optimizer.load_state_dict(payload['optimizer'])
    stream=np.memmap(base_data/'pretrain-train.bin',dtype=np.uint16,mode='r');dev=np.memmap(data/'pretrain-development.bin',dtype=np.uint16,mode='r')
    document_stream = DocumentChunkStream(stream, tokenizer.token_to_id('<|bos|>'), tokenizer.token_to_id('<|eos|>'), pad, plan['length'])
    coverage=restore_coverage(payload, contract, story_source, story_documents)
    chunk_source=base_manifest['files']['pretrain-train.bin']
    chunk_coverage=restore_chunk_coverage(payload, chunk_spec, chunk_source, document_stream)
    sft=load_rows(dialogue_path,plan['length']);sft_dev=load_rows(data/'dialogue-development-tokens.jsonl',plan['length'])
    dialogue_coverage = None
    dialogue_source = None
    if dialogue_identity is not None:
        dialogue_source = dialogue_identity['files']['context.jsonl']
        dialogue_records = [dict(source=dialogue_source, pair_id=row['pair_id'], mask=row['mask']) for row in sft]
        if len(dialogue_records) != dialogue_identity['pairs']:
            raise ValueError('Candidate dialogue loader excluded records')
        dialogue_coverage = (DialogueReplayCoverage(dialogue_records) if payload is None else
                             DialogueReplayCoverage.from_dict(dialogue_records, payload['dialogue_coverage']))
        if sum(s['target_presentations'] for s in dialogue_coverage.summaries().values()) != counts.get('dialogue_source_targets', 0):
            raise ValueError('Dialogue coverage differs from target count')
    teacher=load_rows(curriculum/'teacher-tokens.jsonl',plan['length'])
    foundation=load_rows(curriculum/'foundations-tokens.jsonl',plan['length'])
    if not teacher or not foundation:raise ValueError('Missing teaching rows')
    focus=json.loads((data/'focus-tokens.json').read_text('utf-8'))
    if not sft or not sft_dev or not focus:raise ValueError('Missing curriculum')
    if payload is None:
        baseline=monitor(model,state,tokenizer,dev,sft_dev,plan['length'],pad)
        if guard_config is not None:assess_guard(guard_config,baseline,baseline)
        baseline.update(teacher_training_nll=teaching_loss(model,state,teacher,pad,231201),foundation_training_nll=teaching_loss(model,state,foundation,pad,231202))
        dump(out/'baseline.json',baseline)
        atomic_save(dict(format='spora-conversation-v1',organism_sha256=plan['parent_sha256'],tokenizer_sha256=manifest['tokenizer_sha256'],config=asdict(model.config),state=interface_state(model)),out/'language-initial.pt')
        if multi_config is not None:
            summary,rows=evaluate_cases(model,state,tokenizer,multi_cases,multi_config['budget'])
            multi_baseline={'turns':len(rows),'ended':sum(r['ended_with_eos'] for r in rows),
                            'loops':sum(r['obvious_loop'] for r in rows)}
            assess_multiturn_guard(multi_config,multi_baseline,multi_baseline)
            dump(out/'multiturn-000000.json',{'summary':summary,'rows':rows})
    else:
        baseline=payload.get('guard_baseline')
        multi_baseline=payload.get('multiturn_baseline')
    guard_result=multi_result=None
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    print(json.dumps(dict(start_step=step,until=args.until,parameters=sum(p.numel() for p in model.trainable_parameters()),sft_rows=len(sft),teacher_rows=len(teacher),foundation_rows=len(foundation),focus_segments=len(focus))),flush=True)
    for step in range(step+1,args.until+1):
        fraction=min(step/plan['warmup'],1.) if step<=plan['warmup'] else .1+.9*.5*(1+math.cos(math.pi*(step-plan['warmup'])/(plan['schedule_steps']-plan['warmup'])))
        for group in optimizer.param_groups:group['lr']=plan['learning_rate']*fraction
        batches=[];update_targets=0;focus_used=False
        while update_targets<plan['target_budget']:
            kind=choose_source(plan['target_shares'],counts)
            windows=None;sampled_labels=None;chunk_windows=None;chunk_labels=None
            if kind=='story':
                if story_documents is not None:
                    x,y,windows,sampled_labels=sample_story(story_documents,rng,plan['microbatch'],plan['length'],'cpu',contract['record_story'])
                else:
                    x,y=text_batch(story_stream,rng,plan['microbatch'],plan['length'],'cpu')
            elif kind=='pretrain':
                x,y,chunk_windows,chunk_labels=sample_chunk(document_stream,rng,plan['microbatch'],plan['length'],'cpu')
            else:
                source={'dialogue':sft,'teacher':teacher,'foundation':foundation}[kind]
                rows=[rng.choice(source) for _ in range(plan['microbatch'])]
                length=min(plan['length'],32*math.ceil((max(len(r['tokens']) for r in rows)-1)/32))
                x,y=padded(rows,length,pad,'cpu')
            focus_n=0;replaced_rows=set()
            if step%plan['focus_every']==0 and not focus_used:
                seq=focus[focus_cursor%len(focus)][:x.shape[1]+1];focus_cursor+=1;focus_used=True
                fx,fy=padded([dict(tokens=seq,mask=[True]*len(seq))],x.shape[1],pad,'cpu')
                x[0]=fx[0];y[0]=fy[0];replaced_rows.add(0);focus_n=int((fy!=-100).sum())
                counts['focus_targets']=counts.get('focus_targets',0)+focus_n
            n=int((y!=-100).sum())
            if n==0:raise ValueError('Source produced no supervision')
            record_story(coverage,story_source,windows,sampled_labels,y,replaced_rows)
            observed=record_chunk(chunk_coverage,chunk_source,chunk_windows,chunk_labels,y,replaced_rows)
            if kind=='pretrain' and observed != n-focus_n:raise AssertionError('Chunk coverage differs from supervised targets')
            if kind=='dialogue' and dialogue_coverage is not None:
                recorded=record_dialogue(dialogue_coverage,dialogue_source,rows,replaced_rows)
                if recorded != n-focus_n:raise AssertionError('Dialogue coverage differs from supervised targets')
            if plan.get('trim_ignored_tail',False):x,y=trim_ignored_tail(x,y)
            batches.append((x,y));update_targets+=n
            counts[kind+'_microbatches']=counts.get(kind+'_microbatches',0)+1
            counts[kind+'_targets_including_focus']=counts.get(kind+'_targets_including_focus',0)+n
            counts[kind+'_source_targets']=counts.get(kind+'_source_targets',0)+n-focus_n
            batch_trace=hashlib.sha256(bytes.fromhex(batch_trace)+x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
        optimizer.zero_grad(set_to_none=True)
        loss,actual=backward_token_budget(model,state,batches,plan.get('unlikelihood_weight',0.0))
        if actual!=update_targets:raise AssertionError('Objective denominator differs from sampler')
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.,error_if_nonfinite=True);optimizer.step()
        targets+=actual
        if step%50==0:print(json.dumps(dict(step=step,loss=float(loss.detach()),targets=targets,seconds=elapsed+time.perf_counter()-start)),flush=True)
        if step%plan['check_every']==0:
            report=monitor(model,state,tokenizer,dev,sft_dev,plan['length'],pad)
            guard_result=assess_guard(guard_config,baseline,report)
            if multi_config is not None:
                summary,rows=evaluate_cases(model,state,tokenizer,multi_cases,multi_config['budget'])
                multi_current={'turns':len(rows),'ended':sum(r['ended_with_eos'] for r in rows),
                               'loops':sum(r['obvious_loop'] for r in rows)}
                multi_result=assess_multiturn_guard(multi_config,multi_baseline,multi_current)
                dump(out/f'multiturn-{step:06d}.json',{'summary':summary,'rows':rows})
            report.update(teacher_training_nll=teaching_loss(model,state,teacher,pad,231201),foundation_training_nll=teaching_loss(model,state,foundation,pad,231202))
            if before!=frozen_digest(organism,state):raise AssertionError('Original individual changed')
            row=dict(step=step,loss=float(loss.detach()),targets=targets,**report);logs.append(row)
            snapshot=dict(format='spora-conversation-v1',organism_sha256=plan['parent_sha256'],tokenizer_sha256=manifest['tokenizer_sha256'],
                config=asdict(model.config),state=interface_state(model),optimizer=optimizer.state_dict(),steps=step,targets=targets,
                focus_cursor=focus_cursor,counts=counts,batch_trace_sha256=batch_trace,python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                seconds=elapsed+time.perf_counter()-start,plan_sha256=metadata['plan_sha256'],manifest_sha256=metadata['manifest_sha256'],
                coverage_entry='spora-story-coverage-v1',story_sampling_mode=mode,source_code_sha256=fingerprints,coverage_contract=contract,
                story_source_sha256=story_source,coverage=coverage.to_dict(),
                chunk_coverage_entry='spora-chunk-coverage-v1',chunk_coverage_contract=chunk_spec,
                pretrain_source_sha256=chunk_source,chunk_coverage=chunk_coverage.to_dict(),development_logs=logs)
            if guard_config is not None:
                snapshot['guard_baseline']=baseline
                snapshot['regression_guard']=guard_result
            if multi_config is not None:
                snapshot['multiturn_baseline']=multi_baseline
                snapshot['multiturn_guard']=multi_result
            if dialogue_coverage is not None:
                snapshot['dialogue_coverage']=dialogue_coverage.to_dict()
                restored=DialogueReplayCoverage.from_dict(dialogue_records,snapshot['dialogue_coverage'])
                if sum(s['target_presentations'] for s in restored.summaries().values()) != counts.get('dialogue_source_targets', 0):
                    raise AssertionError('Dialogue coverage and target counts differ')
            restore_coverage(snapshot, contract, story_source, story_documents)
            restore_chunk_coverage(snapshot, chunk_spec, chunk_source, document_stream)
            atomic_save(snapshot,out/'checkpoint.pt')
            if step%plan['snapshot_every']==0:atomic_save(snapshot,out/f'step{step:06d}.pt')
            dump(out/'development.json',logs)
            print(json.dumps(dict(step=step,pretrain_nll=report['pretrain_nll'],dialogue_nll=report['dialogue_nll'],samples=[dict(prompt=r['prompt'],characters=len(r['text']),ended=r['ended_with_eos'],utf8_valid=r['utf8_valid']) for r in report['samples']])),flush=True)
            if (guard_result is not None and guard_result['stop']) or (multi_result is not None and multi_result['stop']):
                print(json.dumps({'regression_guard':guard_result,'multiturn_guard':multi_result}),flush=True)
                break
    for name,digest in protected.items():
        if sha(ROOT/name)!=digest:raise AssertionError('Protected artifact changed: '+name)
    if fingerprints!={name:sha(ROOT/name) for name in SOURCES}:raise AssertionError('Implementation changed during segment')
    if candidate_identity is not None:
        if (sha(candidate_dir/'summary.json') != candidate_identity['summary_sha256'] or
                any(sha(candidate_dir/name) != digest for name,digest in candidate_identity['files'].items())):
            raise AssertionError('Candidate changed during segment')
    if dialogue_identity is not None:
        if (sha(dialogue_dir/'manifest.json') != dialogue_identity['manifest_sha256'] or
                any(sha(dialogue_dir/name) != digest for name,digest in dialogue_identity['files'].items())):
            raise AssertionError('Dialogue candidate changed during segment')
    stopped=(guard_result and guard_result['stop']) or (multi_result and multi_result['stop'])
    result=dict(status='stopped_regression_not_chat_accepted' if stopped else 'segment_completed_not_chat_accepted',steps=step,targets=targets,counts=counts,focus_cursor=focus_cursor,
        batch_trace_sha256=batch_trace,old_tensors_exact=before==frozen_digest(organism,state),memory=memory_answers(SimpleNamespace(model=model,state=state)),
        seconds=elapsed+time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=sha(out/'checkpoint.pt'))
    source_total=sum(counts.get(k+'_source_targets',0) for k in plan['target_shares'])
    result['coverage_contract']=contract
    result['chunk_coverage_contract']=chunk_spec
    if dialogue_coverage is not None:
        summaries=dialogue_coverage.summaries().values()
        result['dialogue_coverage']={key:sum(row[key] for row in summaries) for key in
            ('presentations','target_presentations','distinct_target_positions','duplicate_target_positions')}
    if guard_config is not None:result['regression_guard']=guard_result
    if multi_config is not None:result['multiturn_guard']=multi_result
    result['realized_source_shares']={k:counts.get(k+'_source_targets',0)/max(1,source_total) for k in plan['target_shares']}
    dump(out/f'segment-{step:06d}.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
