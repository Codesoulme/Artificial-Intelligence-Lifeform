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
from spora.growing import load
from spora.language_training import interface_state,atomic_save,frozen_digest
from chinese_curriculum_common import memory_answers
from types import SimpleNamespace
from prepare_conversation_v2 import sha,SPECIAL
from train_wording import dump,protected_manifest
ROOT=Path(__file__).resolve().parents[1]
SOURCES=('spora/conversation.py','spora/conversation_data.py','scripts/train_p111_chunk_coverage.py','spora/document_chunk_stream.py','spora/uniform_document_stream.py','spora/story_coverage_run.py','spora/document_stream_observation.py','spora/supervision_coverage.py','spora/token_objective.py','spora/conversation_state.py','spora/training_mix.py','spora/document_stream.py','spora/source_review.py','spora/state_codec.py','spora/growing.py','scripts/prepare_teaching_curriculum.py','scripts/curate_teaching_curriculum.py','scripts/prepare_conversation_v4.py','scripts/prepare_conversation_v2.py','scripts/curate_conversation.py','scripts/prepare_prose_pretrain_pilot.py','spora/language_training.py')
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

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--until',type=int,required=True);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();plan=json.loads((ROOT/args.plan).read_text('utf-8'));data=ROOT/plan['data'];base_data=ROOT/plan.get('base_data',plan['data']);out=ROOT/args.output
    if not 0<args.until<=plan['schedule_steps'] or args.until%plan['check_every']:raise ValueError('End at a planned checkpoint')
    mode=story_sampling_mode(plan)
    contract=coverage_contract(plan)
    chunk_spec=chunk_contract(plan)
    source_reviews=verify_training_source_reviews(ROOT,plan)
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
    story_stream=np.memmap(data/'pretrain-train.bin',dtype=np.uint16,mode='r')
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
    fingerprints={name:sha(ROOT/name) for name in SOURCES}
    metadata=dict(entry='spora-p111-chunk-coverage-v1',story_sampling_mode=mode,pretrain_sampling_mode='chunk',chunk_coverage_contract=chunk_spec,coverage_contract=contract,plan=plan,plan_sha256=sha(ROOT/args.plan),manifest_sha256=sha(data/'manifest.json'),base_manifest_sha256=sha(base_data/'manifest.json'),data_audit_sha256=audits,source_reviews=source_reviews,source_code_sha256=fingerprints,protected=protected)
    torch.set_num_threads(2);torch.manual_seed(plan['seed']);rng=random.Random(plan['seed'])
    organism,state,_=load(ROOT/plan['parent'],'cuda');before=frozen_digest(organism,state)
    payload=None;step=0;targets=0;focus_cursor=0;counts={};logs=[];elapsed=0.;batch_trace='00'*32
    if args.resume:
        if json.loads((out/'config.json').read_text('utf-8'))!=metadata:raise ValueError('Resume requires exact implementation/data/plan/protected sources')
        payload=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=True)
        validate_identity(payload,contract,fingerprints,manifest['files']['pretrain-train.bin'])
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
    coverage=restore_coverage(payload, contract, manifest['files']['pretrain-train.bin'], story_documents)
    chunk_source=base_manifest['files']['pretrain-train.bin']
    chunk_coverage=restore_chunk_coverage(payload, chunk_spec, chunk_source, document_stream)
    sft=load_rows(data/'dialogue-train-tokens.jsonl',plan['length']);sft_dev=load_rows(data/'dialogue-development-tokens.jsonl',plan['length'])
    teacher=load_rows(curriculum/'teacher-tokens.jsonl',plan['length'])
    foundation=load_rows(curriculum/'foundations-tokens.jsonl',plan['length'])
    if not teacher or not foundation:raise ValueError('Missing teaching rows')
    focus=json.loads((data/'focus-tokens.json').read_text('utf-8'))
    if not sft or not sft_dev or not focus:raise ValueError('Missing curriculum')
    if payload is None:
        baseline=monitor(model,state,tokenizer,dev,sft_dev,plan['length'],pad)
        baseline.update(teacher_training_nll=teaching_loss(model,state,teacher,pad,231201),foundation_training_nll=teaching_loss(model,state,foundation,pad,231202))
        dump(out/'baseline.json',baseline)
        atomic_save(dict(format='spora-conversation-v1',organism_sha256=plan['parent_sha256'],tokenizer_sha256=manifest['tokenizer_sha256'],config=asdict(model.config),state=interface_state(model)),out/'language-initial.pt')
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
            record_story(coverage,manifest['files']['pretrain-train.bin'],windows,sampled_labels,y,replaced_rows)
            observed=record_chunk(chunk_coverage,chunk_source,chunk_windows,chunk_labels,y,replaced_rows)
            if kind=='pretrain' and observed != n-focus_n:raise AssertionError('Chunk coverage differs from supervised targets')
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
            report.update(teacher_training_nll=teaching_loss(model,state,teacher,pad,231201),foundation_training_nll=teaching_loss(model,state,foundation,pad,231202))
            if before!=frozen_digest(organism,state):raise AssertionError('Original individual changed')
            row=dict(step=step,loss=float(loss.detach()),targets=targets,**report);logs.append(row)
            snapshot=dict(format='spora-conversation-v1',organism_sha256=plan['parent_sha256'],tokenizer_sha256=manifest['tokenizer_sha256'],
                config=asdict(model.config),state=interface_state(model),optimizer=optimizer.state_dict(),steps=step,targets=targets,
                focus_cursor=focus_cursor,counts=counts,batch_trace_sha256=batch_trace,python_rng=rng.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                seconds=elapsed+time.perf_counter()-start,plan_sha256=metadata['plan_sha256'],manifest_sha256=metadata['manifest_sha256'],
                coverage_entry='spora-story-coverage-v1',story_sampling_mode=mode,source_code_sha256=fingerprints,coverage_contract=contract,
                story_source_sha256=manifest['files']['pretrain-train.bin'],coverage=coverage.to_dict(),
                chunk_coverage_entry='spora-chunk-coverage-v1',chunk_coverage_contract=chunk_spec,
                pretrain_source_sha256=chunk_source,chunk_coverage=chunk_coverage.to_dict(),development_logs=logs)
            restore_coverage(snapshot, contract, manifest['files']['pretrain-train.bin'], story_documents)
            restore_chunk_coverage(snapshot, chunk_spec, chunk_source, document_stream)
            atomic_save(snapshot,out/'checkpoint.pt')
            if step%plan['snapshot_every']==0:atomic_save(snapshot,out/f'step{step:06d}.pt')
            dump(out/'development.json',logs)
            print(json.dumps(dict(step=step,pretrain_nll=report['pretrain_nll'],dialogue_nll=report['dialogue_nll'],samples=[dict(prompt=r['prompt'],characters=len(r['text']),ended=r['ended_with_eos'],utf8_valid=r['utf8_valid']) for r in report['samples']])),flush=True)
    for name,digest in protected.items():
        if sha(ROOT/name)!=digest:raise AssertionError('Protected artifact changed: '+name)
    if fingerprints!={name:sha(ROOT/name) for name in SOURCES}:raise AssertionError('Implementation changed during segment')
    result=dict(status='segment_completed_not_chat_accepted',steps=step,targets=targets,counts=counts,focus_cursor=focus_cursor,
        batch_trace_sha256=batch_trace,old_tensors_exact=before==frozen_digest(organism,state),memory=memory_answers(SimpleNamespace(model=model,state=state)),
        seconds=elapsed+time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=sha(out/'checkpoint.pt'))
    source_total=sum(counts.get(k+'_source_targets',0) for k in plan['target_shares'])
    result['coverage_contract']=contract
    result['chunk_coverage_contract']=chunk_spec
    result['realized_source_shares']={k:counts.get(k+'_source_targets',0)/max(1,source_total) for k in plan['target_shares']}
    dump(out/f'segment-{step:06d}.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
