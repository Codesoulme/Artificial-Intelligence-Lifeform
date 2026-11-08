"""Trace and conservatively group P104 exact links; never print or export text."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import p117_dialogue_duplicate_audit as base
from p104_screen import assess

ROOT = base.ROOT
AGG = ROOT/'runs/p117-dialogue-duplicate-audit/aggregate.json'
OUT = ROOT/'runs/p117-dialogue-link-review-v2'
# Exact self-contained stock responses only. No unrecognized response is declared substantive.
GENERIC = re.compile(r'^(好的|好|明白了|我明白了|谢谢|谢谢你|感谢|不客气|没问题|可以|是的|对|当然|收到|再见|抱歉|对不起)[。！!,.，]*$')


def sha(b): return hashlib.sha256(b).hexdigest()


def links(rows):
    """Return exact-match evidence, group pairs, line coordinates, and text hashes."""
    turns=defaultdict(list); dialogues=defaultdict(list)
    for group, part, line, messages in rows:
        if not isinstance(messages,list) or any(not isinstance(m,dict) or m.get('role') not in ('user','assistant') or not isinstance(m.get('content'),str) for m in messages): continue
        seq=[(m['role'],base.normalize(m['content'])) for m in messages]
        if seq: dialogues[tuple(seq)].append((group,line))
        for pos,(role,text) in enumerate(seq):
            if not text or (role=='user' and len(text)<base.MIN_USER): continue
            turns[(role,text)].append((group,line,pos,seq))
    edges=defaultdict(lambda: {'evidence':[], 'qa_shared':False, 'all_generic':True})
    def add(members, role, text, dtype):
        for i,a in enumerate(members):
            for b in members[i+1:]:
                if a[0]==b[0]: continue
                key=tuple(sorted((a[0],b[0])))
                edge=edges[key]
                evidence={'kind':dtype,'role':role,'text_sha256':sha(text.encode()),'source_lines':sorted({a[1],b[1]})}
                if evidence not in edge['evidence']: edge['evidence'].append(evidence)
                generic=(dtype=='turn' and role=='assistant' and bool(GENERIC.fullmatch(text)))
                edge['all_generic'] &= generic
                if dtype=='dialogue': edge['qa_shared']=True
                elif role=='user':
                    ap=a[2]; bp=b[2]
                    if ap+1<len(a[3]) and bp+1<len(b[3]):
                        x=a[3][ap+1]; y=b[3][bp+1]
                        if x==y and x[0]=='assistant' and len(x[1])>=20 and not GENERIC.fullmatch(x[1]): edge['qa_shared']=True
    for (role,text),members in turns.items(): add(members,role,text,'turn')
    for seq,members in dialogues.items():
        if len({x[0] for x in members})>1:
            encoded=json.dumps(seq,ensure_ascii=False,separators=(',',':'))
            add(members,'dialogue',encoded,'dialogue')
    result=[]
    for pair, e in sorted(edges.items()):
        # Exact shared text is a fact; task specificity is not inferable from length alone.
        judgment='shared_prompt_and_nontrivial_answer' if e['qa_shared'] else ('generic_response_only' if e['all_generic'] else 'shared_turn_semantics_uncertain')
        result.append({'groups':list(pair),'judgment':judgment,'evidence':sorted(e['evidence'],key=lambda x:(x['kind'],x['role'],x['text_sha256'],x['source_lines']))})
    return result


def plan(states, edge_records):
    names=sorted(states); index={g:i for i,g in enumerate(names)}; uf=base.UnionFind(len(names))
    # Preserve existing held and quarantine memberships; block linked candidates.
    for edge in edge_records:
        a,b=edge['groups']; uf.union(index[a],index[b])
    members=defaultdict(list)
    for g in names: members[uf.find(index[g])].append(g)
    assignments={}
    for gs in members.values():
        present={states[g] for g in gs}
        family=sha(('p117-family-v1\0'+'\n'.join(gs)).encode())
        for g in gs:
            target = 'quarantine' if states[g]=='candidate' and ('quarantine' in present or 'development' in present) else states[g]
            assignments[g]={'group_sha256':g,'family_sha256':family,'original':states[g],'proposed':target}
    return [assignments[g] for g in names]


def main():
    if OUT.exists(): raise FileExistsError('output exists')
    prior_bytes=AGG.read_bytes(); prior=json.loads(prior_bytes)
    index_bytes=base.INDEX.read_bytes(); index=json.loads(index_bytes)
    manifest_bytes=base.MANIFEST.read_bytes(); manifest=json.loads(manifest_bytes)
    if sha(index_bytes)!=prior['inputs']['p104_index_sha256'] or sha(manifest_bytes)!=prior['inputs']['manifest_sha256'] or sha(Path(base.__file__).read_bytes())!=prior['inputs']['audit_script_sha256'] or sha((ROOT/'scripts/p104_screen.py').read_bytes())!=index['script_sha256'] or manifest['files']['dialogue-train.jsonl']!=base.EXPECTED or index['source_sha256']!=base.EXPECTED:
        raise ValueError('input fingerprint mismatch')
    meta=[]; h=hashlib.sha256()
    with base.SOURCE.open('rb') as f:
        for line in f:
            h.update(line); r=json.loads(line); m=r.get('messages'); first=m[0].get('content') if isinstance(m,list) and m and isinstance(m[0],dict) else None
            meta.append((r.get('group'),base.normalize(first) if isinstance(first,str) else '',isinstance(m,list) and len(m)>=4))
    if h.hexdigest()!=base.EXPECTED or len(meta)!=index['source_rows']: raise ValueError('source mismatch')
    ranked,total=base.reconstruct(meta)
    if total!=index['source_groups_after_exact_prompt_union']: raise ValueError('group mismatch')
    selected={i:key for _,key,members in ranked for i in members}; grouped=defaultdict(list); h=hashlib.sha256()
    with base.SOURCE.open('rb') as f:
        for i,line in enumerate(f):
            h.update(line)
            if i in selected: grouped[selected[i]].append((i,json.loads(line)))
    if h.hexdigest()!=base.EXPECTED: raise ValueError('source changed')
    rows=[]; states={}; indexed=[]
    for _,key,members in ranked:
        rs=grouped[key]
        if sorted(i for i,_ in rs)!=sorted(members): raise ValueError('member mismatch')
        state='quarantine' if any(assess(r) for _,r in rs) else ('development' if int(sha((base.SEED+'\0split\0'+key).encode())[:8],16)%5==0 else 'candidate')
        states[key]=state
        for i,r in rs:
            rows.append((key,state,i+1,r.get('messages')))
            if state!='quarantine' and isinstance(r.get('messages'),list) and len(r['messages'])>=4: indexed.append({'line':i+1,'group_sha256':key,'partition':state})
    if sorted(indexed,key=lambda x:x['line'])!=sorted(index['rows'],key=lambda x:x['line']): raise ValueError('P104 index mismatch')
    # Validate complete P117 edge universe, including its near-duplicate method, before relying on exact edges.
    fresh=base.audit([(g,p,m) for g,p,_,m in rows])
    if json.loads(json.dumps(fresh)) != prior['result']: raise ValueError('P117 results mismatch')
    if fresh['edge_counts']['turn_near_user'] or fresh['edge_counts']['turn_near_assistant']: raise ValueError('near links require explicit review implementation')
    edges=links(rows)
    if len(edges)!=fresh['counts']['distinct_cross_group_edges']: raise ValueError('edge mismatch')
    proposal=plan(states,edges)
    summary={'input_sha256':{'training':base.EXPECTED,'p104_index':sha(index_bytes),'p117_aggregate':sha(prior_bytes),'manifest':sha(manifest_bytes),'review_script':sha(Path(__file__).read_bytes()),'prior_script':sha(Path(base.__file__).read_bytes())},'edge_count':len(edges),'judgments':dict(Counter(e['judgment'] for e in edges)),'state_pairs':dict(Counter('/'.join(sorted(states[g] for g in e['groups'])) for e in edges)), 'proposed_changes':dict(Counter(x['original']+'->'+x['proposed'] for x in proposal if x['original']!=x['proposed'])), 'family_count':len({x['family_sha256'] for x in proposal})}
    OUT.mkdir(parents=True,exist_ok=False)
    for name,obj in [('summary.json',summary),('links.json',edges),('proposed-families.json',proposal)]:
        with (OUT/name).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__': main()
