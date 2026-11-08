"""P104 bounded train-only duplicate audit; aggregate output only, no corpus export."""
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/conversation-data-v5/dialogue-train.jsonl'
INDEX = ROOT / 'reports/P104-multiturn-screen.json'
MANIFEST = ROOT / 'runs/conversation-data-v5/manifest.json'
OUTPUT = ROOT / 'runs/p117-dialogue-duplicate-audit/aggregate.json'
EXPECTED = '2629f0c8668ef6736c2a0431f0681c24944299e5d60c8ae8ac78d688b565973e'
SEED = 'p104-20261109-v1'
LIMIT = 500
MIN_USER = 20
MIN_NEAR = 30
MAX_NEAR = 12000
MAX_POSTING = 64
THRESHOLD = 0.8


def sha(data): return hashlib.sha256(data).hexdigest()
def normalize(s): return re.sub(r'\s+', '', unicodedata.normalize('NFKC', s).casefold())


class UnionFind:
    def __init__(self, n): self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        if x != y: self.parent[x] = y


def reconstruct(meta, seed=SEED, limit=LIMIT):
    """Reconstruct P104 original-group OR exact-first-prompt closure and rank."""
    uf = UnionFind(len(meta)); groups = {}; prompts = {}
    for i, (group, prompt, multi) in enumerate(meta):
        if not isinstance(group, str) or not group: raise ValueError('missing original group')
        if group in groups: uf.union(i, groups[group])
        groups[group] = i
        if prompt:
            if prompt in prompts: uf.union(i, prompts[prompt])
            prompts[prompt] = i
    components = defaultdict(list)
    for i in range(len(meta)): components[uf.find(i)].append(i)
    ranked = []
    for members in components.values():
        if not any(meta[i][2] for i in members): continue
        key = sha('\n'.join(sorted({meta[i][0] for i in members})).encode())
        ranked.append((sha((seed+'\0'+key).encode()), key, members))
    ranked.sort()
    if len(ranked) < limit: raise ValueError('not enough groups')
    return ranked[:limit], len(components)


def shingles(s): return {s[i:i+5] for i in range(len(s)-4)}


def audit(rows):
    """Rows are (P104-group, partition, messages); no text or IDs returned."""
    group_names = sorted({r[0] for r in rows}); gi = {g:i for i,g in enumerate(group_names)}
    groups = UnionFind(len(group_names)); counts = Counter(); exact_dialogues = defaultdict(set)
    # A turn can occur in multiple records/families. Use distinct normalized role+text nodes.
    turns = defaultdict(set)
    partitions = defaultdict(set)
    for group, partition, messages in rows:
        g = gi[group]; partitions[g].add(partition)
        if not isinstance(messages,list):
            counts['malformed_rows_excluded_from_text_comparison'] += 1; continue
        if any(not isinstance(m,dict) or m.get('role') not in ('user','assistant') or not isinstance(m.get('content'),str) for m in messages):
            counts['malformed_rows_excluded_from_text_comparison'] += 1; continue
        canonical = []
        for m in messages:
            if not isinstance(m,dict) or m.get('role') not in ('user','assistant') or not isinstance(m.get('content'),str):
                canonical = []; break
            role = m['role']; s = normalize(m['content']); canonical.append([role,s])
            if not s: continue
            counts['turn_occurrences'] += 1
            if role == 'user' and len(s) < MIN_USER:
                counts['short_user_turn_occurrences_excluded'] += 1; continue
            turns[(role,s)].add(g)
        if canonical:
            exact_dialogues[json.dumps(canonical,ensure_ascii=False,separators=(',',':'))].add(g)
        else: counts['malformed_rows_excluded_from_text_comparison'] += 1
    edge_types = {'dialogue_exact':set(), 'turn_exact_user':set(), 'turn_exact_assistant':set(), 'turn_near_user':set(), 'turn_near_assistant':set()}
    def add_edges(group_set, kind):
        items = sorted(group_set)
        for a_i, a in enumerate(items):
            for b in items[a_i+1:]: edge_types[kind].add((a,b)); groups.union(a,b)
    for gs in exact_dialogues.values(): add_edges(gs,'dialogue_exact')
    entries = sorted(turns.items())
    for (role,s), gs in entries: add_edges(gs,'turn_exact_'+role)
    # Capped inverted lists bound memory and comparisons; ignored frequent shingles cause false negatives.
    candidates_by_role = defaultdict(list)
    for (role,s), gs in entries:
        if MIN_NEAR <= len(s) <= MAX_NEAR:
            candidates_by_role[role].append((shingles(s), gs))
        else: counts['unique_turns_outside_near_length'] += 1
    for role, values in candidates_by_role.items():
        inverted = {}
        for i,(grams,_) in enumerate(values):
            for gram in grams:
                if gram not in inverted: inverted[gram] = [i]
                elif inverted[gram] is not None:
                    inverted[gram].append(i)
                    if len(inverted[gram]) > MAX_POSTING: inverted[gram] = None
        usable = {k:v for k,v in inverted.items() if v is not None}
        counts['frequent_shingles_ignored_'+role] = len(inverted)-len(usable)
        compared = 0; matching = 0
        for i,(grams,gs) in enumerate(values):
            other = set()
            for gram in grams:
                for j in usable.get(gram,()):
                    if j < i: other.add(j)
            for j in other:
                othergrams, othergs = values[j]
                if not any(a != b for a in gs for b in othergs): continue
                compared += 1
                if len(grams & othergrams) / len(grams | othergrams) >= THRESHOLD:
                    matching += 1
                    for a in gs:
                        for b in othergs:
                            if a != b:
                                pair = tuple(sorted((a,b))); edge_types['turn_near_'+role].add(pair); groups.union(a,b)
        counts['near_pairs_compared_'+role] = compared
        counts['near_text_pairs_matching_'+role] = matching
    edge_union = set().union(*edge_types.values())
    component_sizes = Counter(Counter(groups.find(i) for i in range(len(group_names))).values())
    counts.update({'rows':len(rows),'p104_groups':len(group_names),'unique_eligible_turns':len(entries),
                   'original_groups_after_duplicate_union':sum(component_sizes.values()),
                   'components_with_multiple_p104_groups':sum(v for k,v in component_sizes.items() if k>1),
                   'p104_groups_in_multi_group_components':sum(k*v for k,v in component_sizes.items() if k>1),
                   'distinct_cross_group_edges':len(edge_union),
                   'candidate_development_edges':sum(1 for a,b in edge_union if ('candidate' in partitions[a] and 'development' in partitions[b]) or ('development' in partitions[a] and 'candidate' in partitions[b])),
                   'quarantine_other_edges':sum(1 for a,b in edge_union if ('quarantine' in partitions[a]) != ('quarantine' in partitions[b]))})
    return {'counts':dict(counts),'edge_counts':{k:len(v) for k,v in edge_types.items()},'component_size_distribution':dict(sorted(component_sizes.items()))}


def main():
    if OUTPUT.parent.exists(): raise FileExistsError('refusing existing output directory')
    index_bytes = INDEX.read_bytes(); index = json.loads(index_bytes)
    manifest_bytes = MANIFEST.read_bytes(); manifest = json.loads(manifest_bytes)
    if index['source'] != 'runs/conversation-data-v5/dialogue-train.jsonl' or index['source_sha256'] != EXPECTED or manifest['files']['dialogue-train.jsonl'] != EXPECTED or index['seed'] != SEED or index['group_limit'] != LIMIT:
        raise ValueError('source or selection mismatch')
    meta=[]; h=hashlib.sha256()
    with SOURCE.open('rb') as f:
        for line in f:
            h.update(line); r=json.loads(line); msgs=r.get('messages')
            first = msgs[0].get('content') if isinstance(msgs,list) and msgs and isinstance(msgs[0],dict) else None
            meta.append((r.get('group'),normalize(first) if isinstance(first,str) else '', isinstance(msgs,list) and len(msgs)>=4))
    if h.hexdigest()!=EXPECTED or len(meta)!=index['source_rows']: raise ValueError('source changed or count mismatch')
    ranked, total = reconstruct(meta)
    if total != index['source_groups_after_exact_prompt_union']: raise ValueError('group count mismatch')
    # P104 assesses every group member, including quarantined and single-turn members.
    if sha((ROOT/'scripts/p104_screen.py').read_bytes()) != index['script_sha256']:
        raise ValueError('P104 script does not match frozen index')
    from p104_screen import assess
    selected = {i:(key, rank) for rank,(_,key,members) in enumerate(ranked) for i in members}
    rows_by_group = defaultdict(list); h=hashlib.sha256()
    with SOURCE.open('rb') as f:
        for i,line in enumerate(f):
            h.update(line)
            if i in selected:
                key,rank=selected[i]; rows_by_group[key].append((i,json.loads(line)))
    if h.hexdigest()!=EXPECTED: raise ValueError('source changed during second pass')
    rows=[]; reconstructed=[]; state_counts=Counter()
    for _,key,members in ranked:
        selected_rows=rows_by_group[key]
        if sorted(i for i,_ in selected_rows)!=sorted(members): raise ValueError('member mismatch')
        flagged=any(assess(row) for _,row in selected_rows)
        part = 'quarantine' if flagged else ('development' if int(sha((SEED+'\0split\0'+key).encode())[:8],16)%5==0 else 'candidate')
        state_counts[part+'_groups'] += 1
        for i,row in selected_rows:
            rows.append((key,part,row.get('messages')))
            if not flagged and isinstance(row.get('messages'),list) and len(row['messages'])>=4:
                reconstructed.append({'line':i+1,'group_sha256':key,'partition':part})
    expected_rows = index['rows']
    if sorted(reconstructed,key=lambda x:x['line']) != sorted(expected_rows,key=lambda x:x['line']): raise ValueError('P104 index mismatch')
    for k,expected in [('candidate_groups','candidate_groups'),('development_groups','development_groups'),('quarantine_groups','groups_quarantined')]:
        if state_counts[k]!=index['counts'][expected]: raise ValueError('P104 partition count mismatch')
    result={'inputs':{'train_path':'runs/conversation-data-v5/dialogue-train.jsonl','train_sha256':EXPECTED,'p104_index_sha256':sha(index_bytes),'manifest_sha256':sha(manifest_bytes),'p104_screen_script_sha256':sha((ROOT/'scripts/p104_screen.py').read_bytes()),'audit_script_sha256':sha(Path(__file__).read_bytes())},
            'parameters':{'selection_seed':SEED,'selected_groups':LIMIT,'normalization':'NFKC, casefold, remove whitespace','exact_dialogue':'role and normalized full turns','turn_exact':'same role; user at least 20 characters','near':'same role, 5-gram set Jaccard >= 0.8, lengths 30..12000, skip shingles in >64 unique turns per role','training_partition_only':True,'reads_final_evaluation':False,'outputs_text_or_identifiers':False},
            'p104_group_states':dict(state_counts), 'result':audit(rows),
            'limitations':['P104 grouping already merges identical first prompts including short prompts; this audit never splits those groups.', 'Frequent-shingle suppression and length limits miss near duplicates. No semantic or cross-source coverage.', 'Group edges are evidence for review, not automatic filtering or proof of equivalence.', 'Intervals and quality, privacy and authorization are not evaluated.']}
    OUTPUT.parent.mkdir(parents=True,exist_ok=False)
    with OUTPUT.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
