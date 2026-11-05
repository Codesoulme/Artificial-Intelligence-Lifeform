"""Collect actual pre-Chinese evidence; refuse missing or failed artifacts."""
import hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_prechinese_runtime import protected_all


def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))


def main():
    sources={'concepts_skills':'runs/p14-p15-audit-v2/audit.json',
        'head_ablation':'runs/p14-head-ablation.json','integration':'runs/p16-runtime-audit-v2/audit.json',
        'candidate':'runs/p16-candidate.json','language_readiness':'runs/p16-language-ready-v2/readiness.json',
        'data_readiness':'runs/chinese-pilot-v1/readiness.json'}
    report={key:read(name) for key,name in sources.items()}
    if not all(part['passed'] for part in report.values()):raise AssertionError('A required gate failed')
    originals=protected_all()
    for name,info in report['candidate']['files'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=info['sha256']:raise AssertionError('Candidate changed')
    tests=(ROOT/'runs/p16-regression-delivery.log').read_text()
    targeted=(ROOT/'runs/p16-targeted-delivery.log').read_text()
    if not tests.rstrip().endswith('OK') or not targeted.rstrip().endswith('OK'):raise AssertionError('Tests failed')
    report['regression']={'full_tests':int(re.search(r'Ran (\d+) tests',tests).group(1)),
        'subsequent_targeted_tests':int(re.search(r'Ran (\d+) tests',targeted).group(1)),
        'reason_for_targeted_rerun':'Added full frozen-tensor digest; preserve configured semantic confidence across proposals',
        'passed':True}
    report.update(date='2026-11-05',parent_commit='87ee56c1761db8808dc55e16bdfab86c4b18344d',
        original_files_unchanged=len(originals),Chinese_training_started=False,
        failures={'first_concept_audit':read('runs/p14-p15-audit/audit.json'),
            'concept_audit_revision':read('configs/p14-audit-revision.json'),
            'runtime_fixture_revision':read('configs/p16-audit-revision.json')},
        pilot=read('configs/chinese-pilot-plan.json'),
        evidence_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources.values()},
        ready_for='Small frozen-organism Chinese engineering pilot only',passed=True)
    path=ROOT/'reports/P14-P16-results.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    skill=[r['controls']['learned']['success_rate'] for r in report['concepts_skills']['skills']]
    print(json.dumps({'passed':True,'tests':report['regression'],'skill_success_range':[min(skill),max(skill)],
        'protected_files':len(originals),'report':str(path)}))


if __name__=='__main__':main()
