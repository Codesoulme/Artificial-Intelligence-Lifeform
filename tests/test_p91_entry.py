import tempfile
import unittest
import json
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_p91_paired as entry
from scripts.run_p91_paired import output_path, run_pair, verify_files
from spora.document_stream import DocumentStream
from spora.paired_experiment import RecordingRandom, sample_update
from spora.source_review import file_sha256
from tests.test_paired_experiment import PairedExperimentTests


class EntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_window_coordinates_match_actual_targets(self):
        raw = np.array([1,10,11,12,13,14,2,1,20,21,2])
        docs = DocumentStream(raw, 1, 2, 0)
        batches, spans, count = sample_update(docs, RecordingRandom(7), 4, 3, 20)
        rows = [(xrow, yrow) for x,y in batches for xrow,yrow in zip(x,y)]
        for (_, y), (doc, start, end) in zip(rows, spans):
            self.assertGreaterEqual(start, 1)
            base = int(docs.starts[doc])
            self.assertEqual(y[y != -100].tolist(), raw[base+start:base+end].tolist())
        self.assertEqual(count, sum(end-start for _,start,end in spans))

    def test_os_lock_rejects_another_process_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            code = ('from pathlib import Path; from scripts.run_p91_paired import directory_lock; '
                    'p=Path(__import__("sys").argv[1]);\n'
                    'with directory_lock(p): print("acquired")')
            with entry.directory_lock(directory):
                child=subprocess.run([sys.executable,'-c',code,directory],capture_output=True,text=True,timeout=30)
                self.assertNotEqual(child.returncode,0)
                self.assertIn('locked',child.stderr)
            child=subprocess.run([sys.executable,'-c',code,directory],capture_output=True,text=True,timeout=30)
            self.assertEqual(child.returncode,0,child.stderr)

    def test_verified_summary_rejects_stale_and_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); checkpoint=path/'checkpoint.pt'
            trainer=PairedExperimentTests().make(); trainer.step(); trainer.save(checkpoint)
            summary={'format':'spora-p91-summary-v1','status':'partial',
                     'checkpoint_sha256':file_sha256(checkpoint), 'steps':trainer.steps,
                     'targets':trainer.targets, 'seed':trainer.spec['seed'],
                     'coverage':entry.coverage(trainer.spans), 'batches':trainer.trace,
                     'updates':trainer.logs, 'evaluations':trainer.evaluations}
            entry.atomic_json(path/'summary.json', summary)
            self.assertEqual(entry.read_verified_summary(path)['status'],'partial')
            summary['targets'] += 1; entry.atomic_json(path/'summary.json',summary)
            with self.assertRaisesRegex(ValueError,'content differs'): entry.read_verified_summary(path)
            checkpoint.write_bytes(b'two')
            with self.assertRaisesRegex(ValueError,'does not match'): entry.read_verified_summary(path)
            entry.atomic_json(path/'summary.json', {'format':'spora-p91-summary-v1','status':'unavailable'})
            with self.assertRaisesRegex(ValueError,'unavailable'): entry.read_verified_summary(path)

    def test_interruption_during_second_arm_discards_partial_step(self):
        maker = PairedExperimentTests()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            reference = maker.make(); run_pair(reference, reference.documents, Path(directory)/'reference.pt', 2, lambda:None, eval_steps={0,2})
            interrupted = maker.make(); run_pair(interrupted, interrupted.documents, path, 1, lambda:None, eval_steps={0,2})
            digest = file_sha256(path)
            with patch.object(interrupted.optimizers[1], 'step', side_effect=RuntimeError('interrupt')):
                with self.assertRaisesRegex(RuntimeError, 'interrupt'):
                    run_pair(interrupted, interrupted.documents, path, 2, lambda:None, eval_steps={0,2})
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                interrupted.save(path)
            self.assertEqual(digest, file_sha256(path))
            restored = maker.make(); restored.restore(path)
            run_pair(restored, restored.documents, path, 2, lambda:None, eval_steps={0,2})
            self.assertEqual(reference.trace, restored.trace)
            self.assertEqual([e['step'] for e in restored.evaluations], [0,2])
            for a,b in zip(reference.models, restored.models):
                for name, value in a.state_dict().items():
                    self.assertTrue(torch.equal(value, b.state_dict()[name]), name)

    def test_ten_step_commits_replay_uncommitted_updates(self):
        maker = PairedExperimentTests()
        def make():
            trainer = maker.make(); trainer.spec['schedule_steps'] = 12
            return trainer
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            reference=make()
            commits=[]; full=[]
            run_pair(reference, reference.documents, Path(directory)/'ref.pt', 12, lambda:None,
                     eval_steps={0,12}, interval=10, full_check=lambda:full.append(reference.steps),
                     on_commit=lambda step, seconds, size:commits.append(step))
            self.assertEqual(commits, [0,10,12])
            self.assertEqual(full, [12])
            interrupted=make(); original=interrupted.optimizers[1].step
            def fail_after_ten(*args, **kwargs):
                if interrupted.steps == 11: raise RuntimeError('power loss')
                return original(*args, **kwargs)
            with patch.object(interrupted.optimizers[1], 'step', side_effect=fail_after_ten):
                with self.assertRaisesRegex(RuntimeError, 'power loss'):
                    run_pair(interrupted, interrupted.documents, path, 12, lambda:None,
                             eval_steps={0,12}, interval=10)
            restored=make(); restored.restore(path)
            self.assertEqual(restored.steps, 10)
            self.assertEqual([r['step'] for r in restored.evaluations], [0])
            run_pair(restored, restored.documents, path, 12, lambda:None,
                     eval_steps={0,12}, interval=10)
            self.assertEqual(restored.trace, reference.trace)
            for a,b in zip(reference.models,restored.models):
                for name,value in a.state_dict().items():
                    self.assertTrue(torch.equal(value,b.state_dict()[name]),name)

    def test_output_and_protected_file_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'runs').mkdir()
            out = output_path(root, 'runs/new', False)
            with self.assertRaises(FileExistsError): output_path(root, 'runs/new', False)
            with self.assertRaises(ValueError): output_path(root, 'runs/new/nested', False)
            p = root/'protected'; p.write_bytes(b'original'); files={str(p):file_sha256(p)}
            verify_files(files); p.write_bytes(b'changed')
            with self.assertRaises(ValueError): verify_files(files)
            self.assertTrue(out.is_dir())

    def test_entry_runs_and_resumes_on_synthetic_files(self):
        maker = PairedExperimentTests()
        class Tokenizer:
            @staticmethod
            def from_file(path): return Tokenizer()
            def token_to_id(self, value): return {'<|bos|>':1, '<|eos|>':2, '<|pad|>':0}[value]
            def get_vocab_size(self): return 64
        def load_individual(path, device):
            trainer = maker.make()
            return trainer.organism, trainer.state, {}
        def query(*args):
            return torch.nn.functional.one_hot(torch.tensor([7,5,9,3]), num_classes=16).float()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('runs','checkpoints','spora','data'): (root/name).mkdir()
            parent=root/'checkpoints/parent.pt'; parent.write_bytes(b'synthetic parent')
            for name in ('pretrain-train.bin','story-development.bin'):
                np.array([1,3,4,2,1,5,6,2], dtype='<u2').tofile(root/'data'/name)
            (root/'data/tokenizer.json').write_text('{}', encoding='utf8')
            for name in ('manifest.json', 'lineage-audit.json'):
                (root/'data'/name).write_text('{}', encoding='utf8')
            for name in ('p28.json', 'p54.json'):
                (root/name).write_text('{}', encoding='utf8')
            plan={'seeds':[7,8], 'schedule_steps':4, 'device':'cpu', 'parent':'checkpoints/parent.pt',
                  'parent_sha256':file_sha256(parent), 'data':'data', 'data_seeds':{'7':8,'8':9},
                  'model':{'vocabulary':64,'width':32,'depth':2,'heads':4,'hidden':64,'context':16,
                           'memory_layers':[0,1]}, 'microbatch':2,'length':8,'target_budget':12,
                  'warmup':1,'learning_rate':0.0001,'evaluation_steps':[0,2,4]}
            (root/'plan.json').write_text(json.dumps(plan), encoding='utf8')
            (root/'review.json').write_text('{}', encoding='utf8')
            args=['entry','--plan','plan.json','--review','review.json','--output','runs/synthetic',
                  '--seed','7','--until','1']
            with patch.object(entry,'ROOT',root), patch.object(entry,'Tokenizer',Tokenizer), \
                 patch.object(entry,'load',side_effect=load_individual), patch.object(entry,'query',side_effect=query), \
                 patch.object(entry.source_gate,'P28','p28.json'), patch.object(entry.source_gate,'P54','p54.json'), patch.object(entry.source_gate,'DATA','data'), \
                 patch.object(entry,'verify_p91_inputs',return_value={'synthetic':True}):
                with patch.object(sys,'argv',args): entry.main()
                with patch.object(sys,'argv',args[:-1]+['2','--resume']): entry.main()
                out=root/'runs/synthetic'
                self.assertEqual(entry.read_verified_summary(out)['steps'],2)
                atomic_before=entry.atomic_json
                def fail_invalidation(path,value):
                    if path.name == 'summary.json' and value.get('status') == 'unavailable':
                        raise OSError('invalidation failure')
                    return atomic_before(path,value)
                with patch.object(entry,'atomic_json',side_effect=fail_invalidation), \
                     patch.object(sys,'argv',args[:-1]+['3','--resume']):
                    with self.assertRaisesRegex(OSError,'invalidation failure'): entry.main()
                self.assertEqual(entry.read_verified_summary(out)['steps'],2)
                with patch.object(entry.PairedTrainer,'step',side_effect=RuntimeError('failure after invalidation')), \
                     patch.object(sys,'argv',args[:-1]+['3','--resume']):
                    with self.assertRaisesRegex(RuntimeError,'failure after invalidation'): entry.main()
                with self.assertRaisesRegex(ValueError,'unavailable'): entry.read_verified_summary(out)
                with patch.object(sys,'argv',args[:-1]+['3','--resume']): entry.main()
                self.assertEqual(entry.read_verified_summary(out)['steps'],3)
                atomic_json=entry.atomic_json
                def fail_publication(path, value):
                    if path.name == 'summary.json' and value.get('status') == 'complete':
                        raise OSError('publication failure')
                    return atomic_json(path,value)
                with patch.object(entry,'atomic_json',side_effect=fail_publication), \
                     patch.object(sys,'argv',args[:-1]+['4','--resume']):
                    with self.assertRaisesRegex(OSError,'publication failure'): entry.main()
                with self.assertRaisesRegex(ValueError,'unavailable'): entry.read_verified_summary(out)
                with patch.object(sys,'argv',args[:-1]+['4','--resume']): entry.main()
                self.assertEqual(entry.read_verified_summary(out)['steps'],4)
                metadata_path=out/'metadata.json'; metadata_bytes=metadata_path.read_bytes()
                stale=json.loads(metadata_bytes); stale['code']['old-source.py']='0'*64
                metadata_path.write_text(json.dumps(stale), encoding='utf8')
                with patch.object(sys,'argv',args[:-1]+['4','--resume']):
                    with self.assertRaisesRegex(ValueError,'implementation'):
                        entry.main()
                metadata_path.write_bytes(metadata_bytes)
                original_step=entry.PairedTrainer.step
                def change_protected(trainer):
                    result=original_step(trainer)
                    parent.write_bytes(b'changed parent')
                    return result
                changed_args=['entry','--plan','plan.json','--review','review.json','--output','runs/changed',
                              '--seed','7','--until','3']
                with patch.object(entry.PairedTrainer,'step',change_protected), patch.object(sys,'argv',changed_args):
                    with self.assertRaisesRegex(ValueError,'metadata changed'):
                        entry.main()
            summary=json.loads((root/'runs/synthetic/summary.json').read_text(encoding='utf8'))
            self.assertEqual(summary['steps'],4)
            self.assertEqual([v['step'] for v in summary['evaluations']],[0,2,4])
            self.assertEqual(summary['process_segment_memory']['start_step'],4)
            self.assertIsNone(summary['process_segment_memory']['cuda_peak_allocated_bytes'])
            self.assertEqual(summary['coverage']['presentations'],summary['targets'])
            io=[json.loads(line) for line in (root/'runs/synthetic/io-cost.jsonl').read_text(encoding='utf8').splitlines()]
            commits=[e for e in io if e['event']=='checkpoint']
            self.assertEqual([e['step'] for e in commits], [0,1,2,3,4])
            self.assertTrue(all(e['logical_bytes_written'] > 0 and e['seconds'] >= 0 for e in commits))
            self.assertTrue(any(e['event']=='full_check' and e['hashes']['bytes'] > 0 for e in io))
            events=[json.loads(line) for line in (root/'runs/changed/io-cost.jsonl').read_text(encoding='utf8').splitlines()]
            self.assertTrue(any(e['event']=='full_check' and e['reason']=='metadata_change' and not e['passed'] for e in events))


if __name__ == '__main__':
    unittest.main()
