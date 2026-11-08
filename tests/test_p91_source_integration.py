import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spora import paired_source_gate as gate
from spora.source_review import file_sha256


class SourceIntegrationTests(unittest.TestCase):
    def test_full_binding_and_modified_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); data=root/'stories'; data.mkdir()
            def save(path, obj):
                path.write_text(json.dumps(obj), encoding='utf8')
            for name in ('pretrain-train.bin','story-development.bin','tokenizer.json'):
                (data/name).write_bytes(name.encode())
            files={name:file_sha256(data/name) for name in ('pretrain-train.bin','story-development.bin','tokenizer.json')}
            save(data/'manifest.json', {'files':files}); manifest=file_sha256(data/'manifest.json')
            p28=root/'p28.json'; save(p28, {'approved_for_pilot':True, 'data_manifest_sha256':manifest}); p28hash=file_sha256(p28)
            p54=root/'p54.json'; save(p54, {'approved_for_pilot':True, 'max_schedule_steps':100,
                'story_source_manifest_sha256':manifest, 'story_original_review_sha256':p28hash,
                'training_source_sha256':{'story':files['pretrain-train.bin']}}); p54hash=file_sha256(p54)
            audit=data/'lineage-audit.json'; save(audit, {'passed':True,'new_train_does_not_overlap_new_held_under_declared_rules':True,
                'manifest_sha256':{'stories':manifest},'data_artifact_sha256':files}); audithash=file_sha256(audit)
            plan=root/'plan.json'; save(plan, {'seeds':[1,2], 'schedule_steps':100, 'data':'stories','target_shares':{'story':1.0},'focus_every':0})
            approval=root/'approval.json'; save(approval, {'approved_for_pilot':True, 'plan_sha256':file_sha256(plan),
                'architectures':['memory_bridge','standard_causal'],'initialization':'paired_random','seeds':[1,2]})
            with patch.multiple(gate, DATA='stories', P28='p28.json', P28_SHA=p28hash, P54='p54.json', P54_SHA=p54hash, MANIFEST_SHA=manifest, AUDIT_SHA=audithash):
                result=gate.verify_p91_inputs(root,plan,approval)
                self.assertEqual(result['source_reviews']['story']['artifact_sha256'],files['pretrain-train.bin'])
                (data/'pretrain-train.bin').write_bytes(b'changed')
                with self.assertRaises(ValueError): gate.verify_p91_inputs(root,plan,approval)


if __name__ == '__main__': unittest.main()
