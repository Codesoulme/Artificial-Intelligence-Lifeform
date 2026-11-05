import copy,tempfile,unittest
from pathlib import Path
import torch
from spora.curriculum import CoveringSampler,CurriculumTrainer
from spora.language_training import ByteTrainer,interface_state,atomic_save,frozen_digest
from spora.language_bridge import LanguageBridge
from spora.temporal import TemporalSPORA,TemporalConfig
from spora.growing import new_state


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(123)
        self.organism=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=64,
            expert_rank=4,memory_layers=(0,1),history_writer='rls',current_address='orthogonal'))
        self.state=new_state(self.organism)
        self.examples=torch.randint(256,(10,12))
        self.groups={'focus':list(range(4)),'basic':list(range(4,10))}
        self.schedule=['focus','basic','basic','focus','basic']
        self.provenance={'source_checkpoint_sha256':'source','batch':2,'data':'random-byte-fixture'}

    def trainer(self):
        return CurriculumTrainer(LanguageBridge(copy.deepcopy(self.organism)),copy.deepcopy(self.state),
            'parent',self.provenance,55,CoveringSampler(self.groups,self.schedule,55))

    def source(self):
        return {'format':'spora-byte-training-v2','organism_sha256':'parent',
            'state':interface_state(LanguageBridge(self.organism)),'steps':100,'trained_tokens':5000}

    def test_every_group_covered_before_repetition_and_ratio_exact(self):
        sampler=CoveringSampler(self.groups,self.schedule,55)
        draws=torch.cat([sampler.take(4) for _ in range(5)]).tolist()
        focus=[i for i in draws if i<4];basic=[i for i in draws if i>=4]
        self.assertEqual(len(focus),8);self.assertEqual(len(basic),12)
        self.assertEqual(len(set(focus[:4])),4);self.assertEqual(len(set(basic[:6])),6)
        self.assertTrue(all(count==2 for count in sampler.counts.values()))

    def test_sampler_checkpoint_rejects_different_data_and_invalid_counts(self):
        sampler=CoveringSampler(self.groups,self.schedule,55);sampler.take(7)
        snapshot=sampler.snapshot()
        other=CoveringSampler(self.groups,self.schedule,999);other.restore(snapshot)
        self.assertTrue(torch.equal(sampler.take(16),other.take(16)))
        changed=CoveringSampler({'focus':[0],'basic':[1]},self.schedule,55)
        with self.assertRaises(ValueError):changed.restore(snapshot)
        snapshot['draws']+=1
        with self.assertRaises(ValueError):other.restore(snapshot)

    def test_begin_inherits_weights_resets_optimizer_and_preserves_organism(self):
        trainer=self.trainer();source=self.source();before=frozen_digest(trainer.model.organism,trainer.state)
        trainer.begin(source,'source')
        self.assertEqual(trainer.lineage['prior_steps'],100)
        self.assertEqual(trainer.steps,0);self.assertEqual(len(trainer.optimizer.state),0)
        for k,v in source['state'].items():self.assertTrue(torch.equal(v,interface_state(trainer.model)[k]))
        trainer.step(self.examples,2)
        self.assertEqual(before,frozen_digest(trainer.model.organism,trainer.state))
        with self.assertRaises(ValueError):trainer.begin(source,'source')
        with self.assertRaises(ValueError):self.trainer().begin(source,'other')

    def test_resume_continues_covering_order_and_weights_exactly(self):
        trainer=self.trainer();trainer.begin(self.source(),'source')
        for _ in range(3):trainer.step(self.examples,2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt';atomic_save(trainer.snapshot(),path)
            loss=[trainer.step(self.examples,2) for _ in range(3)]
            other=self.trainer();payload=torch.load(path,weights_only=True);other.restore(payload)
            self.assertEqual(loss,[other.step(self.examples,2) for _ in range(3)])
            for k,v in interface_state(trainer.model).items():self.assertTrue(torch.equal(v,interface_state(other.model)[k]))
            self.assertEqual(trainer.sampler.counts,other.sampler.counts)
            legacy=ByteTrainer(other.model,other.state,'parent',self.provenance,55)
            with self.assertRaises(ValueError):legacy.restore(payload)


if __name__=='__main__':unittest.main()
