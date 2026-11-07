import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from spora.chinese_query import ChineseQuery, ChineseQueryConfig, answer_memory, attach_query, consistent_address
from spora.durable import TensorStore
from spora.growing import new_state, save, load, grow
from spora.state_codec import flatten
from spora.temporal import TemporalConfig, TemporalSPORA
from scripts.chinese_query_task import CHARACTERS, examples, render, HELD_KEYS


class ChineseQueryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(26)
        self.module=ChineseQuery(ChineseQueryConfig(CHARACTERS)).eval()
        self.core=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=64,
            expert_rank=4,memory_layers=(0,1),history_writer='rls',current_address='orthogonal'))
        self.state=new_state(self.core)

    def test_query_labels_are_training_only_and_compositional_splits_disjoint(self):
        training=examples(1,256)
        held=examples(2,256,'dev',True,True)
        self.assertTrue(all(r['key'] not in HELD_KEYS for r in training if r['kind']!=3))
        self.assertTrue(all(r['key'] in HELD_KEYS for r in held if r['kind']!=3))
        self.assertTrue(all(r['event'] is None or r['event']%17!=7 for r in training))
        self.assertTrue(all(r['event'] is None or r['event']%17==7 for r in held))
        for row in training:
            if row['kind']==3:continue
            for digit,position in zip(row['digits'],row['positions']):
                self.assertEqual(digit,0 if position==0 else int(row['text'][position-1]))
        tokens=self.module.encode([r['text'] for r in training[:8]])
        out=self.module(tokens)
        targets=torch.tensor([r['kind'] for r in training[:8]])
        F.cross_entropy(out['intent'],targets).backward()
        self.assertTrue(torch.isfinite(self.module.embedding.weight.grad).all())

    def test_padding_does_not_change_question_and_invalid_input_rejects(self):
        short='对象12现在的值是多少？'
        long='请问，对象12在事件1023的来源是什么？请告诉我。'
        with torch.no_grad():
            alone=self.module(self.module.encode([short]))
            batch=self.module(self.module.encode([short,long]))
            self.assertTrue(torch.allclose(alone['intent'][0],batch['intent'][0],atol=1e-6))
            self.assertTrue(torch.allclose(alone['digits'][0],batch['digits'][0],atol=1e-6))
        for text in ('', '🦋', '对'*100):
            with self.assertRaises(ValueError):self.module.encode([text])

    def test_pointer_guard_rejects_duplicate_or_partial_numbers_without_repair(self):
        row=render(1,44,7,2,(0,3))
        self.assertTrue(consistent_address(row['text'],row['positions'],row['digits'],1))
        points=list(row['positions']);digits=list(row['digits'])
        points[-2]=points[-1];digits[-2]=7
        self.assertFalse(consistent_address(row['text'],points,digits,1))
        self.assertFalse(consistent_address('对象123现在的值是多少？',[3,4,0,0,0,0],[1,2,0,0,0,0],0))
        self.assertFalse(consistent_address('对象12现在的值是多少？',[3,4,0,0,0,0],[1,3,0,0,0,0],0))

    def test_low_confidence_never_dispatches_to_memory(self):
        attach_query(self.core,self.module)
        with torch.no_grad():
            self.module.intent.weight.zero_();self.module.intent.bias.zero_()
        with patch('spora.growing.query',side_effect=AssertionError('No read on uncertain intent')):
            result=answer_memory(self.core,self.state,'对象12现在的值是多少？')
        self.assertEqual(result['status'],'rejected')

    def test_future_and_unrecorded_source_are_unknown_without_invention(self):
        attach_query(self.core,self.module)
        parsed=dict(kind='history',entity=12,event=500,confidence=.99,accepted=True)
        with patch.object(self.module,'parse',return_value=[parsed]), patch('spora.growing.query',side_effect=AssertionError('Future read')):
            result=answer_memory(self.core,self.state,'对象12在事件500的值是多少？')
        self.assertEqual(result['status'],'unknown')
        self.state.cursor=1
        parsed.update(kind='source',event=0)
        with patch.object(self.module,'parse',return_value=[parsed]):
            result=answer_memory(self.core,self.state,'对象12在事件0的来源是什么？')
        self.assertEqual(result['status'],'unknown')
        self.assertIsNone(result['source'])

    def test_save_restore_transaction_and_growth_preserve_question_weights(self):
        before_schema,before=flatten(self.core,self.state)
        before={k:v.detach().clone() for k,v in before.items()}
        attach_query(self.core,self.module)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'individual.pt';save(path,self.core,self.state)
            restored,state,_=load(path)
            _,actual=flatten(restored,state)
            self.assertTrue(all(torch.equal(v,actual[k]) for k,v in before.items()))
            expected=self.module(self.module.encode(['对象12现在的值是多少？']))['intent']
            got=restored.chinese_query(restored.chinese_query.encode(['对象12现在的值是多少？']))['intent']
            self.assertTrue(torch.equal(expected,got))
            with TensorStore.create(Path(folder)/'individual.sqlite',restored,state) as store:
                expanded=grow(restored,state)
                def fail(stage):
                    if stage=='after_manifest':raise RuntimeError('Injected crash')
                with self.assertRaises(RuntimeError):store.commit(restored,expanded,store.revision,hook=fail)
                old,old_state,_,_=store.read()
                self.assertEqual(len(old_state.pages),1)
                store.commit(restored,expanded,store.revision)
                new,new_state,_,_=store.read()
                self.assertEqual(len(new_state.pages),2)
                self.assertTrue(all(torch.equal(v,new.chinese_query.state_dict()[k]) for k,v in self.module.state_dict().items()))

    def test_development_gate_is_bound_to_weights_and_source(self):
        from scripts.validate_chinese_memory import verify_development_binding
        from scripts.train_chinese_query import fingerprint
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'module.bin';path.write_bytes(b'original')
            source=Path(__file__).resolve()
            report=dict(passed=True,development={'group':{'passed':True}},
                checkpoint_sha256=fingerprint(path),source_sha256={str(source):fingerprint(source)})
            verify_development_binding(report,path)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'weights'):verify_development_binding(report,path)
            path.write_bytes(b'original');report['source_sha256'][str(source)]='incorrect'
            with self.assertRaisesRegex(ValueError,'source'):verify_development_binding(report,path)


if __name__=='__main__':unittest.main()
