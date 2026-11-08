import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
import torch
from spora.temporal import TemporalSPORA, TemporalConfig
from spora.growing import new_state, grow, save, load
from spora.semantic_memory import SemanticMemory, consolidate_semantic_store, propose_semantic
from spora.skills import FeedbackPolicy, learn_feedback
from spora.state_codec import flatten, restore
from spora.durable import TensorStore
from spora.language_bridge import LanguageBridge
from spora.language_training import ByteTrainer, interface_state, load_interface, atomic_save
from spora.text_data import overlap_audit
from spora.train_language import main as train_main
from spora.language_eval import generate, evaluate
from spora.language_bridge import VOCAB, EOS


class PreChineseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.model = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2,
            memory_width=64, expert_rank=4, memory_layers=(0,1), history_writer='rls', current_address='orthogonal'))
        self.state = new_state(self.model)
        self.model.semantic_memory = SemanticMemory()
        self.model.skill_policy = FeedbackPolicy()

    def test_optional_modules_survive_growth_portable_and_transaction(self):
        state = grow(self.model, self.state)
        schema, tensors = flatten(self.model, state)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/'model.pt'
            save(file, self.model, state)
            model, other, _ = load(file)
            self.assertEqual(schema, flatten(model, other)[0])
            with TensorStore.create(file.with_suffix('.sqlite'), model, other) as store:
                restored, rs, _, _ = store.read()
                for key, value in flatten(restored, rs)[1].items():
                    self.assertTrue(torch.equal(value, tensors[key]), key)

    def test_rejected_semantic_publication_leaves_transaction_unchanged(self):
        self.model.semantic_memory=SemanticMemory(confidence=.99)
        with patch('spora.semantic_memory.accepted_facts', return_value=(torch.arange(16),torch.zeros(16,dtype=torch.long))):
            candidate,gate=propose_semantic(self.model,self.state,steps=1)
            self.assertEqual(candidate.confidence,.99)
            self.assertFalse(gate['accepted'])
        with tempfile.TemporaryDirectory() as directory:
            with TensorStore.create(Path(directory)/'model.sqlite', self.model, self.state) as store:
                with patch('spora.semantic_memory.propose_semantic', return_value=(SemanticMemory(), {'accepted':False})):
                    report = consolidate_semantic_store(store, self.model, self.state, store.revision)
                self.assertFalse(report['accepted'])
                self.assertEqual(store.read()[3], 0)

    def test_new_module_without_manifest_is_rejected(self):
        schema, tensors = flatten(self.model, self.state)
        schema.pop('skill_policy')
        with self.assertRaises(RuntimeError):
            restore(schema, tensors)

    def test_interface_cannot_replace_frozen_organism_or_omit_tensors(self):
        bridge = LanguageBridge(self.model)
        payload = {'format':'spora-byte-interface-v1','organism_sha256':'parent','state':interface_state(bridge)}
        load_interface(bridge, payload, 'parent')
        for change in ('insert', 'omit', 'nan'):
            bad = copy.deepcopy(payload)
            if change == 'insert':bad['state']['organism.semantic_memory.head.bias'] = torch.zeros(4)
            elif change == 'omit':bad['state'].pop('read_strength')
            else:bad['state']['read_strength'].fill_(float('nan'))
            with self.assertRaises(ValueError):load_interface(bridge, bad, 'parent')
        with self.assertRaises(ValueError):load_interface(bridge, payload, 'different-parent')

    def test_random_byte_resume_matches_uninterrupted_and_keeps_every_old_tensor(self):
        bridge = LanguageBridge(self.model)
        old = {k:v.detach().clone() for k,v in flatten(self.model,self.state)[1].items()}
        examples = torch.randint(256, (9, 12))
        trainer = ByteTrainer(bridge, self.state, 'parent', {'data':'random-bytes','batch':2}, 19)
        trainer.step(examples, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'step1.pt'
            atomic_save(trainer.snapshot(), path)
            losses = [trainer.step(examples, 2) for _ in range(2)]
            other = ByteTrainer(LanguageBridge(copy.deepcopy(self.model)), copy.deepcopy(self.state),
                'parent', {'data':'random-bytes','batch':2}, 19)
            other.restore(torch.load(path, weights_only=True))
            self.assertEqual(losses, [other.step(examples, 2) for _ in range(2)])
            for key, value in interface_state(bridge).items():
                self.assertTrue(torch.equal(value, interface_state(other.model)[key]), key)
            self.assertEqual(trainer.tokens_seen, other.tokens_seen)
            other.provenance['batch'] = 3
            with self.assertRaises(ValueError):other.restore(torch.load(path, weights_only=True))
        for key, value in flatten(self.model,self.state)[1].items():
            self.assertTrue(torch.equal(value, old[key]), key)

    def test_failed_atomic_checkpoint_preserves_last_good_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            atomic_save({'step':1}, path)
            with patch('spora.language_training.os.replace', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):atomic_save({'step':2}, path)
            self.assertEqual(torch.load(path, weights_only=True)['step'], 1)
            self.assertFalse(path.with_suffix('.pt.tmp').exists())

    def test_near_copies_and_embedded_cross_split_documents_are_rejected(self):
        text = ''.join(chr(0x4e00+i) for i in range(180))
        for other in (text[:90]+'修改'+text[92:], 'extra-prefix'+text+'tail'):
            docs = [{'path':'train','split':'train','text':text}, {'path':'test','split':'test','text':other}]
            with self.assertRaisesRegex(ValueError, 'near-duplicate'):overlap_audit(docs)
        docs = [{'path':'a','split':'train','text':text}, {'path':'b','split':'test','text':'independent material'}]
        self.assertTrue(overlap_audit(docs)['passed'])

    def test_skill_rejects_invalid_feedback_and_outside_domain(self):
        policy = self.model.skill_policy
        zero = torch.zeros(2,2)
        with self.assertRaises(ValueError):policy.act(zero,zero)
        optimizer = torch.optim.Adam(policy.parameters(),lr=.01)
        with self.assertRaises(ValueError):learn_feedback(policy,optimizer,zero,zero,torch.zeros(2,dtype=torch.long),zero,torch.tensor([float('nan'),0.]))
        self.assertEqual(int(policy.feedback_steps),0)
        learn_feedback(policy,optimizer,zero,zero,torch.zeros(2,dtype=torch.long),zero,torch.zeros(2))
        self.assertEqual(int(policy.feedback_steps),1)
        with self.assertRaises(ValueError):policy.act(zero,zero+4)

    def test_real_entrypoint_resume_on_random_ascii_fixtures(self):
        # No natural-language examples: independently sampled printable bytes.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            parent=root/'parent.pt'
            save(parent,self.model,self.state)
            digest=hashlib.sha256(parent.read_bytes()).hexdigest()
            context_layers=2
            payload={'format':'spora-byte-interface-v2','context_layers':context_layers,'organism_sha256':digest,
                'state':interface_state(LanguageBridge(self.model,context_layers=context_layers)),'trained_tokens':0}
            interface=root/'interface.pt';atomic_save(payload,interface)
            rows=[]
            generator=torch.Generator().manual_seed(303)
            for split in ('train','development','test'):
                text=bytes(torch.randint(33,127,(80,),generator=generator).tolist())
                file=root/(split+'.txt');file.write_bytes(text)
                rows.append({'path':file.name,'sha256':hashlib.sha256(text).hexdigest(),
                    'split':split,'source':'random-byte fixture','license':'test fixture'})
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'schema':'spora-text-v1','documents':rows}))
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()
            review=root/'review.json';review.write_text(json.dumps({
                'approved_for_pilot':True,'max_schedule_steps':4,
                'training_source_sha256':{'artifact':manifest_sha256,'manifest':manifest_sha256},
                'parent_checkpoint_sha256':digest,
                'quality_gate':{'status':'passed','independent_review':True,
                    'sample_count':3,'major_count':0,'unsafe_count':0}}))
            common=['train','--individual',str(parent),'--manifest',str(manifest),
                '--review',str(review),'--start-language-training','--device','cpu','--batch','2','--length','12']
            for label,count,flag,file in [('full',4,'--interface',interface),('first',2,'--interface',interface),
                ('second',2,'--resume',root/'first'/'checkpoint.pt')]:
                args=common+['--context-layers',str(context_layers),'--steps',str(count),'--output',str(root/label),flag,str(file)]
                with patch.object(sys,'argv',args),redirect_stdout(io.StringIO()):train_main()
            full=torch.load(root/'full'/'checkpoint.pt',weights_only=True)
            resumed=torch.load(root/'second'/'checkpoint.pt',weights_only=True)
            self.assertEqual(full['steps'],resumed['steps'])
            self.assertEqual(full['trained_tokens'],resumed['trained_tokens'])
            for key,value in full['state'].items():self.assertTrue(torch.equal(value,resumed['state'][key]),key)
            self.assertEqual(hashlib.sha256(parent.read_bytes()).hexdigest(),digest)

    def test_generation_reports_invalid_utf8_and_stops_at_eos_without_writes(self):
        bridge=LanguageBridge(self.model)
        old={k:v.clone() for k,v in flatten(self.model,self.state)[1].items()}
        def logits(tokens,state):
            result=torch.zeros(1,len(tokens[0]),VOCAB)
            result[0,-1,128 if tokens.shape[1]==1 else EOS]=10
            return result
        with patch.object(bridge,'forward',side_effect=logits):
            result=generate(bridge,self.state,'',4)
        self.assertFalse(result['utf8_valid'])
        self.assertTrue(result['ended_with_eos'])
        self.assertEqual(result['generated_byte_values'],[128])
        for k,v in flatten(self.model,self.state)[1].items():self.assertTrue(torch.equal(v,old[k]))
        with self.assertRaises(ValueError):generate(bridge,self.state,'x'*255,128)

    def test_evaluation_weights_real_targets_including_short_final_window(self):
        bridge=LanguageBridge(self.model)
        examples=torch.tensor([[256,10,11,12],[256,13,257,258]])
        result=evaluate(bridge,self.state,examples)
        self.assertEqual(result['target_count'],5)
        self.assertGreater(result['bits_per_target'],0)


if __name__ == '__main__':unittest.main()
