"""Integrated transport tests with a tiny synthetic decoder, not DeepSeek accuracy."""
import unittest
from unittest.mock import patch

import torch

from spora.event_focus import enable_event_focus, bind_event_focus
from spora.integrated_checkpoint import IntegratedSPORARuntime, load_integrated_runtime
from spora.integrated_open_base import IntegratedOpenBase, NativeSPORAMemoryReader
from spora.internal_memory import attach, WriteControl, enable_event_key_index, observe_sourced
from tests import test_integrated_checkpoint as fixtures


class IntegratedEventFocusTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.fixture=fixtures.IntegratedCheckpointTests()
        self.fixture.setUp()
        f=self.fixture
        controller=WriteControl()
        with torch.no_grad():
            for p in controller.parameters(): p.zero_()
            controller.network[-1].bias[4]=20
        state=enable_event_key_index(f.organism,attach(f.organism,f.state,controller))
        state,_=observe_sourced(f.organism,state,2,1,'direct',1.)
        state=enable_event_focus(f.organism,state)
        self.provenance={'source':'direct','action':'new','valid_at':0}
        self.logits=torch.zeros(1,f.organism.config.value_count);self.logits[0,1]=10
        with patch('spora.event_focus.read_provenance',return_value=self.provenance), \
             patch('spora.event_focus.query',return_value=self.logits):
            state=bind_event_focus(f.organism,state,2,0)
        reader=NativeSPORAMemoryReader(3,64,.5)
        reader.initialize_from_legacy(f.reader)
        decoder=IntegratedOpenBase(f.base(),reader,(0,2))
        self.runtime=IntegratedSPORARuntime(decoder,f.organism,state,f.root,f.path)

    def tearDown(self):
        self.fixture.tearDown()

    def test_generate_uses_focus_and_rejects_conflicting_controls(self):
        r=self.runtime
        with patch('spora.event_focus.read_provenance',return_value=self.provenance), \
             patch('spora.event_focus.query',return_value=self.logits), \
             patch.object(r.decoder,'generate',return_value='result') as generate:
            self.assertEqual(r.generate('question',use_event_focus=True),'result')
            args=generate.call_args.kwargs
            self.assertEqual(args['memory_slots'].tolist(),[1])
            self.assertEqual(len(args['memory_addresses']),2)
            self.assertIs(args['memory_state'],r.state)
            for kwargs in ({'memory_key':2},{'memory_slots':torch.tensor([0])},
                           {'current_only':True},{'memory_time':1}):
                with self.assertRaises(ValueError): r.generate('question',use_event_focus=True,**kwargs)
        with patch('spora.event_focus.read_provenance',return_value={'reason':'unknown_event'}), \
             patch.object(r.decoder,'generate') as generate:
            with self.assertRaises(ValueError): r.generate('question',use_event_focus=True)
            generate.assert_not_called()

    def test_integrated_checkpoint_and_failed_commit_preserve_focus(self):
        r=self.runtime;r.save()
        recovered=load_integrated_runtime(self.fixture.path,self.fixture.base(),self.fixture.root)
        self.assertEqual(recovered.state.internal.event_focus.tolist(),[2,0])

        before=r.state
        with patch('spora.integrated_checkpoint.save_integrated',side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError): r.observe_sourced(3,1,'simulation',1.,update_event_focus=True)
        self.assertIs(r.state,before)
        self.assertEqual(r.state.internal.event_focus.tolist(),[2,0])
        result=r.observe_sourced(3,1,'simulation',1.,update_event_focus=True)
        self.assertFalse(result['current_write'])
        recovered=load_integrated_runtime(self.fixture.path,self.fixture.base(),self.fixture.root)
        self.assertEqual(recovered.state.cursor,2)
        self.assertEqual(recovered.state.internal.event_focus.tolist(),[2,0])

    def test_actual_decoder_uses_same_historical_matrix_as_explicit_route(self):
        from spora.event_focus import focused_memory_args
        r=self.runtime
        x=torch.ones(1,2,3)
        with patch('spora.event_focus.read_provenance',return_value=self.provenance), \
             patch('spora.event_focus.query',return_value=self.logits):
            controls,_=focused_memory_args(r.organism,r.state)
            expected=r.decoder.generate(x,memory_state=r.state,**controls)
            actual=r.generate(x,use_event_focus=True)
            self.assertTrue(torch.equal(actual,expected))
            empty=r.decoder.generate(x)
            self.assertFalse(torch.equal(actual,empty))
