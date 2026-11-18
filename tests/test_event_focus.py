"""State/route tests; mocks do not establish neural language understanding."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from spora.durable import TensorStore
from spora.event_focus import (enable_event_focus, bind_event_focus, clear_event_focus,
                               read_focused_event, observe_focused, focused_memory_args)
from spora.growing import new_state, grow, save, load, scalars
from spora.internal_memory import WriteControl, attach, enable_event_key_index, observe_sourced
from spora.state_codec import flatten, restore
from spora.temporal import TemporalSPORA, TemporalConfig


class EventFocusTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7751)
        torch.set_num_threads(2)
        self.model = TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=64,
                     expert_rank=4,memory_layers=(0,1),history_writer='rls',current_address='orthogonal'))
        self.state = enable_event_key_index(self.model,attach(self.model,new_state(self.model),WriteControl()))
        with torch.no_grad():
            for p in self.model.write_control.parameters(): p.zero_()
            self.model.write_control.network[-1].bias[4]=20
        for key in (2,3,2):
            self.state,_=observe_sourced(self.model,self.state,key,1,'direct',1.)
        self.state=enable_event_focus(self.model,self.state)
        self.provenance={'source':'direct','action':'new','valid_at':0}
        self.logits=torch.zeros(1,self.model.config.value_count)
        self.logits[0,1]=10

    def bind(self,state,key,event):
        with patch('spora.event_focus.read_provenance',return_value=self.provenance), \
             patch('spora.event_focus.query',return_value=self.logits):
            return bind_event_focus(self.model,state,key,event)

    def test_optional_index_required_and_empty_focus(self):
        self.assertEqual(read_focused_event(self.model,self.state)['reason'],'no_event_focus')
        with self.assertRaises(ValueError): enable_event_focus(self.model,self.state)
        other=copy.deepcopy(self.state)
        other.internal.event_focus=None
        other.internal.event_keys=None
        with self.assertRaises(ValueError): enable_event_focus(self.model,other)
        self.assertEqual(read_focused_event(self.model,other)['reason'],'event_focus_disabled')

    def test_binding_clear_and_growth_do_not_mutate_prior_state(self):
        first=self.bind(self.state,2,0)
        second=self.bind(first,3,1)
        self.assertEqual(self.state.internal.event_focus.tolist(),[-1,-1])
        self.assertEqual(first.internal.event_focus.tolist(),[2,0])
        self.assertEqual(second.internal.event_focus.tolist(),[3,1])
        self.assertEqual(grow(self.model,first).internal.event_focus.tolist(),[2,0])
        self.assertEqual(clear_event_focus(self.model,first).internal.event_focus.tolist(),[-1,-1])
        old=copy.deepcopy(first);old.internal.event_focus=None
        self.assertEqual(scalars(first)['total']-scalars(old)['total'],2)

    def test_cross_object_future_and_nonfactual_event_rejected(self):
        for key,event in ((3,0),(2,3),(True,0),(2,-1)):
            with self.assertRaises(ValueError): self.bind(self.state,key,event)
        for provenance in ({'source':'simulation','action':'defer','valid_at':0},
                           {'source':'direct','action':'defer','valid_at':0},
                           {'source':None,'reason':'uncertain_provenance'},
                           {'source':'direct','action':'new','valid_at':2}):
            with patch('spora.event_focus.read_provenance',return_value=provenance):
                with self.assertRaises(ValueError): bind_event_focus(self.model,self.state,2,0)

    def test_changed_neural_read_refuses_without_fallback(self):
        state=self.bind(self.state,2,0)
        with patch('spora.event_focus.read_provenance',return_value={'reason':'unknown_event'}):
            self.assertEqual(read_focused_event(self.model,state)['status'],'abstain')
            with self.assertRaises(ValueError): focused_memory_args(self.model,state)
        for kind in ('unknown','nan'):
            bad=self.logits.clone()
            if kind=='unknown': bad[0,self.model.config.unknown]=20
            else: bad[0,0]=float('nan')
            with patch('spora.event_focus.read_provenance',return_value=self.provenance), \
                 patch('spora.event_focus.query',return_value=bad):
                self.assertEqual(read_focused_event(self.model,state)['reason'],'focused_event_unreadable')
        self.assertEqual(state.internal.event_focus.tolist(),[2,0])

    def test_zero_content_cannot_be_rescued_by_confident_classifier(self):
        state=self.bind(self.state,2,0)
        damaged=copy.deepcopy(state)
        with torch.no_grad():
            for site in damaged.pages[0]: site[0].zero_()
        with patch('spora.event_focus.read_provenance',return_value=self.provenance), \
             patch('spora.event_focus.query',return_value=self.logits) as query:
            result=read_focused_event(self.model,damaged)
            self.assertEqual(result['reason'],'empty_or_invalid_event_read')
            query.assert_not_called()

    def test_pointer_roundtrip_and_corruption_rejection(self):
        state=self.bind(self.state,3,1)
        schema,tensors=flatten(self.model,state)
        m,s=restore(schema,tensors)
        self.assertEqual(s.internal.event_focus.tolist(),[3,1])
        for data in (torch.tensor([2,1]),torch.tensor([3,99]),torch.tensor([-1,0]),torch.tensor([3.,1.])):
            changed=dict(tensors);changed['internal/event_focus']=data
            with self.assertRaises(ValueError): restore(schema,changed)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.pt';save(path,self.model,state)
            _,s,_=load(path);self.assertEqual(s.internal.event_focus.tolist(),[3,1])
        legacy=copy.deepcopy(self.state);legacy.internal.event_focus=None
        schema,tensors=flatten(self.model,legacy)
        self.assertNotIn('event_focus',schema['internal'])
        self.assertIsNone(restore(schema,tensors)[1].internal.event_focus)

    def test_transaction_interruption_keeps_previous_focus(self):
        first=self.bind(self.state,2,0);second=self.bind(first,3,1)
        with tempfile.TemporaryDirectory() as temp:
            with TensorStore.create(Path(temp)/'state.sqlite',self.model,first) as store:
                def interrupt(stage):
                    if stage=='after_manifest': raise RuntimeError('interrupted')
                with self.assertRaises(RuntimeError): store.commit(self.model,second,0,hook=interrupt)
                _,recovered,_,revision=store.read()
                self.assertEqual(revision,0)
                self.assertEqual(recovered.internal.event_focus.tolist(),[2,0])
                store.commit(self.model,second,0)
                _,recovered,_,revision=store.read()
                self.assertEqual(revision,1)
                self.assertEqual(recovered.internal.event_focus.tolist(),[3,1])

    def test_deferred_observation_preserves_focus(self):
        first=self.bind(self.state,2,0)
        for source in ('direct','simulation'):
            candidate,decision=observe_focused(self.model,first,3,1,source,1.)
            self.assertFalse(decision['current_write'])
            self.assertEqual(candidate.internal.event_focus.tolist(),[2,0])
            self.assertEqual(candidate.cursor,first.cursor+1)

    def test_controls_address_exact_historical_slot(self):
        state=self.bind(self.state,3,1)
        with patch('spora.event_focus.read_provenance',return_value=self.provenance), \
             patch('spora.event_focus.query',return_value=self.logits):
            controls,selection=focused_memory_args(self.model,state)
        self.assertEqual(selection['key'],3)
        self.assertEqual(controls['memory_slots'].tolist(),[1])
        self.assertFalse(controls['current_only'])
        k=torch.tensor([3]);m=torch.ones_like(k)
        _,bound=self.model.representation(k,m,torch.tensor([1]))
        for site,address in zip(self.model.memories,controls['memory_addresses']):
            self.assertTrue(torch.equal(address,self.model.memory_address(site,bound,k,m)))
