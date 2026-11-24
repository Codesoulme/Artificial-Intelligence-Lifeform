import tempfile
import unittest
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from spora.durable import RevisionConflict
from spora.growing import new_state, save
from spora.integrated_checkpoint import _file_sha256, load_integrated, save_integrated
from spora.integrated_durable import create_integrated_durable, load_integrated_durable
from spora.integrated_open_base import NativeSPORAMemoryReader
from spora.internal_memory import WriteControl, attach, enable_event_key_index, read_provenance
from spora.memory_output import MemoryOutputAdapter
from spora.language_training import frozen_digest
from spora.sleep_memory import SealedPage
from spora.structure_transfer import SPORALayerReplacement
from spora.temporal import TemporalConfig, TemporalSPORA
from tests.test_integrated_open_base import Base


class IntegratedDurableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'model.safetensors').write_bytes(b'base weights')
        (self.root/'config.json').write_text('{"hidden_size":3,"num_hidden_layers":3,"num_attention_heads":1,"vocab_size":9}', encoding='utf-8')
        for name in ('tokenizer.json', 'tokenizer_config.json'):
            (self.root/name).write_text('{}', encoding='utf-8')
        self.bootstrap = self.root/'bootstrap.pt'
        self.database = self.root/'individual.sqlite'
        self.write_bootstrap()

    def write_bootstrap(self, *, time_count=64, with_output=False):
        organism = TemporalSPORA(TemporalConfig(width=32, layers=2, heads=2, memory_width=64,
                                  expert_rank=4, memory_layers=(0, 1),
                                  history_writer='rls', current_address='orthogonal', time_count=time_count))
        state = new_state(organism)
        control = WriteControl()
        with torch.no_grad():
            for param in control.parameters():
                param.zero_()
            control.network[-1].bias[0] = 20
        state = attach(organism, state, control)
        reader = NativeSPORAMemoryReader(3, 64, 0.5)
        layer = SPORALayerReplacement(3, 1, max_length=8, layer_idx=1)
        adapter = MemoryOutputAdapter(organism.config.value_count, 9, rank=2) if with_output else None
        save_integrated(self.bootstrap, organism, state, reader, self.root, (0, 2), replacements={1: layer},
                        output_adapter=adapter)

    def tearDown(self):
        self.temp.cleanup()

    def base(self):
        model = Base()
        model.config = SimpleNamespace(hidden_size=3, num_attention_heads=1)
        return model

    def create(self, *, calendar=False):
        seed = None
        if calendar:
            _, organism, state = load_integrated(self.bootstrap, self.base(), self.root)
            state = enable_event_key_index(organism, state)
            seed = self.root / 'calendar-seed.pt'
            save(seed, organism, state, {'decoder_checkpoint_sha256': _file_sha256(self.bootstrap),
                                        'observation_calendar': {'format': 'spora-observation-calendar-v1', 'days': []}})
        return create_integrated_durable(self.database, self.bootstrap, self.base(), self.root,
                                         calendar_individual=seed)

    def open(self):
        return load_integrated_durable(self.database, self.bootstrap, self.base(), self.root)

    def test_structured_confirmation_cannot_overwrite_a_newer_fact(self):
        config = {'memory_width': 64, 'identity_slots': 1, 'fact_slots': 1}
        bounds = {'identity_length': 1, 'fact_length': 1}
        current = torch.ones(64)
        proposed = torch.full((64,), 2.)
        newer = torch.full((64,), 3.)
        with self.create() as runtime:
            first = runtime.write_structured_relation(4, '地点', current, bounds, codec_config=config,
                                                      expected_record_revision=None)
            runtime.write_structured_relation(4, '联系人', current, bounds, expected_record_revision=None)
            second = runtime.write_structured_relation(4, '地点', newer, bounds, replace_existing=True,
                                                       retain_previous=True)
            with self.assertRaisesRegex(ValueError, 'changed after'):
                runtime.write_structured_relation(4, '地点', proposed, bounds, replace_existing=True,
                                                  retain_previous=True, expected_record_revision=first['record_revision'])
            with self.assertRaisesRegex(ValueError, 'changed after'):
                runtime.write_structured_relation(4, '地点', proposed, bounds, replace_existing=True,
                                                  expected_record_revision=None)
            runtime.write_structured_relation(4, '地点', proposed, bounds, replace_existing=True, retain_previous=True,
                                              expected_record_revision=second['record_revision'])
        with self.open() as runtime:
            torch.testing.assert_close(runtime.read_structured_relation(4, '地点')['packet'], proposed)
            torch.testing.assert_close(runtime.read_structured_relation(4, '联系人')['packet'], current)
            old = runtime.read_structured_relation(4, '地点', as_of_revision=second['record_revision'])
            torch.testing.assert_close(old['packet'], newer)

    def test_versioned_relation_preserves_old_values_across_restart(self):
        config = {'memory_width': 64, 'identity_slots': 1, 'fact_slots': 1}
        bounds = {'identity_length': 1, 'fact_length': 1}
        with self.create() as runtime:
            first = runtime.write_structured_relation(4, '地点', torch.ones(64), bounds, codec_config=config)
            runtime.write_structured_relation(5, '地点', torch.full((64,), 99.), bounds)
        with self.open() as runtime:
            second = runtime.write_structured_relation(4, '地点', torch.full((64,), 2.), bounds,
                                                       replace_existing=True, retain_previous=True)
            runtime.write_structured_relation(4, '地点', torch.full((64,), 3.), bounds, replace_existing=True)
            with self.assertRaisesRegex(ValueError, 'Relation-bound'):
                runtime.write_structured_text(first['slot'], torch.zeros(64), bounds)
        with self.open() as runtime:
            for revision, expected in ((first['record_revision'], 1.), (second['record_revision'], 2.), (None, 3.)):
                value = runtime.read_structured_relation(4, '地点', as_of_revision=revision)
                self.assertTrue(torch.equal(value['packet'], torch.full((64,), expected)))
            self.assertIsNone(runtime.read_structured_relation(4, '地点', as_of_revision=0))
            self.assertTrue(torch.equal(runtime.read_structured_relation(5, '地点')['packet'], torch.full((64,), 99.)))

    def test_addressed_generation_uses_the_same_committed_value_and_snapshot(self):
        self.write_bootstrap(with_output=True)
        with self.create() as runtime:
            runtime.observe_sourced(2, 1, 'direct', 1.)
            runtime.observe_sourced(2, 3, 'direct', 1.)
            current_logits = torch.zeros(1, runtime.organism.config.value_count)
            earlier_logits = torch.zeros_like(current_logits)
            current_logits[0, 3], earlier_logits[0, 1] = 1., 1.
            with (patch('spora.integrated_durable.query', side_effect=(current_logits, earlier_logits)) as query,
                  patch.object(runtime.decoder, 'generate', return_value='reply') as generate):
                self.assertEqual(runtime.generate_addressed_memory(key=2), (3, 'reply'))
                current = generate.call_args.kwargs
                self.assertIs(current['memory_state'], runtime.state)
                self.assertEqual(current['memory_slots'].tolist(), [0])
                self.assertEqual(current['memory_output_logits'].argmax(-1).tolist(), [3])
                self.assertEqual(runtime.generate_addressed_memory(key=2, event_time=0), (1, 'reply'))
                earlier = generate.call_args.kwargs
                self.assertIs(earlier['memory_state'], runtime.state)
                self.assertEqual(earlier['memory_slots'].tolist(), [1])
                self.assertEqual(earlier['memory_output_logits'].argmax(-1).tolist(), [1])
                for call, expected in zip(query.call_args_list, ((2, 0, 0), (2, 1, 0)), strict=True):
                    self.assertIs(call.args[1], runtime.state)
                    self.assertEqual(tuple(int(index.item()) for index in call.args[2:]), expected)
                with self.assertRaisesRegex(ValueError, 'owns the memory evidence'):
                    runtime.generate_addressed_memory(key=2, memory_output_logits=torch.ones(1, 9))
                self.assertEqual(generate.call_count, 2)

    def test_name_enrollment_commits_and_resolves_after_restart(self):
        initial_sha = _file_sha256(self.bootstrap)
        with self.create() as runtime:
            self.assertIsNone(runtime.resolve_confirmed_name('小芮')['key'])
            self.assertEqual(runtime.enroll_name(2, '小芮')['revision'], 1)
            self.assertEqual(runtime.resolve_confirmed_name('小芮')['key'], 2)
            self.assertIsNone(runtime.resolve_confirmed_name('小芮璇')['key'])
            with self.assertRaises(ValueError):
                runtime.enroll_name(3, '小芮')
            self.assertEqual(runtime.store.revision, 1)
        with self.open() as runtime:
            self.assertEqual(runtime.resolve_confirmed_name('小芮')['key'], 2)
            self.assertEqual(runtime.verify_detected_name('问小芮', [(1, 3)])['key'], 2)
            self.assertIsNone(runtime.verify_detected_name('问小芮璇', [(1, 4)])['key'])
            self.assertIsNone(runtime.verify_detected_name('问小芮和阿奕', [(1, 3), (4, 6)])['key'])
        self.assertEqual(_file_sha256(self.bootstrap), initial_sha)

    def test_name_enrollment_publishes_only_committed_bindings(self):
        def fail_at(target):
            def hook(stage):
                if stage == target:
                    raise RuntimeError(target)
            return hook

        with self.create() as runtime:
            with self.assertRaisesRegex(RuntimeError, 'before_commit'):
                runtime.enroll_name(2, '小芮', hook=fail_at('before_commit'))
            self.assertEqual(runtime.store.revision, 0)
            self.assertIsNone(runtime.resolve_confirmed_name('小芮')['key'])
            with self.assertRaisesRegex(RuntimeError, 'after_commit'):
                runtime.enroll_name(3, '阿奕', hook=fail_at('after_commit'))
            self.assertEqual(runtime.store.revision, 1)
            self.assertEqual(runtime.resolve_confirmed_name('阿奕')['key'], 3)
        with self.open() as runtime:
            self.assertIsNone(runtime.resolve_confirmed_name('小芮')['key'])
            self.assertEqual(runtime.resolve_confirmed_name('阿奕')['key'], 3)

    @torch.no_grad()
    def test_multiple_query_key_groups_survive_reopen_and_remain_bound(self):
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ModuleNotFoundError as error:
            if error.name != 'transformers':
                raise
            self.skipTest('Qwen2 inference dependencies are optional')
        from spora.integrated_checkpoint import IntegratedSPORARuntime, load_integrated
        from spora.integrated_open_base import IntegratedOpenBase
        from spora.pretrained_attention import ValuePositionAttention
        from spora.research_partial_query_key import PartialQueryKeyPositionAttention

        _, organism, state = load_integrated(self.bootstrap, self.base(), self.root)
        config = Qwen2Config(vocab_size=9, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=32, pad_token_id=0, eos_token_id=2)
        config._attn_implementation = 'sdpa'
        config.to_json_file(self.root / 'config.json')
        base = Qwen2ForCausalLM(config).eval()
        weights = {name: value.clone() for name, value in base.state_dict().items()}

        def fresh_base():
            restored = Qwen2ForCausalLM(config).eval()
            restored.load_state_dict(weights)
            return restored

        decoder = IntegratedOpenBase(base, NativeSPORAMemoryReader(16, 64, .5), (0, 2)).eval()
        decoder.migrate_pretrained_blocks()
        decoder.migrate_position_encoder(16)
        layer = base.model.layers[1]
        value = ValuePositionAttention(layer.self_attn, 0, torch.randn(16, 16),
                                       torch.randn(4, 16), torch.randn(4))
        first = PartialQueryKeyPositionAttention(value, (1,), torch.randn(6, 16), torch.randn(6))
        layer.self_attn = PartialQueryKeyPositionAttention(first, (0,), torch.randn(6, 16), torch.randn(6),
                                                          group=1).eval()
        layout = {'layers': [1], 'window': 4, 'prefix': 1}
        decoder.configure_anchored_attention(layout)
        IntegratedSPORARuntime(decoder, organism, state, self.root).save(self.bootstrap)
        tokens = torch.tensor([[3, 5, 6, 4, 7, 8]])
        expected = decoder(tokens, memory_state=state, use_cache=False).logits
        generated = decoder.generate(tokens, memory_state=state, use_cache=True, max_new_tokens=3,
                                     pad_token_id=0, eos_token_id=None, return_dict_in_generate=True)
        self.assertEqual(generated.past_key_values.layers[1].keys.shape[-2], 3)
        with create_integrated_durable(self.database, self.bootstrap, fresh_base(), self.root) as runtime:
            actual = runtime.decoder(tokens, memory_state=runtime.state, use_cache=False).logits
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(runtime.decoder.anchored_attention_layout, layout)
            restored = runtime.generate(tokens, use_cache=True, max_new_tokens=3, pad_token_id=0,
                                        eos_token_id=None, return_dict_in_generate=True)
            self.assertTrue(torch.equal(restored.sequences, generated.sequences))
            self.assertEqual(restored.past_key_values.layers[1].keys.shape[-2], 3)
            runtime.observe_sourced(4, 2, 'direct', 1.0)
            digest = frozen_digest(runtime.organism, runtime.state)
        with load_integrated_durable(self.database, self.bootstrap, fresh_base(), self.root) as runtime:
            self.assertEqual(frozen_digest(runtime.organism, runtime.state), digest)
            attention = runtime.decoder.base.model.layers[1].self_attn
            self.assertEqual([(group.group, group.pairs) for group in attention.migrated_groups()],
                             [(0, (1,)), (1, (0,))])
            attention.parent._coordinates[0] = 0
            with self.assertRaisesRegex(ValueError, 'Decoder changed'):
                runtime.observe_sourced(4, 3, 'direct', 1.0)

    def test_incremental_write_and_reopen_preserve_structure_and_state(self):
        with self.create() as runtime:
            decision, stats = runtime.observe_sourced(4, 2, 'direct', 1.0)
            self.assertTrue(decision['current_write'])
            self.assertEqual(stats['revision'], 1)
            digest = frozen_digest(runtime.organism, runtime.state)
            self.assertLess(stats['tensor_bytes_written'], self.bootstrap.stat().st_size)
        with self.open() as restored:
            self.assertEqual(restored.state.cursor, 1)
            self.assertEqual(frozen_digest(restored.organism, restored.state), digest)
            self.assertIsInstance(restored.decoder.base.model.layers[1], SPORALayerReplacement)

    @torch.no_grad()
    def test_dialogue_checkpoint_preserves_the_bound_individual_and_memory_logits(self):
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ModuleNotFoundError as error:
            if error.name != 'transformers':
                raise
            self.skipTest('Qwen2 inference dependencies are optional')
        from spora.integrated_checkpoint import IntegratedSPORARuntime
        from spora.integrated_open_base import IntegratedOpenBase
        from spora.pretrained_attention import SPORAGeluGate, SPORAResidualBlock, without_dialogue_residuals

        _, organism, state = load_integrated(self.bootstrap, self.base(), self.root)
        config = Qwen2Config(vocab_size=9, hidden_size=16, intermediate_size=24, num_hidden_layers=4,
                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32)
        config.to_json_file(self.root/'config.json')
        base = Qwen2ForCausalLM(config).eval()
        weights = {name: value.clone() for name, value in base.state_dict().items()}

        def fresh_base():
            model = Qwen2ForCausalLM(config).eval()
            model.load_state_dict(weights)
            return model

        decoder = IntegratedOpenBase(base, NativeSPORAMemoryReader(16, 64, .5), (0, 2)).eval()
        decoder.migrate_pretrained_blocks()
        checkpoint = IntegratedSPORARuntime(decoder, organism, state, self.root)
        checkpoint.save(self.bootstrap)
        residual = SPORAResidualBlock(base.model.layers[1], rank=2, ordinary_only=True,
                                      memory_context=decoder._memory_context).eval()
        residual.up.weight.fill_(.3)
        base.model.layers[1] = residual
        candidate = self.root/'dialogue.pt'
        checkpoint.save(candidate)
        base.model.layers[3].mlp.activation = SPORAGeluGate().eval()
        gated_candidate = self.root/'dialogue-gated.pt'
        checkpoint.save(gated_candidate)
        tokens = torch.tensor([[3, 5, 6]])

        with create_integrated_durable(self.database, self.bootstrap, fresh_base(), self.root) as runtime:
            runtime.observe_sourced(4, 2, 'direct', 1.)
            revision = runtime.store.revision
            digest = frozen_digest(runtime.organism, runtime.state)
            ordinary = runtime.decoder(tokens, use_cache=False).logits
            memory = runtime.decoder(tokens, memory_state=runtime.state, use_cache=False).logits
            routing = runtime.decoder.base(tokens, use_cache=False, output_hidden_states=True).hidden_states[3]
        with self.assertRaisesRegex(ValueError, 'before memory feature extraction ends'):
            load_integrated_durable(self.database, self.bootstrap, fresh_base(), self.root,
                                    dialogue_checkpoint=gated_candidate, late_gate_start=4)
        with load_integrated_durable(self.database, self.bootstrap, fresh_base(), self.root,
                                     dialogue_checkpoint=gated_candidate, late_gate_start=3) as runtime:
            self.assertEqual(frozen_digest(runtime.organism, runtime.state), digest)
            self.assertIsInstance(runtime.decoder.base.model.layers[3].mlp.activation, SPORAGeluGate)
            with without_dialogue_residuals():
                actual = runtime.decoder.base(tokens, use_cache=False, output_hidden_states=True).hidden_states[3]
            torch.testing.assert_close(actual, routing, atol=0, rtol=0)
        with load_integrated_durable(self.database, self.bootstrap, fresh_base(), self.root,
                                     dialogue_checkpoint=candidate) as runtime:
            self.assertEqual(frozen_digest(runtime.organism, runtime.state), digest)
            self.assertEqual(runtime.store.revision, revision)
            self.assertFalse(torch.equal(runtime.decoder(tokens, use_cache=False).logits, ordinary))
            with without_dialogue_residuals():
                torch.testing.assert_close(runtime.decoder(tokens, use_cache=False).logits, ordinary,
                                           atol=0, rtol=0)
            self.assertFalse(torch.equal(runtime.decoder(tokens, use_cache=False).logits, ordinary))
            torch.testing.assert_close(runtime.decoder(tokens, memory_state=runtime.state, use_cache=False).logits,
                                       memory, atol=0, rtol=0)
            runtime.observe_sourced(4, 3, 'direct', 1.)
            digest = frozen_digest(runtime.organism, runtime.state)
            invalid = torch.load(candidate, weights_only=True)
            reader_key = next(key for key in invalid['tensors'] if key.startswith('reader/'))
            invalid['tensors'][reader_key].reshape(-1)[0] += 1.
            torch.save(invalid, candidate)
            with self.assertRaisesRegex(ValueError, 'Dialogue checkpoint changed during runtime'):
                runtime.generate_ordinary(tokens)
        with self.assertRaisesRegex(ValueError, 'Dialogue checkpoint changed existing tensors'):
            load_integrated_durable(self.database, self.bootstrap, fresh_base(), self.root,
                                    dialogue_checkpoint=candidate)
        with load_integrated_durable(self.database, self.bootstrap, fresh_base(), self.root) as runtime:
            self.assertEqual(runtime.store.revision, revision + 1)
            self.assertEqual(frozen_digest(runtime.organism, runtime.state), digest)

    def test_structured_text_records_use_the_open_base_transaction(self):
        config = {'memory_width': 64, 'identity_slots': 1, 'fact_slots': 1}
        packet = torch.arange(64, dtype=torch.float32)
        metadata = {'identity_length': 1, 'fact_length': 1}
        with self.create() as runtime:
            registered = runtime.register_structured_text(config)
            self.assertTrue(registered['registered'])
            written = runtime.write_structured_text(4, packet, metadata)
            self.assertEqual(written['state_revision'], 1)
            record = runtime.read_structured_text(4)
            self.assertTrue(torch.equal(record['packet'], packet))
        with self.open() as restored:
            record = restored.read_structured_text(4)
            self.assertTrue(torch.equal(record['packet'], packet))
            self.assertEqual(record['metadata'], metadata)
            self.assertEqual(record['revision'], 1)

    def test_relation_records_preserve_legacy_and_isolate_corrections(self):
        config = {'memory_width': 64, 'identity_slots': 1, 'fact_slots': 1}
        metadata = {'identity_length': 1, 'fact_length': 1}
        with self.create() as runtime:
            runtime.write_structured_text(0, torch.full((64,), 9.), metadata, codec_config=config)
            first = runtime.write_structured_relation(4, 'drink', torch.ones(64), metadata)
            second = runtime.write_structured_relation(4, 'hobby', torch.full((64,), 2.), metadata)
            self.assertNotEqual(first['slot'], second['slot'])
            self.assertNotEqual(first['slot'], 0)
            with self.assertRaisesRegex(ValueError, 'explicit replacement'):
                runtime.write_structured_relation(4, 'drink', torch.full((64,), 3.), metadata)
            with self.assertRaisesRegex(ValueError, 'Relation-bound'):
                runtime.write_structured_text(first['slot'], torch.zeros(64), metadata)
            runtime.write_structured_relation(4, 'drink', torch.full((64,), 3.), metadata,
                                              replace_existing=True)
            self.assertIsNone(runtime.read_structured_relation(4, 'unknown'))
        with self.open() as restored:
            self.assertTrue(torch.equal(restored.read_structured_text(0)['packet'], torch.full((64,), 9.)))
            self.assertTrue(torch.equal(restored.read_structured_relation(4, 'drink')['packet'], torch.full((64,), 3.)))
            self.assertTrue(torch.equal(restored.read_structured_relation(4, 'hobby')['packet'], torch.full((64,), 2.)))
            self.assertIsNone(restored.read_structured_relation(5, 'hobby'))

    def test_relation_transaction_recovers_and_rejects_stale_allocation(self):
        config = {'memory_width': 64, 'identity_slots': 1, 'fact_slots': 1}
        metadata = {'identity_length': 1, 'fact_length': 1}
        with self.create() as first:
            first.register_structured_text(config)
            with self.open() as stale:
                first.write_structured_relation(4, 'drink', torch.ones(64), metadata)
                with self.assertRaises(RevisionConflict):
                    stale.write_structured_relation(4, 'hobby', torch.full((64,), 2.), metadata)
                self.assertIsNone(stale.read_structured_relation(4, 'hobby'))
            def interrupt(stage):
                if stage == 'after_manifest':
                    raise RuntimeError('injected interruption')
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                first.write_structured_relation(4, 'hobby', torch.full((64,), 2.), metadata, hook=interrupt)
            self.assertIsNone(first.read_structured_relation(4, 'hobby'))
        with self.open() as restored:
            self.assertTrue(torch.equal(restored.read_structured_relation(4, 'drink')['packet'], torch.ones(64)))
            self.assertIsNone(restored.read_structured_relation(4, 'hobby'))

    def test_relation_process_termination_preserves_atomicity(self):
        code = r'''
import os, sys, torch
from pathlib import Path
from types import SimpleNamespace
from tests.test_integrated_open_base import Base
from spora.integrated_durable import load_integrated_durable
root = Path(sys.argv[1])
base = Base(); base.config = SimpleNamespace(hidden_size=3, num_attention_heads=1)
runtime = load_integrated_durable(root/'individual.sqlite', root/'bootstrap.pt', base, root)
def stop(stage):
    if stage == sys.argv[2]: os._exit(42)
runtime.write_structured_relation(4, sys.argv[3], torch.ones(64),
    {'identity_length': 1, 'fact_length': 1}, hook=stop)
os._exit(0)
'''
        with self.create() as runtime:
            runtime.register_structured_text({'memory_width': 64, 'identity_slots': 1, 'fact_slots': 1})
        for stage, relation, present in [('after_manifest', 'before', False), ('after_commit', 'after', True)]:
            result = subprocess.run([sys.executable, '-c', code, str(self.root), stage, relation],
                                    cwd=Path(__file__).resolve().parents[1], timeout=60)
            self.assertEqual(result.returncode, 42)
            with self.open() as restored:
                record = restored.read_structured_relation(4, relation)
                self.assertEqual(record is not None, present)
                if present:
                    self.assertTrue(torch.equal(record['packet'], torch.ones(64)))

    def test_v4_adapter_survives_incremental_write_and_rejects_mutation(self):
        from spora.integrated_checkpoint import load_integrated
        decoder, organism, state = load_integrated(self.bootstrap, self.base(), self.root)
        replacement = decoder.base.model.layers[1]
        adapter = MemoryOutputAdapter(organism.config.value_count, 9, rank=4)
        with torch.no_grad():
            adapter.rank_to_vocab.weight[0, 0] = 0.5
        save_integrated(self.bootstrap, organism, state, decoder.reader, self.root, (0, 2),
                        replacements={1: replacement}, output_adapter=adapter)
        original = self.bootstrap.read_bytes()
        with self.create() as runtime:
            expected = {name: tensor.detach().clone() for name, tensor in
                        runtime.decoder.output_adapter.state_dict().items()}
            runtime.observe_sourced(4, 2, 'direct', 1.0)
            with torch.no_grad():
                runtime.decoder.output_adapter.rank_to_vocab.weight[0, 0] += 1
            with self.assertRaisesRegex(ValueError, '[Dd]ecoder'):
                runtime.observe_sourced(5, 3, 'direct', 1.0)
            self.assertEqual(runtime.state.cursor, 1)
        self.assertEqual(self.bootstrap.read_bytes(), original)
        with self.open() as restored:
            self.assertEqual(restored.state.cursor, 1)
            for name, tensor in restored.decoder.output_adapter.state_dict().items():
                self.assertTrue(torch.equal(tensor, expected[name]), name)

    def test_failed_transaction_does_not_publish_state(self):
        with self.create() as runtime:
            digest = frozen_digest(runtime.organism, runtime.state)
            def fail(stage):
                if stage == 'after_first_tensor':
                    raise RuntimeError('injected failure')
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                runtime.observe_sourced(4, 2, 'direct', 1.0, hook=fail)
            self.assertEqual(runtime.state.cursor, 0)
            self.assertEqual(frozen_digest(runtime.organism, runtime.state), digest)
        with self.open() as restored:
            self.assertEqual(restored.state.cursor, 0)
            self.assertEqual(frozen_digest(restored.organism, restored.state), digest)

    def test_failure_after_commit_publishes_committed_state(self):
        with self.create() as runtime:
            def fail(stage):
                if stage == 'after_commit':
                    raise RuntimeError('injected post-commit failure')
            with self.assertRaisesRegex(RuntimeError, 'post-commit'):
                runtime.observe_sourced(4, 2, 'direct', 1.0, hook=fail)
            self.assertEqual(runtime.state.cursor, 1)
            self.assertEqual(runtime.store.revision, 1)
        with self.open() as restored:
            self.assertEqual(restored.state.cursor, 1)

    def test_stale_writer_is_rejected_without_publishing_state(self):
        with self.create() as first, self.open() as second:
            first.observe_sourced(4, 2, 'direct', 1.0)
            with self.assertRaises(RevisionConflict):
                second.observe_sourced(5, 3, 'direct', 1.0)
            self.assertEqual(second.state.cursor, 0)

    def test_calendar_commit_keeps_effective_and_observed_time_distinct(self):
        first_day, next_day = date(2026, 1, 1), date(2026, 1, 3)
        with self.create(calendar=True) as writer, self.open() as stale:
            self.assertEqual(writer.observation_calendar.days, ())
            writer.observe_sourced(4, 2, 'direct', 1., observed_on=first_day)
            with self.assertRaises(RevisionConflict):
                stale.observe_sourced(5, 3, 'direct', 1., observed_on=first_day)
            self.assertEqual(stale.observation_calendar.days, ())
            writer.observe_sourced(5, 3, 'direct', 1., observed_on=next_day, effective_on=first_day)
            with self.assertRaisesRegex(ValueError, 'earlier effective date'):
                writer.observe_sourced(6, 4, 'direct', 1., observed_on=next_day, effective_on=date(2026, 1, 2))
            with self.assertRaisesRegex(ValueError, 'nondecreasing'):
                writer.observe_sourced(6, 4, 'direct', 1., observed_on=first_day)
            self.assertEqual(writer.store.revision, 2)
        with self.open() as restored:
            self.assertEqual(restored.observation_calendar.days, (first_day, next_day))
            self.assertEqual(restored.state.cursor, 2)
            self.assertEqual(read_provenance(restored.organism, restored.state, 5, 1)['valid_at'], 0)

    def test_calendar_consolidation_gates_and_recovers_the_old_page(self):
        self.write_bootstrap(time_count=2)
        first, second, third = date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)
        with self.create(calendar=True) as runtime:
            runtime.observe_sourced(4, 2, 'direct', 1., observed_on=first)
            runtime.observe_sourced(5, 3, 'direct', 1., observed_on=second)
            original = frozen_digest(runtime.organism, runtime.state)
            rejected = runtime.consolidate_history(0, rank=8, method='svd', verify=lambda _: False)
            self.assertFalse(rejected['accepted'])
            self.assertIs(rejected['decoder_verified'], False)
            self.assertEqual(runtime.store.revision, 2)

            def fail(stage):
                if stage == 'after_manifest':
                    raise RuntimeError('sleep transaction interrupted')

            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                runtime.consolidate_history(0, rank=8, method='svd', hook=fail)
            self.assertEqual(frozen_digest(runtime.organism, runtime.state), original)
        with self.open() as runtime:
            self.assertEqual(frozen_digest(runtime.organism, runtime.state), original)
            result = runtime.consolidate_history(0, rank=8, method='svd')
            self.assertTrue(result['accepted'])
            self.assertEqual(result['indexed_gate']['same_value'], 2)
            runtime.observe_sourced(6, 4, 'direct', 1., observed_on=third)
            final = frozen_digest(runtime.organism, runtime.state)
        with self.open() as restored:
            self.assertEqual(frozen_digest(restored.organism, restored.state), final)
            self.assertIsInstance(restored.state.pages[0], SealedPage)
            self.assertEqual(restored.observation_calendar.days, (first, second, third))

    def test_decoder_mutation_and_bootstrap_change_block_commit(self):
        with self.create() as runtime:
            layer = runtime.decoder.base.model.layers[1]
            with torch.no_grad():
                layer.position.weight[0, 0] += 1
            with self.assertRaisesRegex(ValueError, '[Dd]ecoder'):
                runtime.observe_sourced(4, 2, 'direct', 1.0)
            self.assertEqual(runtime.state.cursor, 0)
        with self.open() as runtime:
            with self.bootstrap.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'bootstrap'):
                runtime.observe_sourced(4, 2, 'direct', 1.0)
            self.assertEqual(runtime.state.cursor, 0)

    def test_changed_bootstrap_is_rejected_before_modifying_base(self):
        with self.create():
            pass
        other = self.root/'other-bootstrap.pt'
        other.write_bytes(self.bootstrap.read_bytes() + b'changed')
        base = self.base()
        previous = list(base.model.layers)
        with self.assertRaisesRegex(ValueError, 'do not match'):
            load_integrated_durable(self.database, other, base, self.root)
        self.assertTrue(all(a is b for a, b in zip(base.model.layers, previous)))

    def test_process_termination_before_and_after_commit(self):
        self.write_bootstrap(time_count=2)
        code = r'''
import os, sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from tests.test_integrated_open_base import Base
from spora.integrated_durable import load_integrated_durable
root = Path(sys.argv[1])
base = Base()
base.config = SimpleNamespace(hidden_size=3, num_attention_heads=1)
runtime = load_integrated_durable(root/'individual.sqlite', root/'bootstrap.pt', base, root)
def stop(stage):
    if stage == sys.argv[2]: os._exit(42)
runtime.observe_sourced(4, 2, 'direct', 1.0, observed_on=date(2026, 1, 3), hook=stop)
os._exit(0)
'''
        with self.create(calendar=True) as runtime:
            runtime.observe_sourced(2, 0, 'direct', 1.0, observed_on=date(2026, 1, 1))
            runtime.observe_sourced(3, 1, 'direct', 1.0, observed_on=date(2026, 1, 2))
            initial = frozen_digest(runtime.organism, runtime.state)
        failed = subprocess.run([sys.executable, '-c', code, str(self.root), 'after_first_tensor'],
                                capture_output=True, timeout=60)
        self.assertEqual(failed.returncode, 42, failed.stderr.decode(errors='replace'))
        with self.open() as restored:
            self.assertEqual(restored.state.cursor, 2)
            self.assertEqual(len(restored.state.pages), 1)
            self.assertEqual(restored.observation_calendar.days, (date(2026, 1, 1), date(2026, 1, 2)))
            self.assertEqual(frozen_digest(restored.organism, restored.state), initial)
        committed = subprocess.run([sys.executable, '-c', code, str(self.root), 'after_commit'],
                                   capture_output=True, timeout=60)
        self.assertEqual(committed.returncode, 42, committed.stderr.decode(errors='replace'))
        with self.open() as restored:
            self.assertEqual(restored.state.cursor, 3)
            self.assertEqual(len(restored.state.pages), 2)
            self.assertEqual(restored.store.revision, 3)
            self.assertEqual(restored.observation_calendar.days[-1], date(2026, 1, 3))
            self.assertEqual(restored.state.internal.event_keys[1].tolist(), [4, -1])


if __name__ == '__main__':
    unittest.main()
