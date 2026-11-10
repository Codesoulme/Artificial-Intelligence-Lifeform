import unittest
from types import SimpleNamespace

import torch
from torch import nn

from spora.integrated_open_base import IntegratedOpenBase, NativeSPORAMemoryReader, ScaledOpenBaseMemory
from spora.open_base_memory import OpenBaseMemory, memory_hooks


class Decoder(nn.Module):
    attention_type = 'full_attention'

    def forward(self, x):
        return x + 0.1


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Decoder(), Decoder(), Decoder()])

    def forward(self, x, fail=False, **kwargs):
        for layer in self.model.layers:
            x = layer(x)
        if fail:
            raise RuntimeError('failure')
        return x

    def generate(self, x, **kwargs):
        return self.forward(x, **kwargs)


def make_state(scale):
    return SimpleNamespace(current=[torch.eye(3).unsqueeze(0) * scale for _ in range(2)], pages=[])


def make_reader():
    reader = OpenBaseMemory(3, 3)
    with torch.no_grad():
        for query, output in zip(reader.query, reader.out):
            query.weight.copy_(torch.eye(3))
            output.weight.copy_(torch.eye(3))
        reader.gate.zero_()
    return reader


class IntegratedOpenBaseTests(unittest.TestCase):
    def test_native_dynamic_address_cannot_read_future_tokens(self):
        torch.manual_seed(90)
        reader = NativeSPORAMemoryReader(3, 3, 1.0).eval()
        with torch.no_grad():
            for site in reader.native_sites:
                site.key.weight.copy_(torch.eye(3))
                site.key.bias.zero_()
                site.read_out.weight.copy_(torch.eye(3))
                site.read_out.bias.zero_()
                site.read_gate.weight.zero_()
                site.read_gate.bias.zero_()
        hidden = torch.tensor([[[1., 0., -1.], [0., 1., -1.], [-1., 0., 1.]]])
        pages = [torch.eye(3).unsqueeze(0), -torch.eye(3).unsqueeze(0)]
        with torch.no_grad():
            prefix = reader(hidden[:, :2], pages, 0)
            extended = reader(hidden, pages, 0)
        torch.testing.assert_close(prefix, extended[:, :2], atol=1e-6, rtol=1e-6)

    def test_matches_existing_hooks_for_multiple_states_and_generate(self):
        base, reader = Base(), make_reader()
        x = torch.tensor([[[1., 0., 0.]]])
        memories = [make_state(1.), make_state(2.)]
        addresses = [torch.tensor([[1., 0., 0.]]), torch.tensor([[1., 0., 0.]])]
        plain = base(x)
        expected = []
        for memory in memories:
            with memory_hooks([base.model.layers[0], base.model.layers[2]], reader, memory,
                              current_only=True, addresses=addresses):
                expected.append(base(x))
        integrated = IntegratedOpenBase(base, reader, (0, 2))
        self.assertTrue(torch.equal(integrated(x), plain))
        for memory, target in zip(memories, expected):
            self.assertTrue(torch.equal(integrated(x, memory_state=memory, memory_addresses=addresses,
                                                   current_only=True), target))
            self.assertTrue(torch.equal(integrated.generate(x, memory_state=memory, memory_addresses=addresses,
                                                            current_only=True), target))
        self.assertFalse(torch.equal(expected[0], expected[1]))
        self.assertFalse(any(layer._forward_hooks for layer in base.model.layers))

    def test_native_spora_delta_site_matches_mapped_legacy_reader(self):
        legacy = ScaledOpenBaseMemory(3, 3, 0.5)
        with torch.no_grad():
            for query, output in zip(legacy.query, legacy.out):
                query.weight.copy_(torch.eye(3))
                output.weight.copy_(torch.eye(3))
            legacy.gate.zero_()
        native = NativeSPORAMemoryReader(3, 3, 0.5)
        native.initialize_from_legacy(legacy)
        hidden = torch.randn(2, 4, 3)
        matrix = torch.randn(2, 3, 3)
        address = torch.randn(2, 3)
        for site in range(2):
            for matrices in ([matrix], [matrix, matrix * 0.3]):
                expected = legacy(hidden, matrices, site, address)
                actual = native(hidden, matrices, site, address)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        self.assertFalse(hasattr(native, 'query'))
        self.assertFalse(hasattr(native, 'out'))

    def test_error_cleans_state_and_external_cache_is_rejected(self):
        base = Base()
        model = IntegratedOpenBase(base, make_reader(), (0, 2))
        x = torch.ones(1, 1, 3)
        plain = model(x)
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            model(x, memory_state=make_state(1), fail=True)
        self.assertTrue(torch.equal(model(x), plain))
        with self.assertRaisesRegex(ValueError, 'caches'):
            model(x, memory_state=make_state(1), past_key_values=object())
        with self.assertRaisesRegex(ValueError, 'positional'):
            model(x, None, memory_state=make_state(1))
        with self.assertRaises(ValueError):
            model(x, memory_addresses=[torch.ones(1, 3)] * 2)

    def test_adapter_receives_gradients_and_memory_is_detached(self):
        reader = make_reader()
        model = IntegratedOpenBase(Base(), reader, (0, 2))
        memory = make_state(1.)
        for matrix in memory.current:
            matrix.requires_grad_()
        model(torch.ones(1, 1, 3), memory_state=memory, current_only=True).sum().backward()
        self.assertTrue(all(p.grad is not None for p in reader.parameters()))
        self.assertTrue(all(t.grad is None for t in memory.current))

    def test_per_request_read_mask_preserves_unselected_output(self):
        model = IntegratedOpenBase(Base(), make_reader(), (0, 2))
        x = torch.ones(2, 1, 3)
        memory = make_state(1.)
        plain = model(x)
        selected = model(x, memory_state=memory, current_only=True)
        mask = torch.tensor([True, False])
        for method in (model.forward, model.generate):
            result = method(x, memory_state=memory, memory_read_mask=mask, current_only=True)
            self.assertTrue(torch.equal(result[0], selected[0]))
            self.assertTrue(torch.equal(result[1], plain[1]))
            self.assertFalse(torch.equal(result[0], plain[0]))
        broadcast = model(x, memory_state=memory, memory_read_mask=torch.tensor([False]), current_only=True)
        self.assertTrue(torch.equal(broadcast, plain))

    def test_read_mask_rejects_invalid_or_mismatched_batch(self):
        model = IntegratedOpenBase(Base(), make_reader(), (0, 2))
        x = torch.ones(2, 1, 3)
        plain = model(x)
        with self.assertRaisesRegex(ValueError, 'require a memory state'):
            model(x, memory_read_mask=torch.tensor([True]))
        for bad in (torch.tensor([1, 0]), torch.empty(0, dtype=torch.bool), torch.ones(2, 1, dtype=torch.bool)):
            with self.subTest(mask=bad), self.assertRaisesRegex(ValueError, 'boolean vector'):
                model(x, memory_state=make_state(1), memory_read_mask=bad)
        with self.assertRaisesRegex(ValueError, 'decoder batch'):
            model(x, memory_state=make_state(1), memory_read_mask=torch.tensor([True, False, True]))
        self.assertTrue(torch.equal(model(x), plain))

    def test_read_mask_is_snapshotted_for_call(self):
        model = IntegratedOpenBase(Base(), make_reader(), (0, 2))
        mask = torch.tensor([True, False])
        with model._snapshot(make_state(1), None, True, mask, None):
            mask.fill_(False)
            self.assertEqual(model._memory_context.get()[2].tolist(), [True, False])
        self.assertIsNone(model._memory_context.get())

    def test_distinct_batch_states_follow_row_and_mask(self):
        model = IntegratedOpenBase(Base(), make_reader(), (0, 2))
        x = torch.ones(2, 1, 3)
        a, b = make_state(1.), make_state(2.)
        def batch(first, second):
            return SimpleNamespace(current=[torch.cat((first.current[i], second.current[i])) for i in range(2)], pages=[])
        forward = model(x, memory_state=batch(a, b), current_only=True)
        reverse = model(x, memory_state=batch(b, a), current_only=True)
        self.assertFalse(torch.equal(forward[0], forward[1]))
        self.assertTrue(torch.equal(forward[0], reverse[1]))
        self.assertTrue(torch.equal(forward[1], reverse[0]))
        masked = model(x, memory_state=batch(a, b), memory_read_mask=torch.tensor([True, False]), current_only=True)
        self.assertTrue(torch.equal(masked[0], forward[0]))
        self.assertTrue(torch.equal(masked[1], model(x)[1]))

    def test_masked_rows_are_not_sent_to_reader(self):
        reader = make_reader()
        model = IntegratedOpenBase(Base(), reader, (0, 2))
        x = torch.ones(3, 1, 3, requires_grad=True)
        memory = SimpleNamespace(current=[torch.eye(3).repeat(3, 1, 1) for _ in range(2)], pages=[])
        addresses = [torch.eye(3) for _ in range(2)]
        seen = []
        handle = reader.register_forward_pre_hook(lambda module, args: seen.append(
            (args[0].shape[0], args[1][0].shape[0], args[3].shape[0])))
        try:
            out = model(x, memory_state=memory, memory_addresses=addresses,
                        memory_read_mask=torch.tensor([False, True, False]), current_only=True)
            self.assertEqual(seen, [(1, 1, 1), (1, 1, 1)])
            out.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in list(reader.out.parameters()) + [reader.gate]))
            seen.clear()
            model(x.detach(), memory_state=memory, memory_read_mask=torch.tensor([False]), current_only=True)
            self.assertEqual(seen, [])
        finally:
            handle.remove()

    def test_inactive_nonfinite_state_does_not_contaminate_gradients(self):
        reader = make_reader()
        model = IntegratedOpenBase(Base(), reader, (0, 2))
        x = torch.ones(2, 1, 3, requires_grad=True)
        matrices = [torch.eye(3).repeat(2, 1, 1) for _ in range(2)]
        for matrix in matrices:
            matrix[1].fill_(float('nan'))
        memory = SimpleNamespace(current=matrices, pages=[])
        result = model(x, memory_state=memory, memory_read_mask=torch.tensor([True, False]), current_only=True)
        result.sum().backward()
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in reader.parameters()))

    def test_memory_slots_select_current_or_own_history_row(self):
        model = IntegratedOpenBase(Base(), make_reader(), (0, 2))
        x = torch.tensor([[[1., 0., 0.]], [[1., 0., 0.]]])
        current = [torch.eye(3).repeat(2, 1, 1) for _ in range(2)]
        page = [[torch.stack((torch.eye(3) * 2, torch.eye(3) * 3)),
                 torch.eye(3).repeat(2, 1, 1)] for _ in range(2)]
        memory = SimpleNamespace(current=current, pages=[page])
        addresses = [torch.tensor([[1., 0., 0.]]), torch.tensor([[1., 0., 0.]])]
        mixed = model(x, memory_state=memory, memory_addresses=addresses,
                      memory_slots=torch.tensor([0, 1]), current_only=False)
        def slice_state(row):
            one_page = [[tensor[row:row + 1] for tensor in site] for site in page]
            return SimpleNamespace(current=[m[row:row + 1] for m in current], pages=[one_page])
        current_batch = model(x[:1], memory_state=slice_state(0),
                              memory_addresses=addresses, memory_slots=torch.tensor([0]), current_only=False)
        history_batch = model(x[1:], memory_state=slice_state(1),
                              memory_addresses=addresses, memory_slots=torch.tensor([1]), current_only=False)
        self.assertTrue(torch.equal(mixed[0], current_batch[0]))
        self.assertTrue(torch.equal(mixed[1], history_batch[0]))
        self.assertFalse(torch.equal(mixed[0], mixed[1]))

    def test_memory_slots_validate_state_dtype_batch_and_range(self):
        model = IntegratedOpenBase(Base(), make_reader(), (0, 2))
        x = torch.ones(2, 1, 3)
        with self.assertRaisesRegex(ValueError, 'require a memory state'):
            model(x, memory_slots=torch.tensor([0], dtype=torch.long))
        memory = make_state(1.)
        for slots in (torch.tensor([0.]), torch.tensor([0, 0], dtype=torch.int32),
                      torch.tensor([0, 0, 0], dtype=torch.long), torch.tensor([-1, 0]), torch.tensor([0, 1])):
            with self.subTest(slots=slots), self.assertRaises((ValueError, RuntimeError)):
                model(x, memory_state=memory, memory_slots=slots, current_only=True)

    def test_per_request_memory_slots_select_current_or_history(self):
        model = IntegratedOpenBase(Base(), make_reader(), (0, 2))
        x = torch.tensor([[[1., 0., 0.]], [[1., 0., 0.]]])
        current = [torch.eye(3).repeat(2, 1, 1) for _ in range(2)]
        page = [[torch.stack((torch.eye(3) * 2, torch.eye(3) * 3)),
                 torch.eye(3).repeat(2, 1, 1)] for _ in range(2)]
        memory = SimpleNamespace(current=current, pages=[page])
        addresses = [torch.tensor([[1., 0., 0.]])] * 2
        mixed = model(x, memory_state=memory, memory_addresses=addresses,
                      memory_slots=torch.tensor([0, 1]), current_only=False)
        def one_row(row):
            single_page = [[tensor[row:row + 1] for tensor in site] for site in page]
            return SimpleNamespace(current=[tensor[row:row + 1] for tensor in current], pages=[single_page])
        current_only = model(x[:1], memory_state=one_row(0), memory_addresses=addresses,
                             memory_slots=torch.tensor([0]), current_only=False)
        history_only = model(x[1:], memory_state=one_row(1), memory_addresses=addresses,
                             memory_slots=torch.tensor([1]), current_only=False)
        self.assertTrue(torch.equal(mixed[0], current_only[0]))
        self.assertTrue(torch.equal(mixed[1], history_only[0]))
        self.assertFalse(torch.equal(mixed[0], mixed[1]))

    def test_memory_slot_requires_available_page(self):
        model = IntegratedOpenBase(Base(), make_reader(), (0, 2))
        with self.assertRaisesRegex(ValueError, 'outside'):
            model(torch.ones(1, 1, 3), memory_state=make_state(1),
                  memory_slots=torch.tensor([1]), current_only=True)


if __name__ == '__main__':
    unittest.main()
