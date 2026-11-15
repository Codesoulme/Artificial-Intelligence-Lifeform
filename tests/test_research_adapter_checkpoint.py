import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from spora import research_adapter_checkpoint as checkpoint


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(3, 2, bias=False)
        self.up = nn.Linear(2, 3, bias=False)

    def forward(self, value):
        return self.up(self.down(value))


class ResearchAdapterCheckpointTests(unittest.TestCase):
    def test_roundtrip_and_optimizer_restore(self):
        torch.manual_seed(31)
        source = Adapter()
        optimizer = torch.optim.AdamW(source.parameters(), lr=0.01)
        source(torch.ones(1, 3)).square().sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.pt'
            checkpoint.save(path, [source], optimizer, base_sha256='base',
                            feature_sha256='features', layers=[2], steps=1)
            target = Adapter()
            restored_optimizer = torch.optim.AdamW(target.parameters(), lr=0.01)
            info = checkpoint.restore(path, [target], base_sha256='base',
                                      feature_sha256='features', layers=[2], optimizer=restored_optimizer)
            self.assertEqual(info['steps'], 1)
            torch.testing.assert_close(target(torch.ones(1, 3)), source(torch.ones(1, 3)))
            for source_state, target_state in zip(optimizer.state.values(), restored_optimizer.state.values()):
                for key in source_state:
                    torch.testing.assert_close(source_state[key], target_state[key])
            with self.assertRaises(FileExistsError):
                checkpoint.save(path, [source], optimizer, base_sha256='base',
                                feature_sha256='features', layers=[2], steps=1)

    def test_wrong_binding_and_bad_tensor_rejected_before_restore(self):
        source, target = Adapter(), Adapter()
        optimizer = torch.optim.AdamW(source.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.pt'
            checkpoint.save(path, [source], optimizer, base_sha256='base',
                            feature_sha256='features', layers=[2], steps=0)
            original = target.down.weight.detach().clone()
            with self.assertRaisesRegex(ValueError, 'binding'):
                checkpoint.restore(path, [target], base_sha256='other',
                                   feature_sha256='features', layers=[2])
            payload = torch.load(path, map_location='cpu', weights_only=True)
            payload['adapters'][0]['up.weight'][0, 0] = float('nan')
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'tensor'):
                checkpoint.restore(path, [target], base_sha256='base',
                                   feature_sha256='features', layers=[2])
            torch.testing.assert_close(target.down.weight, original)

    def test_invalid_optimizer_does_not_modify_adapter(self):
        source, target = Adapter(), Adapter()
        optimizer = torch.optim.AdamW(source.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.pt'
            checkpoint.save(path, [source], optimizer, base_sha256='base',
                            feature_sha256='features', layers=[2], steps=0)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            payload['optimizer']['param_groups'][0]['params'].pop()
            torch.save(payload, path)
            original = {key: value.clone() for key, value in target.state_dict().items()}
            target_optimizer = torch.optim.AdamW(target.parameters())
            with self.assertRaises(ValueError):
                checkpoint.restore(path, [target], base_sha256='base',
                                   feature_sha256='features', layers=[2], optimizer=target_optimizer)
            for key, value in target.state_dict().items():
                torch.testing.assert_close(value, original[key])
            self.assertEqual(len(target_optimizer.state), 0)

    def test_nonfinite_optimizer_rejected(self):
        source = Adapter()
        optimizer = torch.optim.AdamW(source.parameters())
        source(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        next(iter(optimizer.state.values()))['exp_avg'][0, 0] = float('nan')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Nonfinite optimizer'):
                checkpoint.save(Path(directory) / 'candidate.pt', [source], optimizer,
                                base_sha256='base', feature_sha256='features', layers=[2], steps=1)

    def test_optimizer_parameter_order_and_ownership(self):
        source, target, unrelated = Adapter(), Adapter(), Adapter()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.pt'
            with self.assertRaisesRegex(ValueError, 'parameters do not match'):
                checkpoint.save(path, [source], torch.optim.AdamW(unrelated.parameters()),
                                base_sha256='base', feature_sha256='features', layers=[2], steps=0)
            checkpoint.save(path, [source], torch.optim.AdamW(source.parameters()),
                            base_sha256='base', feature_sha256='features', layers=[2], steps=0)
            reversed_optimizer = torch.optim.AdamW(list(target.parameters())[::-1])
            original = target.down.weight.detach().clone()
            with self.assertRaisesRegex(ValueError, 'parameter order'):
                checkpoint.restore(path, [target], base_sha256='base', feature_sha256='features',
                                   layers=[2], optimizer=reversed_optimizer)
            torch.testing.assert_close(target.down.weight, original)

    def test_legacy_optimizer_requires_explicit_permission(self):
        source, target = Adapter(), Adapter()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'candidate.pt'
            checkpoint.save(path, [source], torch.optim.AdamW(source.parameters()),
                            base_sha256='base', feature_sha256='features', layers=[2], steps=0)
            payload = torch.load(path, map_location='cpu', weights_only=True)
            del payload['optimizer_layout']
            torch.save(payload, path)
            optimizer = torch.optim.AdamW(target.parameters())
            with self.assertRaisesRegex(ValueError, 'Legacy optimizer'):
                checkpoint.restore(path, [target], base_sha256='base', feature_sha256='features',
                                   layers=[2], optimizer=optimizer)
            checkpoint.restore(path, [target], base_sha256='base', feature_sha256='features',
                               layers=[2], optimizer=optimizer, allow_legacy_optimizer=True)


if __name__ == '__main__':
    unittest.main()
