"""Explicit new-curriculum lineage and deterministic, covering group sampler."""
import copy
import hashlib
import json
import torch
from .language_training import ByteTrainer, load_interface


class CoveringSampler:
    def __init__(self, groups, schedule, seed):
        self.groups = {name: list(indices) for name, indices in groups.items()}
        self.schedule = list(schedule)
        if not self.groups or any(not v for v in self.groups.values()) or set(schedule) != set(groups):
            raise ValueError('Every nonempty group must appear in the schedule')
        flat = [i for values in self.groups.values() for i in values]
        if any(type(i) is not int or i < 0 for i in flat) or len(set(flat)) != len(flat):
            raise ValueError('Groups require unique nonnegative example positions')
        self.signature = hashlib.sha256(json.dumps([self.groups, self.schedule], sort_keys=True).encode()).hexdigest()
        self.generator = torch.Generator().manual_seed(seed)
        self.orders = {k: [] for k in self.groups}
        self.positions = {k: 0 for k in self.groups}
        self.draws = 0
        self.counts = {i: 0 for i in flat}

    def take(self, count):
        if type(count) is not int or not 1 <= count <= 16:
            raise ValueError('Finite batch size required')
        result = []
        for _ in range(count):
            group = self.schedule[self.draws % len(self.schedule)]
            if self.positions[group] == len(self.orders[group]):
                permutation = torch.randperm(len(self.groups[group]), generator=self.generator).tolist()
                self.orders[group] = [self.groups[group][i] for i in permutation]
                self.positions[group] = 0
            index = self.orders[group][self.positions[group]]
            self.positions[group] += 1
            self.counts[index] += 1
            self.draws += 1
            result.append(index)
        return torch.tensor(result, dtype=torch.long)

    def snapshot(self):
        return {'signature': self.signature, 'rng': self.generator.get_state(),
            'orders': copy.deepcopy(self.orders), 'positions': dict(self.positions),
            'draws': self.draws, 'counts': dict(self.counts)}

    def restore(self, data):
        if data['signature'] != self.signature or set(data['orders']) != set(self.groups):
            raise ValueError('Sampler curriculum differs')
        if type(data['draws']) is not int or data['draws'] < 0 or set(data['counts']) != set(self.counts):
            raise ValueError('Invalid sampler counters')
        if any(type(v) is not int or v < 0 for v in data['counts'].values()) or sum(data['counts'].values()) != data['draws']:
            raise ValueError('Inconsistent coverage counters')
        for group, order in data['orders'].items():
            if order and sorted(order) != sorted(self.groups[group]):
                raise ValueError('Invalid group permutation')
            if type(data['positions'][group]) is not int or not 0 <= data['positions'][group] <= len(order):
                raise ValueError('Invalid traversal position')
        self.generator.set_state(data['rng'])
        self.orders = copy.deepcopy(data['orders'])
        self.positions = dict(data['positions'])
        self.draws, self.counts = data['draws'], dict(data['counts'])


class CurriculumTrainer(ByteTrainer):
    def __init__(self, model, state, parent_sha256, provenance, seed, sampler):
        super().__init__(model, state, parent_sha256, provenance, seed)
        self.sampler = sampler
        self.lineage = None

    def begin(self, payload, source_sha256):
        if self.steps or self.lineage is not None or self.sampler.draws:
            raise ValueError('New curriculum can only begin once on a pristine trainer')
        if self.provenance.get('source_checkpoint_sha256') != source_sha256 or payload.get('format') != 'spora-byte-training-v2':
            raise ValueError('Explicit trained source lineage required')
        steps = payload.get('steps')
        if type(steps) is not int or not 1 <= steps <= 10000:
            raise ValueError('Invalid source training count')
        previous = payload.get('lineage')
        prior_steps = previous['prior_steps'] if previous else 0
        load_interface(self.model, payload, self.parent_sha256)
        self.lineage = {'source_checkpoint_sha256': source_sha256, 'prior_steps': prior_steps+steps,
            'prior_targets': payload['trained_tokens'] + (previous.get('prior_targets', 0) if previous else 0),
            'optimizer_policy': 'Fresh Adam state for explicitly changed curriculum; interface weights inherited'}

    def step(self, examples, batch_size):
        if self.lineage is None:
            raise ValueError('Begin or restore the explicit curriculum before learning')
        return self.step_batch(examples[self.sampler.take(batch_size)])

    def snapshot(self):
        data = super().snapshot()
        data.update(lineage=copy.deepcopy(self.lineage), covering_sampler=self.sampler.snapshot())
        return data

    def restore(self, payload):
        if not payload.get('lineage') or not payload.get('covering_sampler'):
            raise ValueError('Use begin for a different curriculum, not resume')
        if payload['lineage']['source_checkpoint_sha256'] != self.provenance.get('source_checkpoint_sha256'):
            raise ValueError('Curriculum lineage changed')
        # Validate on a separate sampler before installing any learned tensors.
        candidate = copy.deepcopy(self.sampler)
        candidate.restore(payload['covering_sampler'])
        if candidate.draws != payload['steps']*self.provenance['batch']:
            raise ValueError('Sampler and optimizer steps disagree')
        super().restore(payload)
        self.sampler, self.lineage = candidate, copy.deepcopy(payload['lineage'])
