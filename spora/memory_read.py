"""Inference read plans derived only from the individual's neural matrices.

Identical matrices share one matrix-vector product while retaining every page's
position weight. Optional pruning uses conservative attention-mass bounds and
falls back to full reading. It does not infer semantic relevance.
"""
import math

import torch


class MemoryReadPlan:
    def __init__(self, matrices):
        if not matrices:
            raise ValueError('At least one memory matrix required')
        first = matrices[0]
        if first.ndim != 3 or first.shape[0] != 1 or first.shape[1] != first.shape[2]:
            raise ValueError('Singleton square memory matrices required')
        if first.dtype not in (torch.float32, torch.float64):
            raise ValueError('Bounds require float32 or float64')
        if any(torch.is_inference(matrix) for matrix in matrices):
            raise ValueError('Unversioned inference tensors require live reads, not reusable plans')
        self.sources = tuple(matrices)
        self.versions = tuple(x._version for x in matrices)
        distinct, mapping = [], []
        for matrix in matrices:
            if matrix.shape != first.shape or matrix.dtype != first.dtype or matrix.device != first.device:
                raise ValueError('Memory shapes, precision and device must match')
            if not bool(torch.isfinite(matrix).all()):
                raise ValueError('Finite memory required')
            match = next((i for i, other in enumerate(distinct) if torch.equal(matrix, other)), None)
            if match is None:
                match = len(distinct)
                distinct.append(matrix)
            mapping.append(match)
        self.banks = torch.stack([x[0].detach() for x in distinct])
        self.mapping = torch.tensor(mapping, device=first.device, dtype=torch.long)
        self.membership = torch.nn.functional.one_hot(self.mapping, len(distinct)).to(first.dtype)
        self.page_count = len(mapping)
        self.width = first.shape[-1]
        self._bounds = None

    def matches(self, matrices):
        return len(matrices) == len(self.sources) and all(
            new is old and new._version == version
            for new, old, version in zip(matrices, self.sources, self.versions))

    @property
    def tensor_bytes(self):
        tensors = [self.banks, self.mapping, self.membership]
        if self._bounds is not None:
            tensors += list(self._bounds)
        return sum(x.numel() * x.element_size() for x in tensors)

    def _prepare_bounds(self):
        if self._bounds is None:
            diagonal = self.banks.diagonal(dim1=-2, dim2=-1).clone()
            residual = self.banks.double() - torch.diag_embed(diagonal.double())
            # Frobenius bounds dominate spectral norms. Inflate for FP rounding;
            # this is not an interval-arithmetic or bitwise-output guarantee.
            epsilon = torch.finfo(self.banks.dtype).eps
            residual_norm = (residual.square().sum((-1, -2)).sqrt() * (1 + 32 * epsilon) + 32 * epsilon).to(self.banks.dtype)
            norm = (self.banks.double().square().sum((-1, -2)).sqrt() * (1 + 32 * epsilon) + 32 * epsilon).to(self.banks.dtype)
            self._bounds = (diagonal, residual_norm, norm)
        return self._bounds

    def _full(self, queries, page_keys):
        values = torch.einsum('pmd,td->tpm', self.banks, queries)
        content = (values * queries[:, None]).sum(-1)
        scores = (content.index_select(1, self.mapping) + queries @ page_keys.T) / math.sqrt(self.width)
        weights = scores.softmax(-1)
        if self.banks.shape[0] != self.page_count:
            # Matrix aggregation avoids CUDA atomic additions for shared pages.
            weights = weights @ self.membership
        return (values * weights.unsqueeze(-1)).sum(1)

    def _selected(self, queries, page_keys, pages):
        matrices = self.banks[self.mapping[pages]]
        values = torch.einsum('tkmd,td->tkm', matrices, queries)
        scores = ((values * queries[:, None]).sum(-1) +
                  (page_keys[pages] * queries[:, None]).sum(-1)) / math.sqrt(self.width)
        return (values * scores.softmax(-1).unsqueeze(-1)).sum(1)

    def read(self, queries, page_keys, mode='exact', error_l2=1e-4, max_selected=2):
        if torch.is_grad_enabled():
            raise RuntimeError('Prepared plans are inference-only; use the differentiable training path')
        if not self.matches(self.sources):
            raise ValueError('Memory changed; rebuild the read plan')
        if mode not in ('exact', 'bounded') or not math.isfinite(error_l2) or error_l2 < 0:
            raise ValueError('Invalid read policy or tolerance')
        if type(max_selected) is not int or max_selected < 1:
            raise ValueError('Positive candidate budget required')
        if queries.ndim < 2 or queries.shape[-1] != self.width or page_keys.shape != (self.page_count, self.width):
            raise ValueError('Query or page-key shape mismatch')
        if queries.numel() == 0 or queries.device != self.banks.device or queries.dtype != self.banks.dtype:
            raise ValueError('Nonempty queries must match memory device and precision')
        if page_keys.device != queries.device or page_keys.dtype != queries.dtype:
            raise ValueError('Page keys must match query device and precision')
        if mode == 'bounded' and (not bool(torch.isfinite(queries).all()) or not bool(torch.isfinite(page_keys).all())):
            raise ValueError('Finite query and page keys required')
        shape = queries.shape
        flat = queries.reshape(-1, self.width)
        count, unique = flat.shape[0], self.banks.shape[0]
        stats = {'mode': mode, 'queries': count, 'pages': self.page_count,
                 'unique_matrices': unique, 'baseline_matrix_vectors': count * self.page_count,
                 'matrix_vectors': count * unique, 'pruned_queries': 0,
                 'fallback_queries': count if mode == 'bounded' else 0, 'max_read_error_bound': 0.0}
        if mode == 'exact' or self.page_count < 3 or unique == 1 or error_l2 == 0:
            return self._full(flat, page_keys).reshape(shape), stats

        diagonal, residual_norm, norm = self._prepare_bounds()
        norm_q = flat.square().sum(-1).sqrt()
        proxy = (flat.square() @ diagonal[self.mapping].T + flat @ page_keys.T) / math.sqrt(self.width)
        radius = norm_q[:, None].square() * residual_norm[self.mapping] / math.sqrt(self.width)
        roundoff = 32 * torch.finfo(flat.dtype).eps * (proxy.abs() + radius + 1)
        upper, lower = proxy + radius + roundoff, proxy - radius - roundoff
        budget = min(max_selected, self.page_count - 1, unique - 1)
        spread = upper.max(-1).values - lower.min(-1).values
        minimum_bound = 2 * norm.max() * norm_q * torch.sigmoid(
            math.log(self.page_count-budget)-math.log(budget)-spread)
        # If even the most favorable score separation cannot meet our own
        # certificate, avoid sorting and candidate evaluation altogether.
        if bool((minimum_bound > error_l2).all()):
            return self._full(flat, page_keys).reshape(shape), stats
        order = upper.argsort(-1, descending=True)
        sorted_upper, sorted_lower = upper.gather(1, order), lower.gather(1, order)
        selected_counts = torch.zeros(count, dtype=torch.long, device=flat.device)
        accepted_bound = torch.zeros(count, dtype=flat.dtype, device=flat.device)
        # The dense mixture and a selected mixture differ by at most
        # omitted_mass * (max omitted value norm + selected value norm).
        # 2 * max_page_norm * ||q|| is a conservative common upper bound.
        for k in range(1, min(max_selected, self.page_count - 1, unique - 1) + 1):
            mass = torch.sigmoid(torch.logsumexp(sorted_upper[:, k:], -1) -
                                 torch.logsumexp(sorted_lower[:, :k], -1))
            bound = 2 * norm.max() * norm_q * mass
            accept = (selected_counts == 0) & torch.isfinite(bound) & (bound <= error_l2)
            selected_counts = torch.where(accept, k, selected_counts)
            accepted_bound = torch.where(accept, bound, accepted_bound)
        if not bool((selected_counts > 0).any()):
            return self._full(flat, page_keys).reshape(shape), stats

        result = torch.empty_like(flat)
        full = selected_counts == 0
        full_count = int(full.sum())
        if full_count:
            result[full] = self._full(flat[full], page_keys)
        operations = full_count * unique
        for k in range(1, min(max_selected, self.page_count - 1, unique - 1) + 1):
            mask = selected_counts == k
            chosen_count = int(mask.sum())
            if chosen_count:
                result[mask] = self._selected(flat[mask], page_keys, order[mask, :k])
                operations += k * chosen_count
        stats.update(matrix_vectors=operations, pruned_queries=count-full_count,
                     fallback_queries=full_count, max_read_error_bound=float(accepted_bound.max()))
        return result.reshape(shape), stats
