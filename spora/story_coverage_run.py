"""Coverage contract and strict checkpoint validation for the independent entry."""
from spora.supervision_coverage import SupervisionCoverage


def coverage_contract(plan):
    enabled = plan.get('observe_story_coverage', False)
    boundary = plan.get('respect_story_document_boundaries', False)
    if type(enabled) is not bool or type(boundary) is not bool:
        raise ValueError('Coverage and story boundary switches must be booleans')
    shares = plan['target_shares']
    return {
        'version': 1,
        'enabled': enabled,
        'story_document_boundaries': boundary,
        'record_story': enabled and boundary,
        'scope': ('story_supervised_positions_only' if enabled and boundary else
                  'unavailable_story_boundaries_disabled' if enabled else 'disabled'),
        'excluded_sources': sorted(set(shares) - {'story'}) + ['focus_replacements'],
        'full_training_coverage': False,
    }


def restore_coverage(payload, contract, source, documents):
    if payload is None:
        return SupervisionCoverage()
    if 'coverage' not in payload or payload.get('coverage_contract') != contract:
        raise ValueError('Missing or incompatible coverage state')
    state = payload['coverage']
    if not isinstance(state, dict) or state.get('version') != 2:
        raise ValueError('Unsupported coverage state')
    coverage = SupervisionCoverage.from_dict(state)
    summaries = coverage.summaries()
    if not contract['record_story']:
        if summaries:
            raise ValueError('Coverage recorded outside its declared scope')
        return coverage
    if documents is None:
        raise ValueError('Story document boundaries required')
    total = 0
    for (digest, document), summary in summaries.items():
        if digest != source.lower() or document >= len(documents.starts):
            raise ValueError('Coverage source or document differs')
        length = int(documents.ends[document] - documents.starts[document])
        if any(start < 1 or end > length for start, end in summary.intervals):
            raise ValueError('Coverage position outside document')
        total += summary.total_positions
    counts = payload.get('counts')
    if not isinstance(counts, dict):
        raise ValueError('Missing source counts')
    expected = counts.get('story_source_targets', 0)
    if type(expected) is not int or expected < 0 or total != expected:
        raise ValueError('Coverage and supervised story counts differ')
    return coverage


def sample_story(documents, rng, batch_size, length, device, record):
    if record:
        from spora.document_stream_observation import batch_with_windows
        x, y, windows = batch_with_windows(documents, rng, batch_size, length, device)
        return x, y, windows, y.clone()
    x, y = documents.batch(rng, batch_size, length, device)
    return x, y, None, None


def record_story(coverage, source, windows, sampled_labels, final_labels, replaced_rows):
    if windows is not None:
        from spora.document_stream_observation import accumulate_unchanged_rows
        return accumulate_unchanged_rows(coverage, source, windows, sampled_labels, final_labels, replaced_rows)
    return 0


def validate_identity(payload, contract, fingerprints, source):
    if (not isinstance(payload, dict) or payload.get('coverage_entry') != 'spora-story-coverage-v1'
            or payload.get('source_code_sha256') != fingerprints
            or payload.get('coverage_contract') != contract
            or payload.get('story_source_sha256') != source):
        raise ValueError('Incompatible coverage checkpoint')
