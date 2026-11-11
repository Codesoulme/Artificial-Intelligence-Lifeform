"""Bounded novelty features for auditable observation decisions."""
import math


def observation_surprise(*, known, old_value, new_value, old_quality, new_quality,
                         old_time, new_time):
    """Return novelty and its components without deciding whether to write."""
    if type(known) is not bool:
        raise ValueError('known must be boolean')
    if type(old_value) is not int or type(new_value) is not int:
        raise ValueError('Values must be integers')
    for name, value in (('old_quality', old_quality), ('new_quality', new_quality)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be finite and in [0, 1]')
    if type(old_time) is not int or type(new_time) is not int or old_time < -1 or new_time < 0:
        raise ValueError('Times must be integers; old_time may be -1')
    components = {
        'new_fact': float(not known),
        'value_change': float(known and old_value != new_value),
        'quality_gain': max(0.0, float(new_quality) - float(old_quality)),
        'time_change': float(old_time >= 0 and old_time != new_time),
    }
    novelty = max(components['new_fact'], components['value_change'],
                  0.5 * components['quality_gain'], 0.25 * components['time_change'])
    return {'score': min(1.0, novelty), 'components': components}
