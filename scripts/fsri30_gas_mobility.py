"""Diagnostic multiplier for internal aligned gas links, not vents or ruptures.

The baseline factor belongs to the scenario; this dimensionless multiplier is
an unmeasured fixture-specific hypothesis, not a viscosity measurement.
"""
from copy import deepcopy
import math


def with_internal_resistance(source, multiplier):
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Internal gas resistance multiplier must be positive and finite')
    scene = deepcopy(source)
    if 'internal_gas_resistance_probe' in scene['metadata']:
        raise ValueError('Apply this diagnostic once to the original scenario')
    changed = 0
    for edge in scene['gas_network']['channels']:
        if (edge.get('kind') == 'passage' and
                edge['from'].startswith('aligned-inner-') and
                edge['to'].startswith('aligned-inner-')):
            edge['hydraulic_resistance_factor'] = edge.get('hydraulic_resistance_factor', 1.) * multiplier
            changed += 1
    if not changed:
        raise ValueError('No internal aligned gas passages found')
    scene['metadata']['internal_gas_resistance_probe'] = {
        'multiplier': multiplier, 'channels': changed,
        'qualification': 'unmeasured fixture-specific effective resistance; not calibrated viscosity',
        'scope': 'aligned internal gas passages only; vent, shell and thermal contacts unchanged',
    }
    return scene
