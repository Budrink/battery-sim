"""Explicit selection of observations used by the current UPV calibration.

Raw channels, mapping and simulation remain intact. C2/ISS is informational
because its post-burst record has disputed continuity (user review 2026-10-22).
"""
import upv_r123_data as dataio

VERSION = 'five-temperature-channels-and-pressure-v4'
INFORMATIONAL_CHANNELS = ('C2_ISS',)
CALIBRATION_CHANNELS = tuple(k for k in dataio.CHANNELS if k not in INFORMATIONAL_CHANNELS)


def metadata():
    return dict(version=VERSION, calibration_channels=list(CALIBRATION_CHANNELS),
                informational_channels=list(INFORMATIONAL_CHANNELS),
                reason='C2/ISS excluded by user: disputed continuity after rupture; raw data retained')


def label(key, title):
    return title+' (справочно, вне подбора)' if key in INFORMATIONAL_CHANNELS else title
