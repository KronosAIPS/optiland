"""Documented conventions shared by the OSLO reader and writer."""

from __future__ import annotations

# OSLO Program Reference, printed p. 123, "Wavelength": initial values of the
# Wavelength_default (wvld) preference, in micrometers (PDF page 137).
# https://lambdares.com/hubfs/Support/support/oslo/oslo_releases/OSLOProgramReference.pdf#page=137
DEFAULT_WAVELENGTHS_UM = (0.58756, 0.48613, 0.65627)

# OSLO Optics Reference, printed p. 151, "Telecentric ray-aiming": object
# distances at or above 1e8 lens units are treated as infinite. This threshold
# differs from the large sentinel values commonly written in lens files.
# https://lambdares.com/hubfs/Support/support/OSLOOpticsReference_Sep21.pdf#page=151
OBJECT_INFINITY_THRESHOLD = 1e8

# Saved OSLO prescriptions use values near 1e10 for infinite non-object gaps
# (including rounded values such as 9.9999999996e9). Keep import and export
# consistent about which finite native spacings would become this sentinel.
THICKNESS_INFINITY_THRESHOLD = 9.9e9
