"""Declarative OSLO configuration data and independent sequential snapshots."""

from __future__ import annotations

import math
import re
from copy import deepcopy

from optiland.fileio.oslo.model import OsloConfiguration, OsloDataModel
from optiland.fileio.oslo.reader.solves import SOLVES


def _configuration(data: OsloDataModel, token: str, *, minimum=1):
    index = int(token)
    if not minimum <= index <= 1000:
        raise ValueError(f"configuration index must be between {minimum} and 1000")
    # Declaring configuration 3 also declares unchanged configuration 2.
    for key in range(1, index + 1):
        data.configurations.setdefault(key, OsloConfiguration())
    return data.configurations[index]


def read_configuration_record(data: OsloDataModel, tokens: list[str]) -> bool:
    """Read supported table or metadata records; return False for unknown items."""
    command = tokens[0]
    if command == "TH":
        if len(tokens) != 4:
            raise ValueError("configuration TH requires surface, configuration, value")
        surface = int(tokens[1])
        if not 0 <= surface <= data.num_surfaces:
            raise ValueError("configuration TH surface is outside the declared lens")
        configuration = _configuration(data, tokens[2], minimum=2)
        target, key = configuration.thicknesses, surface
    elif re.fullmatch(r"W[VW][1-9]\d*", command):
        if len(tokens) != 3:
            raise ValueError(
                f"configuration {command} requires configuration and value"
            )
        configuration = _configuration(data, tokens[1], minimum=2)
        key = int(command[2:])
        if key > 1001:
            raise ValueError("configuration wavelength index exceeds 1001")
        target = (
            configuration.wavelengths
            if command.startswith("WV")
            else configuration.wavelength_weights
        )
    elif command in {"CFWT", "CFAC"}:
        if len(tokens) != 3:
            raise ValueError(f"{command} requires configuration and value")
        configuration = _configuration(data, tokens[1])
        if command == "CFAC":
            if tokens[2].upper() not in {"YES", "NO"}:
                raise ValueError("CFAC expects YES or NO")
            configuration.active = tokens[2].upper() == "YES"
            return True
        value = float(tokens[2])
        if not math.isfinite(value) or value < 0:
            raise ValueError("CFWT weight must be finite and nonnegative")
        configuration.weight = value
        return True
    else:
        return False
    value = float(tokens[-1])
    if not math.isfinite(value):
        raise ValueError(f"configuration {command} requires a finite value")
    if command.startswith("WV") and value <= 0:
        raise ValueError("configuration wavelengths must be positive")
    if command.startswith("WW") and value < 0:
        raise ValueError("configuration wavelength weights must be nonnegative")
    target[key] = value
    return True


def configuration_spectrum(data: OsloDataModel, configuration: OsloConfiguration):
    """Validate and assemble a configuration's complete indexed spectrum."""
    spectrum = deepcopy(data.wavelengths)
    values, weights = spectrum["values"], spectrum["weights"]
    for index, value in sorted(configuration.wavelengths.items()):
        if index > len(values) + 1:
            raise ValueError("configuration leaves undefined wavelength slots")
        if index == len(values) + 1:
            values.append(value)
            weights.append(1.0)
        else:
            values[index - 1] = value
    for index, weight in configuration.wavelength_weights.items():
        if index > len(values):
            raise ValueError("configuration weight references an undefined wavelength")
        weights[index - 1] = weight
    if not weights[0]:
        raise ValueError("configuration primary wavelength weight must be positive")
    return spectrum


def select_configuration(data: OsloDataModel, index: int) -> OsloDataModel:
    """Copy the base and apply overrides before coordinates or pickups are resolved.

    Alternate solve execution depends on CSLV, which is outside this subset.
    Literal overrides of pickup-controlled thicknesses are also rejected until
    their precedence has an independently verified mapping.
    """
    if type(index) is not int or index not in data.configurations:
        raise ValueError(f"OSLO configuration {index!r} is not defined")
    selected = deepcopy(data)
    configuration = data.configurations[index]
    if index != 1 and any(SOLVES.keys() & s.keys() for s in data.surfaces.values()):
        raise ValueError(
            "OSLO alternate configurations with solves require CSLV support"
        )
    for surface, thickness in configuration.thicknesses.items():
        target = selected.surfaces[surface]
        if any(
            pickup[0].upper() in {"TH", "THM", "LN", "LNM"}
            for pickup in target.get("pickups", [])
        ):
            raise ValueError(
                "configuration TH cannot override a pickup-controlled thickness"
            )
        target["TH"] = thickness
    selected.wavelengths = configuration_spectrum(data, configuration)
    return selected
