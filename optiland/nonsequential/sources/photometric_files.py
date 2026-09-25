"""Measured photometric files: IES LM-63 (Type C) and EULUMDAT.

A file holds luminous intensity (cd) on a grid of vertical angles ``gamma``
from the nadir (0) to the zenith (180 degrees) and of C-planes, the azimuth
about the vertical axis, with ``C = 0`` along the luminaire's length and C
increasing counterclockwise seen from above. :class:`PhotometricTable` holds
that grid expanded to the full 0-360 degree range; :meth:`PhotometricTable.to_source_config`
turns it into a :class:`~optiland.nonsequential.sources.configs.TabulatedSourceConfig`
in the **luminaire frame**: local +z up (toward the zenith), local +x along
``C = 0``, local +y along ``C = 90``, so ``theta = 180 - gamma`` and
``phi = C``, and a downlight emits toward local -z.

The IES reader is this module's own: the only permissively licensed Python
reader found (``eulumdat-ies`` 1.1.0, MIT) returns a quadrant- or
half-symmetric file as an incomplete matrix and replaces absolute photometry by
a synthetic 1000 lm, so it cannot be the reading path. The parsing follows
Blender Cycles' reader (``intern/cycles/util/ies.cpp``, Apache-2.0): commas as
separators, the candela multiplied by the candela multiplier, the ballast factor
and the ballast-lamp photometric factor. The lateral symmetry is expanded by
mirroring (a 90-270 degree file about its own plane, where that reader rotates
it instead). Tilt data and photometry types A and B are refused by name.

EULUMDAT files are read through the optional package ``eulumdat-py`` (import
name ``pyldt``, MIT), which expands the file's symmetry index to the full
C-plane matrix; the intensities, in cd per 1000 lm of lamp flux, are scaled by
the file's conversion factor and the first lamp set's flux.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass
class PhotometricTable:
    """Luminous intensity on (C-plane, vertical angle), expanded to 0-360 degrees.

    Attributes:
        c_angles_deg: C-plane angles [deg], increasing from 0 to 360 (the rows
            at 0 and 360 are the same plane).
        gamma_angles_deg: Vertical angles [deg] from the nadir, increasing.
        candela: Intensity [cd], shape (len(c_angles_deg), len(gamma_angles_deg)).
        file_format: What the table was read from.
        keywords: The file's keyword lines (IES) or header fields (EULUMDAT).
        luminous_opening_mm: (width, length, height) of the luminous opening
            [mm] as the file states it; a negative width is a circle's diameter.
    """

    c_angles_deg: np.ndarray
    gamma_angles_deg: np.ndarray
    candela: np.ndarray
    file_format: str = ""
    keywords: dict = field(default_factory=dict)
    luminous_opening_mm: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        self.c_angles_deg = np.asarray(self.c_angles_deg, dtype=np.float64)
        self.gamma_angles_deg = np.asarray(self.gamma_angles_deg, dtype=np.float64)
        self.candela = np.asarray(self.candela, dtype=np.float64)
        if self.candela.shape != (self.c_angles_deg.size, self.gamma_angles_deg.size):
            raise ValueError("PhotometricTable: candela must be (n_C, n_gamma).")
        if self.c_angles_deg[0] != 0.0 or self.c_angles_deg[-1] != 360.0:
            raise ValueError("PhotometricTable: C-planes must run from 0 to 360 degrees.")

    @property
    def rotationally_symmetric(self) -> bool:
        """True when every C-plane holds the same intensities."""
        return bool(np.all(self.candela == self.candela[0]))

    def luminous_flux(self) -> float:
        """The table's luminous flux [lm]: ``integral I dOmega`` of its interpolant."""
        from optiland.nonsequential.sources.tabulated import (  # noqa: PLC0415
            cell_integrals,
        )

        theta = np.radians(180.0 - self.gamma_angles_deg[::-1])
        rows = cell_integrals(theta, self.candela[:, ::-1]).sum(axis=1)
        return float(np.sum(0.5 * (rows[:-1] + rows[1:]) * np.diff(np.radians(self.c_angles_deg))))

    def to_source_config(
        self,
        spectrum,
        width: float | None = None,
        height: float | None = None,
        aperture_radius: float | None = None,
        medium=None,
    ):
        """A tabulated source in the luminaire frame (+z up, +x along C = 0).

        The intensities are absolute candela, so the source's flux is the
        table's luminous flux converted to watts with ``spectrum``.

        Args:
            spectrum: The source's spectrum (its efficacy converts lm to W).
            width, height, aperture_radius: An emitting area in the local x-y
                plane [mm]; none: a point.
            medium: The medium the source sits in.

        Returns:
            A :class:`TabulatedSourceConfig`.
        """
        from optiland.nonsequential.sources.configs import (  # noqa: PLC0415
            TabulatedSourceConfig,
        )

        theta = 180.0 - self.gamma_angles_deg[::-1]
        table = self.candela[:, ::-1]
        if self.rotationally_symmetric:
            return TabulatedSourceConfig(
                spectrum=spectrum, polar_angles_deg=theta, intensity=table[0],
                intensity_units="cd", width=width, height=height,
                aperture_radius=aperture_radius, medium=medium,
            )
        return TabulatedSourceConfig(
            spectrum=spectrum, polar_angles_deg=theta, intensity=table,
            azimuth_angles_deg=self.c_angles_deg, intensity_units="cd", width=width,
            height=height, aperture_radius=aperture_radius, medium=medium,
        )


# ---------------------------------------------------------------------------
# Lateral symmetry
# ---------------------------------------------------------------------------


def _close(a: float, b: float) -> bool:
    return abs(a - b) < 1e-6


def expand_lateral_symmetry(c_angles, rows):
    """Expand a Type C file's C-planes to 0-360 degrees by mirroring.

    ``[0]``: rotational symmetry; ``0..90``: symmetric in each quadrant,
    ``I(C) = I(180 - C)`` then ``I(C) = I(360 - C)``; ``0..180``: symmetric
    about the 0-180 plane, ``I(C) = I(360 - C)``; ``90..270``: symmetric about
    the 90-270 plane, ``I(C) = I(180 - C)`` (mod 360); ``0..last`` with
    ``last > 180``: no symmetry, and a missing 360 plane is the 0 plane.

    Args:
        c_angles: The file's C angles [deg], increasing.
        rows: One intensity row per C angle.

    Returns:
        ``(c_full, rows_full)`` from 0 to 360 inclusive.
    """
    c = [float(a) for a in c_angles]
    r = [np.asarray(x, dtype=np.float64) for x in rows]
    if len(c) == 1:
        return np.array([0.0, 360.0]), np.stack([r[0], r[0]])
    if _close(c[0], 90.0) and _close(c[-1], 270.0):
        full_c, full_r = [], []
        for ang, row in zip(c, r, strict=True):  # the given 90..270
            full_c.append(ang)
            full_r.append(row)
        for ang, row in zip(c, r, strict=True):  # mirror: C' = 180 - C (mod 360)
            m = (180.0 - ang) % 360.0
            if not (90.0 <= m <= 270.0) or _close(m, 90.0) or _close(m, 270.0):
                full_c.append(m)
                full_r.append(row)
        order = np.argsort(full_c)
        c_sorted = [full_c[i] for i in order]
        r_sorted = [full_r[i] for i in order]
        # the mirror of 90 is 90 and of 270 is 270; C = 0 comes from 180
        c_out, r_out = [], []
        for ang, row in zip(c_sorted, r_sorted, strict=True):
            if c_out and _close(ang, c_out[-1]):
                continue
            c_out.append(ang)
            r_out.append(row)
        if not _close(c_out[0], 0.0):
            raise ValueError("a 90-270 degree file needs its 180 degree plane")
        c_out.append(360.0)
        r_out.append(r_out[0])
        return np.array(c_out), np.stack(r_out)
    if not _close(c[0], 0.0):
        raise ValueError(
            f"unsupported C-plane range {c[0]:g}-{c[-1]:g} degrees: Type C files start at "
            "0 (or run from 90 to 270)"
        )
    if _close(c[-1], 90.0):
        c = c + [180.0 - a for a in reversed(c[:-1])]
        r = r + list(reversed(r[:-1]))
    if _close(c[-1], 180.0):
        c = c + [360.0 - a for a in reversed(c[:-1])]
        r = r + list(reversed(r[:-1]))
    if not _close(c[-1], 360.0):
        c = c + [360.0]
        r = r + [r[0]]
    return np.array(c), np.stack(r)


# ---------------------------------------------------------------------------
# IES LM-63
# ---------------------------------------------------------------------------


def read_ies(source) -> PhotometricTable:
    """Read an IES LM-63 file (1991, 1995, 2002, 2019; photometry Type C).

    Args:
        source: A path, or the file's text.

    Returns:
        The table, expanded to 0-360 degrees, in absolute candela.

    Raises:
        NotImplementedError: For tilt data or photometry types A and B.
        ValueError: For a malformed file.
    """
    text = _read_text(source)
    lines = text.splitlines()
    keywords: dict[str, str] = {}
    header = lines[0].strip() if lines else ""
    tilt_index = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.upper().startswith("TILT"):
            tilt_index = i
            break
        m = re.match(r"\[(.*?)\]\s*(.*)", s)
        if m:
            keywords[m.group(1).strip()] = m.group(2).strip()
    if tilt_index is None:
        raise ValueError("IES file: no TILT line")
    tilt = lines[tilt_index].split("=", 1)[1].strip().upper() if "=" in lines[tilt_index] else ""
    if tilt != "NONE":
        raise NotImplementedError(
            f"IES file: TILT={tilt} carries lamp-tilt multipliers that this reader does not "
            "apply; ignoring them would change the intensities silently"
        )
    tokens = _NUMBER.findall(" ".join(lines[tilt_index + 1:]).replace(",", " "))
    values = [float(t) for t in tokens]
    if len(values) < 13:
        raise ValueError("IES file: the two numeric header lines are incomplete")
    (_n_lamps, lamp_lumens, multiplier, n_v, n_h, ptype, units, width, length, height,
     ballast, blp_factor, _watts) = values[:13]
    n_v, n_h, ptype, units = int(n_v), int(n_h), int(ptype), int(units)
    if ptype != 1:
        raise NotImplementedError(
            f"IES file: photometry type {'B' if ptype == 2 else 'A' if ptype == 3 else ptype} "
            "is not supported; only Type C (1) is"
        )
    need = 13 + n_v + n_h + n_v * n_h
    if len(values) < need:
        raise ValueError(f"IES file: {len(values)} numbers where {need} are needed")
    pos = 13
    gamma = np.array(values[pos:pos + n_v])
    pos += n_v
    c = np.array(values[pos:pos + n_h])
    pos += n_h
    cd = np.array(values[pos:pos + n_v * n_h]).reshape(n_h, n_v)
    cd = cd * multiplier * ballast * blp_factor
    if not np.all(np.diff(gamma) > 0) or gamma[0] < 0 or gamma[-1] > 180:
        raise ValueError("IES file: vertical angles must increase within 0-180 degrees")
    if n_h > 1 and not np.all(np.diff(c) > 0):
        raise ValueError("IES file: horizontal angles must increase")
    c_full, rows = expand_lateral_symmetry(c, cd)
    scale = 304.8 if units == 1 else 1000.0  # feet or metres to mm
    keywords["_header"] = header
    keywords["_lamp_lumens"] = lamp_lumens
    return PhotometricTable(
        c_angles_deg=c_full,
        gamma_angles_deg=gamma,
        candela=rows,
        file_format=header or "IES LM-63",
        keywords=keywords,
        luminous_opening_mm=(width * scale, length * scale, height * scale),
    )


def write_ies(table: PhotometricTable, path=None, keywords: dict | None = None) -> str:
    """Write a table as an IES LM-63-2002 Type C file with absolute photometry.

    Every C-plane of the table is written (no symmetry is claimed), values in
    full double precision, so reading the file back returns the same table.

    Args:
        table: The table.
        path: Where to write; ``None`` returns the text only.
        keywords: Keyword lines, e.g. ``{"TEST": "...", "MANUFAC": "..."}``.

    Returns:
        The file's text.
    """
    kw = {"TEST": "", "MANUFAC": "", "LUMINAIRE": "", "LAMP": ""}
    kw.update(keywords or {})
    out = ["IESNA:LM-63-2002"]
    out += [f"[{k}] {v}".rstrip() for k, v in kw.items() if not k.startswith("_")]
    out.append("TILT=NONE")
    w, length, h = (v / 1000.0 for v in table.luminous_opening_mm)
    out.append(
        f"1 -1 1 {table.gamma_angles_deg.size} {table.c_angles_deg.size} 1 2 "
        f"{w:.17g} {length:.17g} {h:.17g}"
    )
    out.append("1 1 0")
    out += _wrap(table.gamma_angles_deg)
    out += _wrap(table.c_angles_deg)
    for row in table.candela:
        out += _wrap(row)
    text = "\n".join(out) + "\n"
    if path is not None:
        Path(path).write_text(text, encoding="ascii")
    return text


def _wrap(values, per_line: int = 8) -> list[str]:
    vals = [f"{float(v):.17g}" for v in values]
    return [" ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]


def _read_text(source) -> str:
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source
                                    and Path(source).exists()):
        return Path(source).read_text(encoding="ascii", errors="replace").lstrip("﻿")
    return str(source).lstrip("﻿")


# ---------------------------------------------------------------------------
# EULUMDAT, through the optional reader
# ---------------------------------------------------------------------------


def read_eulumdat(path) -> PhotometricTable:
    """Read an EULUMDAT (.ldt) file through the optional ``eulumdat-py`` package.

    Intensities in the file are cd per 1000 lm of lamp flux; they are scaled by
    the file's conversion factor and the first lamp set's flux (number of lamps
    times lamp flux). ``pyldt`` expands the symmetry index to the full matrix.

    Raises:
        ImportError: If ``eulumdat-py`` is not installed.
    """
    try:
        from pyldt import LdtReader  # noqa: PLC0415
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError(
            "EULUMDAT files are read with the optional package 'eulumdat-py' (MIT, import "
            "name pyldt); install it, or convert the file to IES."
        ) from err
    ldt = LdtReader.read(str(path), expand_symmetry=True)
    h = ldt.header
    flux = abs(h.num_lamps[0]) * h.lamp_flux[0] if h.num_lamps and h.lamp_flux else 1000.0
    cd = np.asarray(ldt.intensities, dtype=np.float64) * h.conv_factor * flux / 1000.0
    c = np.asarray(h.c_angles, dtype=np.float64)
    if c.size != cd.shape[0]:
        c = np.arange(cd.shape[0]) * h.dc
    if not _close(c[-1], 360.0):
        c = np.append(c, 360.0)
        cd = np.vstack([cd, cd[:1]])
    return PhotometricTable(
        c_angles_deg=c,
        gamma_angles_deg=np.asarray(h.g_angles, dtype=np.float64),
        candela=cd,
        file_format="EULUMDAT",
        keywords={"luminaire_name": h.luminaire_name, "isym": h.isym, "company": h.company},
        luminous_opening_mm=(h.width_lum_area, h.length_lum_area, 0.0),
    )
