"""Where the absorbed power went: per-component books and deposition tallies.

The bounce loop attenuates every ray through the medium it travels in
(Beer-Lambert) and every lossy surface books what it removes. Summed over
the scene those give two numbers, ``total_flux_bulk_absorbed`` and
``total_flux_coating``. A thermal model of one part needs more: the power
THAT part absorbed, and where inside it the power was deposited. This module
keeps both.

Which component a bulk loss belongs to
--------------------------------------
A ray inside a solid travels in that solid's interior medium until it
reaches the solid's boundary; the boundary is the first surface it can hit
from inside. So the loss of the segment a ray has just travelled is booked
to the component that owns the surface ENDING the segment -- when the ray
was inside a medium (medium stack depth above zero). A segment travelled in
the ambient medium (depth zero) is booked to ``"ambient"``, and a segment
that ends on a detector rather than a surface to ``"unassigned"``. The rule
is exact for closed solids that do not touch; two solids that share a face
(a cemented pair registered as two components) can see the neighbour's
coincident face first, which is why a cemented pair is one component
(``add_doublet``).

The deposition tally
--------------------
A :class:`DepositionTally` is a grid over one named component: a Cartesian
``xyz`` grid, or an ``rz`` grid about an axis (a lens or a window). Every
segment whose loss is booked to that component is split into sub-steps no
longer than half the grid's smallest pitch, and each sub-step receives its
exact Beer-Lambert share ``F0 exp(-a s0) (1 - exp(-a ds))``, placed at the
sub-step's midpoint. The shares of one segment sum to ``F0 (1 - exp(-a L))``,
the loss the ledger books, so the tally's total plus what fell outside the
grid equals the component's bulk book to rounding. The map is power per
bin (W) and power per unit volume (W/mm^3, divided by each bin's volume:
``pi (r1^2 - r0^2) dz`` for ``rz``).

The tally reads each bounce's ray state on the host; on a device backend
that is one device-to-host copy per bounce while a tally is registered (and
none otherwise). The per-component books stay where the ray state lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import optiland.backend as be
from optiland.backend.utils import to_numpy

AMBIENT = "ambient"
UNASSIGNED = "unassigned"


@dataclass
class DepositionTally:
    """A grid over one component that collects the bulk power it absorbs.

    Attributes:
        name: Key of the map in ``SimulationResult.deposition``.
        component: Registry name of the component whose bulk loss is tallied.
        kind: ``"xyz"`` (a Cartesian grid in the scene's frame) or ``"rz"``
            (radius about ``axis`` through ``origin``, and the coordinate
            along ``axis`` measured from ``origin``).
        bounds: ``xyz``: ``(x0, x1, y0, y1, z0, z1)`` in mm. ``rz``:
            ``(r0, r1, z0, z1)`` in mm, ``r0 >= 0``.
        shape: Bins per axis: ``(nx, ny, nz)`` or ``(nr, nz)``.
        origin: A point on the axis of an ``rz`` grid, mm (scene frame).
        axis: The axis direction of an ``rz`` grid (normalised here).
    """

    name: str
    component: str
    kind: str = "rz"
    bounds: tuple = ()
    shape: tuple = ()
    origin: tuple = (0.0, 0.0, 0.0)
    axis: tuple = (0.0, 0.0, 1.0)
    _power: np.ndarray = field(default=None, init=False, repr=False)
    _outside: float = field(default=0.0, init=False, repr=False)
    _booked: float = field(default=0.0, init=False, repr=False)
    _segments: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in ("xyz", "rz"):
            raise ValueError(f"deposition tally {self.name!r}: kind must be 'xyz' or 'rz', not {self.kind!r}")
        dims = 3 if self.kind == "xyz" else 2
        self.bounds = tuple(float(b) for b in self.bounds)
        self.shape = tuple(int(n) for n in self.shape)
        if len(self.bounds) != 2 * dims or len(self.shape) != dims:
            raise ValueError(
                f"deposition tally {self.name!r}: a {self.kind} grid takes {2 * dims} bounds and {dims} bin counts, "
                f"got {len(self.bounds)} and {len(self.shape)}"
            )
        for i in range(dims):
            if not self.bounds[2 * i + 1] > self.bounds[2 * i]:
                raise ValueError(f"deposition tally {self.name!r}: bound {i} is empty or reversed")
            if self.shape[i] < 1:
                raise ValueError(f"deposition tally {self.name!r}: every axis needs at least one bin")
        if self.kind == "rz" and self.bounds[0] < 0.0:
            raise ValueError(f"deposition tally {self.name!r}: r0 must be >= 0")
        ax = np.asarray(self.axis, dtype=float)
        norm = float(np.linalg.norm(ax))
        if not norm > 0.0:
            raise ValueError(f"deposition tally {self.name!r}: the axis has zero length")
        self.axis = tuple(float(a) for a in ax / norm)
        self.origin = tuple(float(o) for o in self.origin)
        self.reset()

    # ------------------------------------------------------------------ geometry
    @property
    def edges(self) -> list[np.ndarray]:
        """The bin edges per axis, mm."""
        return [np.linspace(self.bounds[2 * i], self.bounds[2 * i + 1], n + 1) for i, n in enumerate(self.shape)]

    @property
    def centres(self) -> list[np.ndarray]:
        """The bin centres per axis, mm."""
        return [0.5 * (e[1:] + e[:-1]) for e in self.edges]

    @property
    def pitch(self) -> float:
        """The smallest bin width, mm."""
        return min((self.bounds[2 * i + 1] - self.bounds[2 * i]) / n for i, n in enumerate(self.shape))

    def bin_volumes(self) -> np.ndarray:
        """Each bin's volume, mm^3, shaped like the map."""
        e = self.edges
        if self.kind == "xyz":
            return np.einsum("i,j,k->ijk", np.diff(e[0]), np.diff(e[1]), np.diff(e[2]))
        ring = np.pi * (e[0][1:] ** 2 - e[0][:-1] ** 2)
        return np.outer(ring, np.diff(e[1]))

    def _coords(self, p: np.ndarray) -> np.ndarray:
        """Scene points (n, 3) in the grid's own coordinates (n, dims)."""
        if self.kind == "xyz":
            return p
        d = p - np.asarray(self.origin)
        a = np.asarray(self.axis)
        z = d @ a
        radial = d - z[:, None] * a[None, :]
        r = np.sqrt(np.einsum("ij,ij->i", radial, radial))
        return np.column_stack((r, z))

    def _flat_index(self, q: np.ndarray) -> np.ndarray:
        """Flat bin index of each point, -1 outside the grid."""
        idx = np.zeros(q.shape[0], dtype=np.int64)
        inside = np.ones(q.shape[0], dtype=bool)
        stride = 1
        for i in reversed(range(len(self.shape))):
            lo, hi, n = self.bounds[2 * i], self.bounds[2 * i + 1], self.shape[i]
            k = np.floor((q[:, i] - lo) / (hi - lo) * n).astype(np.int64)
            inside &= (q[:, i] >= lo) & (q[:, i] <= hi)
            k = np.clip(k, 0, n - 1)
            idx += k * stride
            stride *= n
        return np.where(inside, idx, -1)

    # ------------------------------------------------------------------ accumulation
    def reset(self) -> None:
        self._power = np.zeros(int(np.prod(self.shape)), dtype=np.float64)
        self._outside = 0.0
        self._booked = 0.0
        self._segments = 0

    def deposit(self, start, direction, length, flux_before, attenuation) -> None:
        """Split each segment into sub-steps and deposit its exact Beer-Lambert shares.

        Args:
            start: (n, 3) segment start points, mm.
            direction: (n, 3) unit directions.
            length: (n,) segment lengths, mm.
            flux_before: (n,) flux entering the segment, W.
            attenuation: (n,) Beer-Lambert coefficient, 1/mm.
        """
        n = int(length.shape[0])
        if n == 0:
            return
        ds_max = 0.5 * self.pitch
        nsub = np.maximum(1, np.ceil(length / ds_max)).astype(np.int64)
        ds = length / nsub
        seg = np.repeat(np.arange(n), nsub)
        # the sub-step's position along its segment: 0, 1, ... nsub-1
        first = np.cumsum(nsub) - nsub
        j = np.arange(seg.size) - np.repeat(first, nsub)
        s0 = j * ds[seg]
        a = attenuation[seg]
        share = flux_before[seg] * np.exp(-a * s0) * (-np.expm1(-a * ds[seg]))
        mid = start[seg] + (s0 + 0.5 * ds[seg])[:, None] * direction[seg]
        k = self._flat_index(self._coords(mid))
        inside = k >= 0
        self._power += np.bincount(k[inside], weights=share[inside], minlength=self._power.size)
        self._outside += float(share[~inside].sum())
        self._booked += float((flux_before * (-np.expm1(-attenuation * length))).sum())
        self._segments += n

    # ------------------------------------------------------------------ the result
    def result(self) -> DepositionMap:
        power = self._power.reshape(self.shape).copy()
        return DepositionMap(
            name=self.name,
            component=self.component,
            kind=self.kind,
            edges=self.edges,
            power=power,
            density=power / self.bin_volumes(),
            outside=self._outside,
            booked=self._booked,
            segments=self._segments,
            origin=self.origin,
            axis=self.axis,
        )


@dataclass
class DepositionMap:
    """One tally's result.

    Attributes:
        name: The tally's name.
        component: The component tallied.
        kind: ``"xyz"`` or ``"rz"``.
        edges: Bin edges per axis, mm.
        power: Absorbed power per bin, W, shaped like the grid.
        density: Absorbed power per unit volume, W/mm^3.
        outside: Power booked to the component but deposited outside the grid, W.
        booked: The component's bulk loss the tally saw, W; equals
            ``power.sum() + outside`` to rounding.
        segments: Number of ray segments deposited.
        origin: Axis point of an ``rz`` grid, mm.
        axis: Axis direction of an ``rz`` grid.
    """

    name: str
    component: str
    kind: str
    edges: list
    power: np.ndarray
    density: np.ndarray
    outside: float
    booked: float
    segments: int
    origin: tuple = (0.0, 0.0, 0.0)
    axis: tuple = (0.0, 0.0, 1.0)

    @property
    def centres(self) -> list[np.ndarray]:
        return [0.5 * (e[1:] + e[:-1]) for e in self.edges]

    @property
    def total(self) -> float:
        return float(self.power.sum())


class ComponentBook:
    """Bulk loss booked per surface of the scene, plus the ambient and unassigned bins.

    Surfaces are grouped into components afterwards
    (:func:`absorbed_by_component`). The per-surface vector lives on the device
    of the ray state, like every other tally of the loop.
    """

    def __init__(self, n_surfaces: int) -> None:
        self.n = int(n_surfaces)
        self._host = np.zeros(self.n + 2, dtype=np.float64)   # [surfaces..., ambient, unassigned]
        self._dev = None

    def add(self, loss, comp_idx, comp_first, det_first, depth) -> None:
        """Book this bounce's per-ray bulk loss.

        Args:
            loss: (n,) flux lost in the segment, W.
            comp_idx: (n,) index of the nearest surface hit, -1 for none.
            comp_first: (n,) the segment ends on a surface.
            det_first: (n,) the segment ends on a detector.
            depth: (n,) medium stack depth while travelling the segment.
        """
        ambient_bin, unassigned_bin = self.n, self.n + 1
        if be.is_torch_tensor(loss):
            import torch  # noqa: PLC0415

            inside = depth > 0
            idx = torch.where(comp_first & inside, comp_idx.to(torch.int64),
                              torch.full_like(comp_idx, unassigned_bin, dtype=torch.int64))
            idx = torch.where(~inside, torch.full_like(idx, ambient_bin), idx)
            used = comp_first | det_first
            w = torch.where(used, loss, torch.zeros_like(loss)).to(torch.float64)
            if self._dev is None:
                self._dev = torch.zeros(self.n + 2, dtype=torch.float64, device=loss.device)
            self._dev.index_add_(0, idx, w)
            return
        loss = np.asarray(loss, dtype=np.float64)
        comp_idx = np.asarray(comp_idx, dtype=np.int64)
        comp_first = np.asarray(comp_first, dtype=bool)
        det_first = np.asarray(det_first, dtype=bool)
        inside = np.asarray(depth) > 0
        idx = np.where(comp_first & inside, comp_idx, unassigned_bin)
        idx = np.where(~inside, ambient_bin, idx)
        used = comp_first | det_first
        self._host += np.bincount(idx[used], weights=loss[used], minlength=self.n + 2)

    def values(self) -> np.ndarray:
        if self._dev is None:
            return self._host
        return self._host + to_numpy(self._dev).astype(np.float64)


def surface_owners(scene) -> list[str]:
    """The registry name of the component that owns each flat scene surface."""
    owners: list[str] = []
    for name in scene.component_names:
        comp = scene.component_registry._registry[name]
        owners.extend([name] * len(comp.surfaces))
    return owners


def absorbed_by_component(scene, bulk_by_surface: np.ndarray) -> dict[str, dict[str, float]]:
    """Group the per-surface books into one record per component.

    Each record carries ``bulk`` (Beer-Lambert loss in the component's
    interior), ``coating`` (a lossy coating or a mirror below unit
    reflectance on its surfaces), ``absorber`` (whole rays absorbed by an
    absorbing surface) and ``total``, all in W. ``ambient`` and
    ``unassigned`` hold bulk loss travelled outside every component and
    bulk loss of segments that ended on a detector.
    """
    from optiland.nonsequential.components.absorbing import AbsorbingComponent  # noqa: PLC0415

    owners = surface_owners(scene)
    out: dict[str, dict[str, float]] = {
        name: {"bulk": 0.0, "coating": 0.0, "absorber": 0.0, "total": 0.0} for name in scene.component_names
    }
    for i, surf in enumerate(scene.surfaces):
        rec = out[owners[i]]
        rec["bulk"] += float(bulk_by_surface[i])
        if hasattr(surf, "coating_loss"):
            rec["coating"] += float(surf.coating_loss)
        if isinstance(surf, AbsorbingComponent):
            rec["absorber"] += float(to_numpy(surf._absorbed_flux))
    n = len(scene.surfaces)
    out[AMBIENT] = {"bulk": float(bulk_by_surface[n]), "coating": 0.0, "absorber": 0.0, "total": 0.0}
    out[UNASSIGNED] = {"bulk": float(bulk_by_surface[n + 1]), "coating": 0.0, "absorber": 0.0, "total": 0.0}
    for rec in out.values():
        rec["total"] = rec["bulk"] + rec["coating"] + rec["absorber"]
    return out
