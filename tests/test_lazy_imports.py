"""vtk, matplotlib and pandas stay out of a plain build-and-trace.

``import optiland.optic`` used to pull in vtk, matplotlib.pyplot and pandas
transitively -- through the visualization package's eager ``__init__.py``
re-exports and, for pandas, through the glass-catalog machinery imported at
module level -- even when nothing is ever drawn and no catalog glass is
resolved. Those imports now live inside the functions that actually draw
(or, for pandas, the functions that actually touch the glass catalog or
format a table for printing), so a caller who only builds and traces a
system never pays for them.

Both tests run in a fresh subprocess: pytest's own collection (this file's
neighbours import matplotlib and vtk directly) already populates
``sys.modules`` for the current process, so only a clean interpreter can
show what a first-time caller actually loads.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# A sequential lens and a small non-sequential scene, both built with
# IdealMaterial (a plain refractive index) rather than a named catalog glass
# such as "N-BK7". Resolving a catalog name is a real, non-formatting use of
# pandas (optiland.materials.registry reads catalog_nk.csv) that is out of
# this fix's scope; using IdealMaterial isolates the avoidable imports this
# fix targets from that legitimate, still lazily-loaded dependency.
_BUILD_AND_TRACE = """
import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.optic import Optic

lens = Optic()
lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
lens.surfaces.add(
    index=1, thickness=7, radius=43.7354, is_stop=True,
    material=IdealMaterial(n=1.67, k=0.0),
)
lens.surfaces.add(index=2, radius=-46.2795, thickness=50)
lens.surfaces.add(index=3)
lens.set_aperture(aperture_type="EPD", value=25)
lens.fields.set_type(field_type="angle")
lens.fields.add(y=0)
lens.wavelengths.add(value=0.5, is_primary=True)
lens.trace(0.0, 0.0, 0.5, num_rays=16)

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    IrradianceDetectorConfig,
    LensConfig,
    NSQMaterial,
    NSQScene,
    PointSourceConfig,
    Spectrum,
)

spec = Spectrum.monochromatic(0.55)
scene = NSQScene()
scene.add_source(
    "S1",
    CoordinateSystem(z=-100),
    PointSourceConfig(spectrum=spec, total_flux=1.0, half_angle_deg=6.0),
)
scene.add_lens(
    "L1",
    CoordinateSystem(z=0),
    LensConfig(
        r1=50, r2=-50, thickness=5,
        material=NSQMaterial(optiland_material=IdealMaterial(n=1.52, k=0.0)),
        front_aperture_radius=12.5,
    ),
)
scene.add_detector(
    "D1",
    CoordinateSystem(z=92),
    IrradianceDetectorConfig(width=4, height=4, num_pixels_x=32, num_pixels_y=32),
)
scene.trace(num_rays=2000, seed=42)
"""

_ASSERT_ABSENT = """
import sys
assert "vtk" not in sys.modules, sorted(m for m in sys.modules if "vtk" in m)
assert "matplotlib.pyplot" not in sys.modules
assert "pandas" not in sys.modules
"""


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_build_and_trace_do_not_import_vtk_matplotlib_pandas():
    """Building and tracing both engines never draws and never resolves a
    catalog glass, so it must never load vtk, matplotlib.pyplot or pandas."""
    result = _run(_BUILD_AND_TRACE + _ASSERT_ABSENT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_viewers_still_work_and_import_their_dependencies_on_demand():
    """The build above stays free of vtk/matplotlib/pandas; each viewer --
    sequential and non-sequential, 2D and 3D -- still draws correctly once
    actually called, and only then pulls in its dependency."""
    script = (
        _BUILD_AND_TRACE
        + _ASSERT_ABSENT
        + """
import matplotlib
matplotlib.use("Agg")

from optiland.visualization import (
    LensInfoViewer,
    OpticViewer,
    OpticViewer3D,
    SurfaceSagViewer,
)

fig, ax, _ = OpticViewer(lens).view(show=False)
assert fig is not None and ax is not None

# OpticViewer3D.__init__ itself needs vtk (it owns a render window), but
# neither the build above nor constructing/using the other viewers does.
viewer_3d = OpticViewer3D(lens)
viewer_3d.iren.Start = lambda *a, **kw: None
viewer_3d.ren_win.Render = lambda *a, **kw: None
viewer_3d.view()

LensInfoViewer(lens).view()

sag_fig, sag_axes = SurfaceSagViewer(lens).view(surface_index=1)
assert sag_fig is not None

from optiland.nonsequential.visualization import NSQViewer2D, NSQViewer3D

fig2, ax2 = NSQViewer2D(scene).view(num_rays=20)
assert fig2 is not None

# NSQViewer3D builds its own window/interactor inside view() rather than
# storing them on self, so silence the same two calls by patching the
# shared helper that creates them (BaseViewer3D._make_window) instead of an
# instance attribute.
from optiland.visualization.base import BaseViewer3D

_orig_make_window = BaseViewer3D._make_window

def _silent_make_window(self, *a, **kw):
    window, interactor = _orig_make_window(self, *a, **kw)
    window.Render = lambda *a, **kw: None
    interactor.Start = lambda *a, **kw: None
    return window, interactor

BaseViewer3D._make_window = _silent_make_window
NSQViewer3D(scene).view(num_rays=20)

assert "vtk" in sys.modules
assert "matplotlib.pyplot" in sys.modules
assert "pandas" in sys.modules
"""
    )
    result = _run(script)
    assert result.returncode == 0, result.stdout + result.stderr
