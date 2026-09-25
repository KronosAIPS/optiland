"""The Apple GPU (torch ``mps``) tier: two silent defects of the device, fenced.

Both were found by running one computation on the CPU and on the Apple GPU and
comparing the bits (torch 2.14.0; the Mac-tier study of 2026-09-24), and both
return a wrong number without an error:

* **The complex square root** (research repository issue 53). On ``mps`` the
  native root of an argument whose one part is small beside the other loses
  that part: ``sqrt(5 + 0.0005j)`` comes back ``2.2360680 + 0j``. The engine's
  thin-film coating takes the layer cosine as such a root (the argument's
  imaginary part is ``-2 n k``), so an absorbing layer's attenuation vanished
  at float32 on the Apple GPU. ``be.csqrt`` forms the root from real
  operations on a device where :attr:`exact_complex_sqrt` is false and is the
  native root, unchanged to the bit, everywhere else. ``TestComplexSqrt``.

* **The copy to a host float64 tensor** (research repository issue 54).
  ``cpu_f64.copy_(x_mps)``, ``x_mps.to("cpu", torch.float64)`` and
  ``torch.as_tensor(x_mps, dtype=torch.float64, device="cpu")`` all write
  zeros; moving to the CPU first and casting there is right. The same holds for
  complex128 and for any source dtype (float32, int, bool). Every conversion of
  the torch backend that can take a device tensor to a host dtype now moves
  before it casts. ``TestHostCopy``.

Tests that need the Apple GPU skip where it is not reachable (every CI host, and
a sandboxed session); each keeps a control that reproduces the defect with the
native operation, so an assertion cannot pass on a device that no longer has
the defect without the test saying so.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Torch not available -- these are torch-device tests")

# ruff: noqa: E402
import optiland.backend as be
from optiland.backend.torch_backend.capabilities import csqrt_from_reals
from optiland.materials import IdealMaterial
from optiland.thin_film import ThinFilmStack

_U32 = 2.0**-24
_U64 = 2.0**-53

needs_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="the Apple GPU (torch mps) is not reachable on this host",
)


@pytest.fixture
def torch_backend_state():
    """Leave ``optiland.backend`` as the test found it (numpy, cpu, float64, no grad)."""
    yield
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    be.grad_mode.disable()
    be.set_backend("numpy")
    be.set_precision("float64")


def _arguments(dtype, n=20_000, seed=0):
    """Complex arguments whose one part is small beside the other, both ways round."""
    g = torch.Generator().manual_seed(seed)
    big = torch.rand(n, generator=g, dtype=torch.float64) * 20.0 - 10.0
    small = (torch.rand(n, generator=g, dtype=torch.float64) * 2.0 - 1.0) * torch.pow(
        10.0, -torch.rand(n, generator=g, dtype=torch.float64) * 7.0
    )
    swap = torch.rand(n, generator=g) < 0.5
    re = torch.where(swap, small, big)
    im = torch.where(swap, big, small)
    return torch.complex(re, im).to(dtype)


def _worst_part_error(got, ref):
    """Largest relative error of either part, each against its own magnitude."""
    got = got.detach().cpu().to(torch.complex128)
    er = ((got.real - ref.real).abs() / ref.real.abs()).max()
    ei = ((got.imag - ref.imag).abs() / ref.imag.abs()).max()
    return float(max(er, ei))


class TestComplexSqrt:
    """``be.csqrt``: exact where the native root is, real arithmetic where it is not.

    The bound on each part is 6 u of the working dtype, relative to that part:
    ``t = sqrt((|z| + |a|) / 2)`` carries the error of ``hypot`` and one
    addition, halved by the square root, plus the root's own; the small part
    ``b / (2t)`` adds one quotient. With each operation faithfully rounded (at
    most one ulp, 2 u) that is 2 + 2 + 2 = 6 u; with correctly rounded
    operations it is 1.75 u. Measured on 200,000 such arguments: 2.5 u on the
    CPU and 3.0 u on the Apple GPU, against 2e7 u for the native root on the
    Apple GPU.
    """

    def test_the_real_arithmetic_root_is_the_principal_root(self):
        # complex64 against the complex128 root of the same arguments: the
        # device that needs this form has no wider type.
        z = _arguments(torch.complex64)
        ref = torch.sqrt(z.to(torch.complex128))
        assert _worst_part_error(csqrt_from_reals(z), ref) <= 6 * _U32

    def test_signed_zeros_and_the_real_axis_match_the_native_root_exactly(self):
        z = torch.tensor(
            [
                complex(4.0, 0.0),
                complex(4.0, -0.0),
                complex(-4.0, 0.0),
                complex(-4.0, -0.0),
                complex(0.0, 0.0),
                complex(0.0, -0.0),
                complex(-0.0, 0.0),
                complex(2.25, 0.0),
            ],
            dtype=torch.complex64,
        )
        got, native = csqrt_from_reals(z), torch.sqrt(z)
        assert torch.equal(got.real, native.real) and torch.equal(got.imag, native.imag)
        assert torch.equal(torch.signbit(got.imag), torch.signbit(native.imag))

    def test_the_gradient_is_the_native_roots(self):
        """Reverse mode through the real-arithmetic form equals the native root's."""
        values = [complex(5.0, 5e-4), complex(-4.0, 1e-3), complex(3.0, -4.0), complex(1e-3, 2.0)]
        grads = []
        for fn in (csqrt_from_reals, torch.sqrt):
            z = torch.tensor(values, dtype=torch.complex128, requires_grad=True)
            w = fn(z)
            (w.real * 0.3 + w.imag * 0.7).sum().backward()
            grads.append(z.grad.clone())
        assert torch.allclose(grads[0], grads[1], rtol=1e-12, atol=0.0)

    def test_the_backend_property(self, torch_backend_state, monkeypatch):
        be.set_backend("numpy")
        assert be.exact_complex_sqrt is True
        be.set_backend("torch")
        be.set_device("cpu")
        assert be.exact_complex_sqrt is True
        # The configured device, without needing the device itself.
        be.set_precision("float32")
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        be.set_device("mps")
        assert be.exact_complex_sqrt is False

    @pytest.mark.parametrize("precision", ["float32", "float64"])
    def test_on_the_cpu_csqrt_is_the_native_root_to_the_bit(self, torch_backend_state, precision):
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision(precision)
        dtype = torch.complex64 if precision == "float32" else torch.complex128
        z = _arguments(dtype)
        got, native = be.csqrt(z), torch.sqrt(z)
        assert torch.equal(got.real, native.real) and torch.equal(got.imag, native.imag)

    def test_on_numpy_csqrt_is_the_native_root_to_the_bit(self, torch_backend_state):
        be.set_backend("numpy")
        z = _arguments(torch.complex128).numpy()
        assert np.array_equal(be.csqrt(z), np.sqrt(z))

    @needs_mps
    def test_on_the_apple_gpu_csqrt_is_within_the_bound(self, torch_backend_state):
        be.set_backend("torch")
        be.set_precision("float32")
        be.set_device("mps")
        z = _arguments(torch.complex64)
        ref = torch.sqrt(z.to(torch.complex128))
        assert _worst_part_error(be.csqrt(z.to("mps")), ref) <= 6 * _U32
        # Control: the native root on the same device is the defect this fences.
        assert _worst_part_error(torch.sqrt(z.to("mps")), ref) > 1e3 * _U32


def _lossy_layer_rt(k, backend, precision, device="cpu"):
    """R and T of one quarter-wave layer n = 2.35 + i k on glass, unpolarized.

    41 wavelengths (0.40-0.80 um) by 17 angles (0-80 deg), through
    ``ThinFilmStack.compute_rtRTA_elementwise`` -- the per-ray path the
    engine's thin-film coating calls.
    """
    wl = np.linspace(0.40, 0.80, 41)
    ang = np.deg2rad(np.linspace(0.0, 80.0, 17))
    WL, TH = (a.ravel() for a in np.meshgrid(wl, ang, indexing="ij"))
    be.set_backend(backend)
    if backend == "torch":
        be.set_device("cpu")
        be.set_precision(precision)
        be.set_device(device)
    else:
        be.set_precision(precision)
    stack = ThinFilmStack(IdealMaterial(1.0), IdealMaterial(1.52), reference_wl_um=0.55)
    stack.add_layer_qwot(IdealMaterial(2.35, k))
    out = stack.compute_rtRTA_elementwise(be.array(WL), be.array(TH), polarization="u")
    R = np.asarray(be.to_numpy(out["R"]), dtype=np.float64)
    T = np.asarray(be.to_numpy(out["T"]), dtype=np.float64)
    return R, T


class TestLossyLayerOnTheAppleGpu:
    """A lossy quarter-wave layer at float32 on the Apple GPU against float64 on the CPU.

    The window is 64 u32 (3.8e-6) on R and on T: the operation-count window of
    the catalogue's float32 chains (64 for a two-refraction pencil ray), a bound
    on the characteristic-matrix chain of one layer (three layer cosines, two
    admittances, a phase, its cosine and sine, the matrix product, the
    amplitude quotient and its squared modulus). Measured on 401 x 81 samples:
    float32 on the CPU 6.3 to 9.2 u32; float32 on the Apple GPU with the
    real-arithmetic root 13 to 17 u32; with the native root 35 to 2,100 u32
    (k = 1e-2 to 1e-5), which this window refuses.
    """

    WINDOW = 64 * _U32

    @needs_mps
    @pytest.mark.parametrize("k", [1e-3, 1e-4])
    def test_reflectance_and_transmittance_within_the_window(self, torch_backend_state, k):
        R64, T64 = _lossy_layer_rt(k, "numpy", "float64")
        R32, T32 = _lossy_layer_rt(k, "torch", "float32", "mps")
        assert np.max(np.abs(R32 - R64)) <= self.WINDOW
        assert np.max(np.abs(T32 - T64)) <= self.WINDOW

    @needs_mps
    def test_control_the_native_root_breaks_the_window(self, torch_backend_state, monkeypatch):
        from optiland.backend.torch_backend import capabilities

        monkeypatch.setattr(capabilities, "_INEXACT_COMPLEX_SQRT_DEVICES", frozenset())
        R64, T64 = _lossy_layer_rt(1e-3, "numpy", "float64")
        R32, T32 = _lossy_layer_rt(1e-3, "torch", "float32", "mps")
        assert np.max(np.abs(T32 - T64)) > self.WINDOW

    @pytest.mark.parametrize("k", [1e-3, 1e-4])
    def test_the_cpu_float32_leg_is_unchanged_and_within_the_window(self, torch_backend_state, k):
        R64, T64 = _lossy_layer_rt(k, "numpy", "float64")
        R32, T32 = _lossy_layer_rt(k, "torch", "float32", "cpu")
        assert np.max(np.abs(R32 - R64)) <= self.WINDOW
        assert np.max(np.abs(T32 - T64)) <= self.WINDOW


# Float32 values (ordinary, negative, tiny normal, near the float32 maximum);
# widening float32 to float64 is exact, so "intact" is bit equality with the
# float32 values widened on the CPU.
_KNOWN = [1.5, 2.5, -3.25, 2.0**-100, 3.0e38]


class TestHostCopy:
    """A device tensor lands intact in a float64 host tensor, on every conversion path.

    ``to_device_dtype`` moves a tensor that leaves its device to the host in its
    own dtype and casts it there. The paths through it: the backend's
    ``to_tensor``, ``cast``, ``asarray``, ``atleast_1d``/``atleast_2d``,
    ``full_like`` with a tensor fill value, ``interp``, ``copy_to``, and the
    detectors' accumulation into a host buffer.
    """

    def test_on_one_device_the_helper_is_the_plain_conversion(self):
        from optiland.backend.torch_backend.capabilities import to_device_dtype

        x = torch.tensor(_KNOWN, dtype=torch.float32, requires_grad=True)
        y = to_device_dtype(x, "cpu", torch.float64)
        assert torch.equal(y, x.to(torch.float64))
        y.sum().backward()  # still on x's graph
        assert torch.equal(x.grad, torch.ones_like(x))
        assert to_device_dtype(x, None, None) is x
        arr = np.asarray(_KNOWN)
        assert torch.equal(to_device_dtype(arr, "cpu", torch.float32),
                           torch.as_tensor(arr, dtype=torch.float32, device="cpu"))

    @needs_mps
    def test_a_known_mps_tensor_lands_intact_in_a_float64_cpu_tensor(self):
        from optiland.backend.torch_backend.capabilities import to_device_dtype

        expected = torch.tensor(_KNOWN, dtype=torch.float32).to(torch.float64)
        for dtype in (torch.float32, torch.int32, torch.bool):
            x = torch.tensor(_KNOWN, dtype=torch.float32).to(dtype).to("mps")
            want = expected.to(dtype).to(torch.float64)
            assert torch.equal(to_device_dtype(x, "cpu", torch.float64), want), dtype
        z = torch.tensor(_KNOWN, dtype=torch.complex64, device="mps")
        assert torch.equal(to_device_dtype(z, "cpu", torch.complex128), expected.to(torch.complex128))

    @needs_mps
    def test_control_the_one_call_copy_writes_zeros(self):
        """The defect itself, on this torch. If this starts failing, torch has
        fixed it and the fence (and this control) can be dated and retired."""
        x = torch.tensor(_KNOWN, dtype=torch.float32, device="mps")
        assert float(x.to("cpu", torch.float64).abs().sum()) == 0.0

    @needs_mps
    def test_every_backend_conversion_path_keeps_the_values(self, torch_backend_state):
        from optiland.nonsequential.detectors.base import _accumulate_into

        expected = torch.tensor(_KNOWN, dtype=torch.float32).to(torch.float64)
        x = torch.tensor(_KNOWN, dtype=torch.float32, device="mps")
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float64")
        assert torch.equal(be.to_tensor(x), expected)
        assert torch.equal(be.cast(x), expected)
        assert torch.equal(be.asarray(x), expected)
        assert torch.equal(be.atleast_1d(x), expected)
        assert torch.equal(be.atleast_2d(x)[0], expected)
        filled = be.full_like(torch.zeros(3, dtype=torch.float64), x[0])
        assert torch.equal(filled, torch.full((3,), 1.5, dtype=torch.float64))
        grid = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], device="mps")
        assert torch.equal(be.interp(grid, grid, x), expected)
        dest = torch.zeros(len(_KNOWN), dtype=torch.float64)
        be.copy_to(x, dest)
        assert torch.equal(dest, expected)
        buf = torch.zeros(len(_KNOWN), dtype=torch.float64)
        _accumulate_into(buf, np.arange(len(_KNOWN)), x)
        assert torch.equal(buf, expected)
