"""Per-surface flux booking for the energy ledger.

``docs/theory/10_ledger_and_diagnostics.md`` (10.1) requires every watt a
source emits to be booked exactly once, into one of eight destinations, and
(10.2) requires the identity to close on *every realisation* rather than
only in expectation. Two of those destinations were missing.

**The coating bin.** A mirror below unit reflectance multiplies the flux by
``R`` and a coating with ``R + T < 1`` multiplies it by a weight whose
expectation is ``R + T``. Neither loss had anywhere to go, so the ledger
reported an energy defect on a physically correct scene -- the size of the
loss, exactly. ``book_loss`` gives it a destination.

**The sampling residual.** The detached-decision estimator (3.11) replaces
an incoming weight ``w`` by ``wR/p`` or ``wT/(1-p)``; unless ``p = R/(R+T)``
that is not ``w`` minus the booked loss, even though it is in expectation.
``book_residual`` accumulates the difference, so the identity closes term by
term. Its expectation is zero and its magnitude shrinks as the square root
of the ray count -- a residual that does not shrink is a weight-update bug,
not sampling noise.

Both accumulate where the ray state lives and are read back once, after the
trace, like every other tally in the loop; and both are detached, because
the ledger is a diagnostic and holding the trace's graph alive in it would
multiply the gradient-mode memory ceiling.

Kramer Harrison, 2026
"""

from __future__ import annotations

from optiland.nonsequential._tally import Tally, masked_sum
from optiland.nonsequential.components.sampling_support import detached


class LedgerBooking:
    """Mixin: a surface books what it removes from, and adds to, the trace.

    Attributes:
        _coating_loss: Flux this surface removed as mirror or coating loss
            over the current trace, in the theory's ``Phi_coat`` bin.
        _sampling_residual: This surface's contribution to ``Phi_samp``.
    """

    def reset_ledger(self) -> None:
        """Start a fresh trace's books."""
        self._coating_loss = Tally()
        self._sampling_residual = Tally()

    def _tally(self, name: str) -> Tally:
        tally = getattr(self, name, None)
        if tally is None:
            self.reset_ledger()
            tally = getattr(self, name)
        return tally

    def book_loss(self, flux, fraction, hit_mask) -> None:
        """Book ``flux * fraction`` on the hit rays as coating loss.

        Args:
            flux: Per-ray flux *before* the weight is applied, shape (N,).
            fraction: Per-ray fraction of it this surface absorbs, shape
                (N,). Zero where the surface is lossless.
            hit_mask: Per-ray mask of rays interacting with this surface.
        """
        self._tally("_coating_loss").add(
            masked_sum(detached(flux * fraction), hit_mask)
        )

    def book_residual(self, flux, residual_fraction, hit_mask) -> None:
        """Book ``flux * residual_fraction`` on the hit rays as Phi_samp.

        Args:
            flux: Per-ray flux *before* the weight is applied, shape (N,).
            residual_fraction: Per-ray ``1 - weight - loss_fraction``, the
                event residual of (10.2) divided by the incoming weight.
            hit_mask: Per-ray mask of rays interacting with this surface.
        """
        self._tally("_sampling_residual").add(
            masked_sum(detached(flux * residual_fraction), hit_mask)
        )

    def book_lobe(self, flux, gate, albedo_gate, hit_mask) -> None:
        """Split a BSDF lobe's ``1 - gate`` between the two bins.

        A lobe whose weight is a physical fraction of the incident flux
        (``BaseBSDF.weight_is_albedo``) passes ``albedo_gate is gate`` and
        the whole of ``1 - gate`` is a surface loss. A lobe whose weight is
        a sampling weight passes the surface's albedo as ``albedo_gate``:
        ``1 - albedo_gate`` is then what the surface physically kept and
        ``albedo_gate - gate`` is the estimator's event residual, which has
        expectation zero. The two always sum to ``1 - gate``, so the
        identity closes the same either way.

        Args:
            flux: Per-ray flux *before* the lobe's weight is applied.
            gate: Per-ray weight the lobe returned, 1 off the lobe.
            albedo_gate: Per-ray albedo, 1 off the lobe; ``gate`` itself
                when the weight is already an albedo.
            hit_mask: Per-ray mask of rays interacting with this surface.
        """
        self.book_loss(flux, 1.0 - albedo_gate, hit_mask)
        if albedo_gate is not gate:
            self.book_residual(flux, albedo_gate - gate, hit_mask)

    @property
    def coating_loss(self) -> float:
        """Flux booked into the coating bin this trace [W]."""
        return self._tally("_coating_loss").value()

    @property
    def sampling_residual(self) -> float:
        """Flux booked into the sampling residual this trace [W]."""
        return self._tally("_sampling_residual").value()
