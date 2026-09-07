"""Per-lot marked exposure. Official FINAL close vs each lot's purchase price."""

from __future__ import annotations

import math
from typing import Iterable


def _valid_px(px: object) -> bool:
    try:
        v = float(px)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0


def marked_units_from_lots(lots: Iterable, price: float | None) -> float | None:
    """Σ(lot.acquisition_units × price / lot.acquisition_price). None if price unusable."""
    if not _valid_px(price):
        return None
    px = float(price)
    total = 0.0
    any_lot = False
    for lot in lots:
        acq = getattr(lot, "acquisition_units", None)
        cost = getattr(lot, "acquisition_price", None)
        if not _valid_px(acq) or not _valid_px(cost):
            continue
        any_lot = True
        total += float(acq) * px / float(cost)
    return total if any_lot else None


def stock_exposure_for_allocation(positions) -> tuple[dict[str, float], dict[str, float], dict[str, float | None], list[str]]:
    """Marked FINAL exposure for allocation. Unpriced holdings are omitted, not fabricated."""
    marked: dict[str, float] = {}
    acquisition: dict[str, float] = {}
    preview: dict[str, float | None] = {}
    unpriced: list[str] = []
    for pos in positions:
        inst = str(pos.instrument_id)
        if "btc" in inst.lower():
            continue
        acquisition[inst] = float(getattr(pos, "acquisition_units_total", 0.0) or 0.0)
        preview[inst] = getattr(pos, "marked_units_intraday_preview", None)
        final = getattr(pos, "marked_units_final", None)
        if final is not None:
            marked[inst] = float(final)
        elif acquisition[inst] > 1e-12:
            unpriced.append(inst)
    return marked, acquisition, preview, unpriced
