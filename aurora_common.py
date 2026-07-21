"""
aurora_common.py -- shared config + helpers for the Aurora forecasting tests.

All three tests (Test1 0.1 HighRes / Test2 0.25 Pretrained / Test3 v1.5) forecast
the SAME site and window so their errors are directly comparable, then extract
10 m wind speed + direction over a small box around the site and write a tidy CSV.

Site (Alexander Valley / Geyserville, Healdsburg deployment test):
    38.6230932 N, -122.8856340 E
Window:
    7/13 - 7/16.  End goal: best 24 h wind forecast issued one day before deploy,
    prioritising wind speed and direction.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from aurora import rollout

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Site / window configuration
# --------------------------------------------------------------------------- #
SITE_NAME = "Alexander Valley / Geyserville"
SITE_LAT = 38.6230932
SITE_LON = -122.8856340          # signed, -180..180
SITE_LON_360 = SITE_LON % 360.0  # 237.114..., ERA5/Aurora grids are 0..360

# Half-width of the extraction box in degrees. 0.5 deg gives a few grid cells on
# the 0.25 grid and ~10 cells on the 0.1 grid -- enough to see local structure
# while staying near the point.
DEFAULT_HALF_DEG = 0.5

# ---- Evaluation window & forecast issue time -------------------------------
# The Healdsburg benchmark scores 7/14-7/15 (charts run 7/14 09:00 -> 7/15 15:00
# PDT = 7/14 16:00 -> 7/15 22:00 UTC). To be comparable, our forecasts must cover
# that window.
#
# Framing = the doc's end goal: "best 24 h forecast issued ONE DAY PRIOR to
# deployment." So we issue from 7/13 and roll out far enough to blanket 7/14-7/15.
#   init 7/13 12:00 UTC (= 7/13 05:00 PDT), +60 h -> 7/16 00:00 UTC (7/15 17:00 PDT).
# The 7/14 valid times then sit at ~12-30 h lead (the ~24 h use case); 7/15 runs
# out to ~54 h lead. NOTE: this is a longer horizon than the benchmark's 2-4 h
# "planning" band, which only the hourly v1.5 (Test 3) can actually hit. For a
# constant ~24 h lead across both days, add a second init on 7/14 for the 7/15
# valid times.
INIT_DATE = "2025-07-13"
INIT_HOUR_UTC = 12       # 05:00 PDT
HORIZON_HOURS = 60       # covers 7/14 09:00 -> 7/15 17:00 PDT

# Kept for reference.
WINDOW_START = "2025-07-13"
WINDOW_END = "2025-07-16"


# --------------------------------------------------------------------------- #
# Wind math
# --------------------------------------------------------------------------- #
def wind_from_uv(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """10 m wind speed (m/s) and meteorological direction (deg the wind blows FROM)."""
    speed = np.hypot(u, v)
    direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return speed, direction


# --------------------------------------------------------------------------- #
# Grid geometry
# --------------------------------------------------------------------------- #
def box_masks(
    lat: np.ndarray,
    lon: np.ndarray,
    center_lat: float = SITE_LAT,
    center_lon_360: float = SITE_LON_360,
    half: float = DEFAULT_HALF_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks selecting a +/- `half` deg box around the site."""
    lat_mask = (lat >= center_lat - half) & (lat <= center_lat + half)
    lon_mask = (lon >= center_lon_360 - half) & (lon <= center_lon_360 + half)
    if not lat_mask.any() or not lon_mask.any():
        raise ValueError(
            "extraction box does not intersect the grid -- check the site coords "
            f"(lat range {lat.min():.2f}..{lat.max():.2f}, "
            f"lon range {lon.min():.2f}..{lon.max():.2f})"
        )
    return lat_mask, lon_mask


def nearest_index(lat: np.ndarray, lon: np.ndarray) -> tuple[int, int]:
    """Indices of the single grid cell closest to the site."""
    iy = int(np.abs(lat - SITE_LAT).argmin())
    ix = int(np.abs(lon - SITE_LON_360).argmin())
    return iy, ix


# --------------------------------------------------------------------------- #
# Forecast driver -- shared by all three tests
# --------------------------------------------------------------------------- #
def run_and_extract(
    model,
    batch,
    *,
    steps: int,
    fine_lead_times: Optional[Sequence[float]] = None,
    half: float = DEFAULT_HALF_DEG,
    device: str = "cpu",
    label: str = "",
) -> list[dict]:
    """Roll `model` forward and return per-cell wind rows over the site box.

    `steps` is the number of *main* (6 h) steps. For hourly output (v1.5) pass
    `fine_lead_times=[1,2,3,4,5,6]`; rollout then yields one Batch per sub-step,
    so 24 h of hourly output = steps=4 with those six sub-steps.

    Valid time and lead are read from each prediction's own metadata, so this is
    correct for both 6-hourly and hourly rollouts.
    """
    # Grid geometry from the original (float64) coords, before any dtype change.
    lat = batch.metadata.lat.detach().cpu().numpy()
    lon = batch.metadata.lon.detach().cpu().numpy()

    model = model.to(device)
    model.eval()
    # MPS has no float64; Aurora runs in float32 anyway, so cast off-CPU devices.
    if device != "cpu":
        batch = batch.type(torch.float32)
    batch = batch.to(device)

    lat_mask, lon_mask = box_masks(lat, lon, half=half)
    box_lat = lat[lat_mask]
    box_lon_disp = ((lon[lon_mask] + 180.0) % 360.0) - 180.0  # -> signed for display
    iy0, ix0 = nearest_index(lat, lon)

    # Display lat/lon of the single nearest cell, to flag the site row in output.
    site_lat_disp = float(lat[iy0])
    site_lon_disp = float(((lon[ix0] + 180.0) % 360.0) - 180.0)

    base_time = _as_utc(batch.metadata.time[0])
    rows: list[dict] = []

    with torch.inference_mode():
        for pred in rollout(model, batch, steps=steps, fine_lead_times=fine_lead_times):
            valid = _as_utc(pred.metadata.time[0])
            lead_h = (valid - base_time).total_seconds() / 3600.0

            u_full = pred.surf_vars["10u"][0, 0].float().cpu().numpy()
            v_full = pred.surf_vars["10v"][0, 0].float().cpu().numpy()
            u = u_full[np.ix_(lat_mask, lon_mask)]
            v = v_full[np.ix_(lat_mask, lon_mask)]
            speed, direction = wind_from_uv(u, v)

            # nearest-cell value = the headline number for the site
            pt_speed, pt_dir = wind_from_uv(
                np.array(u_full[iy0, ix0]), np.array(v_full[iy0, ix0])
            )
            print(
                f"[{label}] +{lead_h:5.1f}h  site {float(pt_speed):5.2f} m/s "
                f"@ {float(pt_dir):5.1f} deg   box mean {speed.mean():5.2f} "
                f"max {speed.max():5.2f} m/s"
            )

            for iy, la in enumerate(box_lat):
                for ix, lo in enumerate(box_lon_disp):
                    rows.append(
                        {
                            "model": label,
                            "lead_hours": round(lead_h, 2),
                            "valid_time_utc": valid.isoformat(),
                            "lat": round(float(la), 4),
                            "lon": round(float(lo), 4),
                            "is_site_cell": int(
                                np.isclose(la, site_lat_disp)
                                and np.isclose(lo, site_lon_disp)
                            ),
                            "speed_ms": round(float(speed[iy, ix]), 3),
                            "dir_deg_from": round(float(direction[iy, ix]), 1),
                        }
                    )
    return rows


def _as_utc(t) -> datetime:
    """Normalise assorted datetime-ish objects to a tz-aware UTC datetime."""
    if isinstance(t, np.datetime64):
        t = t.astype("datetime64[s]").astype(datetime)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t


def write_csv(rows: Iterable[dict], out_path: str | Path) -> Path:
    rows = list(rows)
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = HERE / out_path
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out_path}")
    return out_path
