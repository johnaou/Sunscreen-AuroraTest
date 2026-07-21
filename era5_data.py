"""
era5_data.py -- download ERA5 initial conditions from the Copernicus CDS and
build a core-variable Aurora `Batch` (2t/10u/10v/msl + z/u/v/t/q + static).

Used by Test 2 (0.25 pretrained). Test 3 (v1.5) needs a larger variable set and
builds its batch itself, but reuses `download_era5_core` for the shared fields.

SETUP
  pip install cdsapi
  Put a CDS API key in ~/.cdsapirc  (https://cds.climate.copernicus.eu/how-to-api)
  and accept the ERA5 licences for single-levels and pressure-levels.

DATA NOTE
  ERA5 is reanalysis, ~5 day latency. Use dates >= ~6 days in the past. This is a
  hindcast of the Healdsburg window for model comparison, exactly what the test
  needs. For a live pre-deployment forecast you would swap in IFS-HRES T0.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from aurora import Batch, Metadata

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "era5_cache"

PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


def download_era5_core(
    date: datetime, times: list[str], extra_surface: list[str] | None = None
) -> tuple[Path, Path, Path]:
    """Download static + surface + pressure-level ERA5 for one day.

    `times` are the 6-hourly steps to fetch (>=2; we use the last two as
    history+init). `extra_surface` lets Test 3 request the v1.5 surface fields in
    the same call. Files are cached and re-used.
    """
    import cdsapi

    DATA_DIR.mkdir(exist_ok=True)
    y, m, d = f"{date:%Y}", f"{date:%m}", f"{date:%d}"
    tag = "_".join(t.replace(":", "") for t in times)
    static_p = DATA_DIR / "static.nc"
    surf_p = DATA_DIR / f"surface_{y}{m}{d}_{tag}.nc"
    atmos_p = DATA_DIR / f"atmospheric_{y}{m}{d}_{tag}.nc"

    client = cdsapi.Client()

    if not static_p.exists():
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": ["geopotential", "land_sea_mask", "soil_type"],
                "year": "2023", "month": "01", "day": "01", "time": "00:00",
                "format": "netcdf",
            },
            str(static_p),
        )

    surface_vars = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
    ] + (extra_surface or [])
    if not surf_p.exists():
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": surface_vars,
                "year": y, "month": m, "day": d, "time": times,
                "format": "netcdf",
            },
            str(surf_p),
        )

    if not atmos_p.exists():
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "specific_humidity",
                    "geopotential",
                ],
                "pressure_level": [str(p) for p in PRESSURE_LEVELS],
                "year": y, "month": m, "day": d, "time": times,
                "format": "netcdf",
            },
            str(atmos_p),
        )

    return static_p, surf_p, atmos_p


def pick(ds: xr.Dataset, *candidates: str) -> str:
    """First coord/var name present (CDS has renamed several across versions)."""
    for c in candidates:
        if c in ds.variables or c in ds.coords:
            return c
    raise KeyError(f"none of {candidates} in {list(ds.variables)}")


def build_core_batch(static_p: Path, surf_p: Path, atmos_p: Path) -> Batch:
    """Build a core-variable Aurora Batch from downloaded ERA5 files."""
    static = xr.open_dataset(static_p, engine="netcdf4")
    surf = xr.open_dataset(surf_p, engine="netcdf4")
    atmos = xr.open_dataset(atmos_p, engine="netcdf4")

    time_dim = pick(surf, "valid_time", "time")
    level_dim = pick(atmos, "pressure_level", "level")
    i = surf.sizes[time_dim] - 1  # last two steps: i-1 (history), i (init)

    def spair(name: str) -> torch.Tensor:
        return torch.from_numpy(surf[name].values[[i - 1, i]][None].copy())

    def apair(name: str) -> torch.Tensor:
        return torch.from_numpy(atmos[name].values[[i - 1, i]][None].copy())

    def sfield(name: str) -> torch.Tensor:
        return torch.from_numpy(static[name].values[0].copy())

    valid = surf[time_dim].values.astype("datetime64[s]").astype(datetime)

    return Batch(
        surf_vars={
            "2t": spair(pick(surf, "t2m", "2t")),
            "10u": spair(pick(surf, "u10", "10u")),
            "10v": spair(pick(surf, "v10", "10v")),
            "msl": spair(pick(surf, "msl")),
        },
        static_vars={
            "z": sfield(pick(static, "z")),
            "slt": sfield(pick(static, "slt")),
            "lsm": sfield(pick(static, "lsm")),
        },
        atmos_vars={
            "t": apair(pick(atmos, "t")),
            "u": apair(pick(atmos, "u")),
            "v": apair(pick(atmos, "v")),
            "q": apair(pick(atmos, "q")),
            "z": apair(pick(atmos, "z")),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf[pick(surf, "latitude")].values.copy()),
            lon=torch.from_numpy(surf[pick(surf, "longitude")].values.copy()),
            time=(valid[i],),
            atmos_levels=tuple(int(x) for x in atmos[level_dim].values),
        ),
    )
