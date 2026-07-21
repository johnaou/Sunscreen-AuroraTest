"""
Test 3 -- Aurora 1.5 (0.25 deg), the newest generation, HOURLY output.

Role in the experiment
  The temporal-detail arm. ~28 km, 721 x 1440, but the only model here with
  hourly steps, and it adds 22 variables the others lack (precip, cloud, soil,
  100 m wind, ...). It tests whether hourly steps + the richer variable set
  verify better at the site than a sharper-but-6-hourly high-res forecast.
  Run on the SAME ERA5 as Test 2 so Test3 - Test2 isolates the 1.5 upgrade.

Model : AuroraV1p5  (ikwessel/aurora-1.5 : aurora-0.25-v1.5.ckpt)
        variable_lead_time=True, so hourly sub-steps work out of the box.
Grid  : 0.25 deg, 721 x 1440
Data  : ERA5 (Copernicus CDS), extended surface set
Step  : hourly (fine_lead_times=[1..6] within each 6 h main step)
Site/window : Alexander Valley 38.6230932, -122.8856340 ; 7/13-7/16
GPU   : ~32 GB (v1.5 uses float16); see Microsoft guidance.

------------------------------------------------------------------------------
INPUT VARIABLES
  v1.5 wants 26 surface + 37 static + 5 atmos fields. Of the 26 surface vars, 7
  are OUTPUT-ONLY and auto zero-padded by the model (i10fg, blh, uvb_1h, ssrd_1h,
  ttr_1h, scaled_tp_1h, scaled_sf_1h). `insolation` is computed here from orbital
  mechanics. That leaves 18 surface fields to pull from ERA5 (mapped below) plus
  the 5 core atmospheric fields.

  The 36 STATIC fields (one-hot soil/vegetation type, sub-grid orography, lake
  cover/depth, ...) load automatically from the OFFICIAL v1.5 static pickle
  (ikwessel/aurora-1.5 : aurora-0.25-v1.5-static.pickle, already 0.25 deg,
  721 x 1440) -- the same pattern the MS 0.1 example uses for its static. No hand-
  built file needed. Pass --static-file only to override with your own NetCDF.

  CHECKPOINT: the docs snippet references microsoft/aurora, but the v1.5 weights
  are actually hosted on ikwessel/aurora-1.5, which is the installed package's
  default -- so we use AuroraV1p5().load_checkpoint() with no override.

Run:
  python "Test3:Aurora1.5(0.25°).py" --date 2025-07-13 --init-hour 12 \
      --hours 60 --device cuda
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime

import numpy as np
import torch
import xarray as xr
from huggingface_hub import hf_hub_download

from aurora import AuroraV1p5, Batch, Metadata
from aurora.insolation import insolation

import aurora_common as C
from era5_data import build_core_batch, download_era5_core, pick

# Official v1.5 static pickle (36 fields). Lives in the same HF repo as the
# checkpoint. NOTE: the docs snippet says microsoft/aurora, but the v1.5 files are
# actually hosted on ikwessel/aurora-1.5 (which is the installed package default).
V1P5_STATIC_REPO = "ikwessel/aurora-1.5"
V1P5_STATIC_FILE = "aurora-0.25-v1.5-static.pickle"

# v1.5 surface inputs beyond the core 4 (batch key -> CDS name / ERA5 short names).
# Excludes the 7 output-only vars and `insolation` (both handled by the model / here).
EXTRA_SURF = {
    "2d": ("2m_dewpoint_temperature", ("d2m", "2d")),
    "tcwv": ("total_column_water_vapour", ("tcwv",)),
    "tcc": ("total_cloud_cover", ("tcc",)),
    "100u": ("100m_u_component_of_wind", ("u100", "100u")),
    "100v": ("100m_v_component_of_wind", ("v100", "100v")),
    "sp": ("surface_pressure", ("sp",)),
    "lcc": ("low_cloud_cover", ("lcc",)),
    "mcc": ("medium_cloud_cover", ("mcc",)),
    "hcc": ("high_cloud_cover", ("hcc",)),
    "skt": ("skin_temperature", ("skt",)),
    "stl1": ("soil_temperature_level_1", ("stl1",)),
    "swvl1": ("volumetric_soil_water_layer_1", ("swvl1",)),
    "ci": ("sea_ice_cover", ("siconc", "ci")),
    # log-transformed internally; supply raw snow depth under the scaled_ key:
    "scaled_sd": ("snow_depth", ("sd",)),
}


def build_v1p5_batch(
    static_p, surf_p, atmos_p, model, static_file: str | None, smoke: bool
) -> Batch:
    """Assemble a v1.5 Batch: core ERA5 + extended surface + insolation + static."""
    batch = build_core_batch(static_p, surf_p, atmos_p)  # 2t/10u/10v/msl + atmos
    surf = xr.open_dataset(surf_p, engine="netcdf4")
    tdim = pick(surf, "valid_time", "time")
    i = surf.sizes[tdim] - 1

    # --- extended surface fields ---
    for key, (_cds, shorts) in EXTRA_SURF.items():
        try:
            name = pick(surf, *shorts)
            batch.surf_vars[key] = torch.from_numpy(
                surf[name].values[[i - 1, i]][None].copy()
            )
        except KeyError:
            if not smoke:
                raise
            ref = batch.surf_vars["2t"]
            batch.surf_vars[key] = torch.zeros_like(ref)

    # --- insolation (computed input channel, both init timesteps) ---
    lat = batch.metadata.lat.numpy().astype(np.float32)
    lon = batch.metadata.lon.numpy().astype(np.float32)
    times = list(surf[tdim].values.astype("datetime64[s]").astype(datetime)[[i - 1, i]])
    sol = insolation(times, lat, lon, enforce_2d=True)          # (2, H, W)
    batch.surf_vars["insolation"] = torch.from_numpy(sol[None].astype(np.float32).copy())

    # --- static fields (36): official v1.5 static pickle, already 0.25 deg ---
    if static_file:
        sds = xr.open_dataset(static_file, engine="netcdf4")
        batch.static_vars = {
            name: torch.from_numpy(np.asarray(sds[name].values, np.float32).copy())
            for name in model.static_vars
        }
    else:
        batch.static_vars = _load_v1p5_static(model)
    return batch


def _load_v1p5_static(model) -> dict:
    """36 v1.5 static fields from the official pickle (mirrors the MS pickle pattern)."""
    path = hf_hub_download(repo_id=V1P5_STATIC_REPO, filename=V1P5_STATIC_FILE)
    with open(path, "rb") as f:
        static = pickle.load(f)
    missing = [k for k in model.static_vars if k not in static]
    if missing:
        raise KeyError(f"v1.5 static pickle is missing fields: {missing}")
    return {k: torch.from_numpy(np.asarray(static[k], np.float32)) for k in model.static_vars}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=C.INIT_DATE, help="init date YYYY-MM-DD")
    ap.add_argument("--init-hour", type=int, default=C.INIT_HOUR_UTC, choices=[0, 6, 12, 18])
    ap.add_argument("--hours", type=int, default=C.HORIZON_HOURS,
                    help="forecast horizon in hours (default covers the 7/14-7/15 window)")
    ap.add_argument("--static-file", help="override: your own v1.5 static NetCDF "
                    "(default loads the official static pickle)")
    ap.add_argument("--smoke", action="store_true",
                    help="zero-fill any extended surface var CDS didn't return (plumbing only)")
    ap.add_argument("--half", type=float, default=C.DEFAULT_HALF_DEG)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="test3_v1p5_wind.csv")
    args = ap.parse_args()

    date = datetime.strptime(args.date, "%Y-%m-%d")
    times = [f"{args.init_hour - 6:02d}:00", f"{args.init_hour:02d}:00"]
    extra_cds = [cds for cds, _ in EXTRA_SURF.values()]

    print(f"[Test3] ERA5 (extended) {args.date} {times} for {C.SITE_NAME}")
    static_p, surf_p, atmos_p = download_era5_core(date, times, extra_surface=extra_cds)

    print("[Test3] loading aurora-0.25-v1.5.ckpt (ikwessel/aurora-1.5) ...")
    model = AuroraV1p5()
    model.load_checkpoint()  # package default -> ikwessel/aurora-1.5 (see header)

    batch = build_v1p5_batch(static_p, surf_p, atmos_p, model, args.static_file, args.smoke)

    # Hourly: 6 sub-steps per 6 h main step; last entry must equal the 6 h base step.
    main_steps = max(1, round(args.hours / 6))
    rows = C.run_and_extract(
        model, batch,
        steps=main_steps,
        fine_lead_times=[1, 2, 3, 4, 5, 6],
        half=args.half,
        device=args.device,
        label="test3-v1.5-hourly",
    )
    C.write_csv(rows, args.out)


if __name__ == "__main__":
    main()
