"""
Test 1 -- Aurora 0.1 Fine-Tuned (highest-resolution checkpoint).

Role in the experiment
  The spatial-detail arm. ~11 km, 1801 x 3600 global grid, fine-tuned on
  IFS-HRES analysis. It is the only model that resolves the Alexander Valley's
  hills and valley floor, so it tests whether that fine terrain structure
  actually improves accuracy at the site versus the coarser 0.25 models. Aurora
  is best-in-class here *only* when fed HRES analysis -- which is the RDA source.

Model : AuroraHighRes  (microsoft/aurora : aurora-0.1-finetuned.ckpt)
Grid  : 0.1 deg, 1801 x 3600
Data  : IFS-HRES operational analysis, NCAR RDA dataset d113001
Step  : 6-hourly
Site/window : Alexander Valley 38.6230932, -122.8856340 ; 7/13-7/16
GPU   : heaviest of the three (1801 x 3600 global) -- use the EC2 G6e box.

------------------------------------------------------------------------------
DATA: this mirrors the official Aurora HRES 0.1 example.
  Source   : https://data.rda.ucar.edu/d113001/  (NCAR RDA "ECMWF IFS HRES").
  Surface  : ec.oper.an.sfc/{YYYYMM}/...128_{param}_{var}.regn1280sc.{YYYYMMDD}.grb
             (one file per day, all synoptic hours; vars 2t,10u,10v,msl)
  Levels   : ec.oper.an.pl/{YYYYMM}/...128_{param}_{var}.regn1280{sc|uv}.{YYYYMMDDHH}.grb
             (one file per synoptic hour; scalars t,q,z use 'sc', winds u,v use 'uv')
  Read     : xarray + cfgrib engine  (pip install eccodes cfgrib)
  Grid     : the archive is a reduced Gaussian grid; we build the Batch then call
             batch.regrid(0.1) to land on the clean 1801 x 3600 grid the model wants.
  Static   : lsm/z/slt from HuggingFace microsoft/aurora : aurora-0.1-static.pickle,
             loaded AFTER regrid (it is already 0.1 deg, 1801x3600). This matches
             the MS 0.1 example; the RDA sfc z/slt/lsm files are reference-only.
  Auth     : the official guide downloads d113001 with plain unauthenticated
             requests.get() -- it is CC-BY / unrestricted, so no account is needed
             for direct download. If the file server ever gates on login, put
             credentials in ~/.netrc for machine data.rda.ucar.edu.

  NOTE: RDA filename conventions are finicky; if a download 404s, open the month
  directory in a browser and adjust build_urls() to match the exact names.

SMOKE PATH (pipeline check, NOT the real 0.1 experiment)
  --source wb2 uses the built-in WeatherBench2 HRES-T0 loader (0.25 deg). It
  exercises the full model + extraction without RDA access; treat its numbers as
  plumbing validation only.

Run:
  python "Test1:Aurora0.1°Fine-Tuned.py" --source rda --date 2025-07-13 \
      --init-hour 12 --steps 4 --device cuda
  python "Test1:Aurora0.1°Fine-Tuned.py" --source wb2 --date 2025-07-13 --device cuda
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import requests
import torch
import xarray as xr
from huggingface_hub import hf_hub_download

from aurora import AuroraHighRes, Batch, Metadata

import aurora_common as C

HERE = Path(__file__).resolve().parent
RDA_CACHE = HERE / "rda_cache"
RDA_BASE = "https://data.rda.ucar.edu/d113001"

# ECMWF GRIB parameter codes (table 128).
PARAM = {
    "2t": "167", "10u": "165", "10v": "166", "msl": "151",
    "t": "130", "u": "131", "v": "132", "q": "133", "z": "129",
}
SURF_VARS = ["2t", "10u", "10v", "msl"]
ATMOS_VARS = ["t", "u", "v", "q", "z"]
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


def build_urls(day: datetime, hours: list[int]) -> tuple[dict, dict]:
    """Return {var: url} for surface (per-day) and {(var,hour): url} for levels."""
    ym, ymd = f"{day:%Y%m}", f"{day:%Y%m%d}"
    surf = {
        v: f"{RDA_BASE}/ec.oper.an.sfc/{ym}/"
           f"ec.oper.an.sfc.128_{PARAM[v]}_{v}.regn1280sc.{ymd}.grb"
        for v in SURF_VARS
    }
    atmos = {}
    for v in ATMOS_VARS:
        grid = "uv" if v in ("u", "v") else "sc"
        for hh in hours:
            atmos[(v, hh)] = (
                f"{RDA_BASE}/ec.oper.an.pl/{ym}/"
                f"ec.oper.an.pl.128_{PARAM[v]}_{v}.regn1280{grid}.{ymd}{hh:02d}.grb"
            )
    return surf, atmos


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}")
    with requests.get(url, stream=True, timeout=600) as r:  # uses ~/.netrc if present
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def _open(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


def build_highres_batch_from_rda(day: datetime, init_hour: int) -> Batch:
    """Download d113001 GRIB for init and init-6h, build Batch, regrid to 0.1."""
    t_init = day.replace(hour=init_hour)
    t_hist = t_init - timedelta(hours=6)
    hours = [t_hist.hour, t_init.hour]
    surf_urls, atmos_urls = build_urls(day, hours)

    # Note: if init_hour==0, history falls on the previous day; the sfc file is
    # per-day, so extend this to fetch the prior day's file for that case.
    if t_hist.date() != t_init.date():
        raise SystemExit("init-hour 0 spans two RDA daily files; use 6/12/18 for now")

    surf_ds = {v: _open(_download(u, RDA_CACHE / Path(u).name)) for v, u in surf_urls.items()}
    atmos_ds = {k: _open(_download(u, RDA_CACHE / Path(u).name)) for k, u in atmos_urls.items()}

    # Positional selection like the MS example's `.values[:2]`. The per-day sfc
    # file holds times [00,06,12,18]; init index = init_hour//6, history = one before.
    idx_init = init_hour // 6
    idx_hist = idx_init - 1

    def surf_pair(v: str) -> torch.Tensor:
        ds = surf_ds[v]
        name = list(ds.data_vars)[0]
        a = ds[name].values[[idx_hist, idx_init]]  # (2, H, W)
        return torch.from_numpy(a[None].copy())  # (1, 2, H, W)

    def atmos_pair(v: str) -> torch.Tensor:
        # per-hour pl files (history, init) stacked -> (1, 2, C, H, W)
        per_time = []
        for hh in hours:
            ds = atmos_ds[(v, hh)]
            name = list(ds.data_vars)[0]
            da = ds[name]  # (level, H, W)
            da = da.sel(isobaricInhPa=LEVELS) if "isobaricInhPa" in da.dims else da
            per_time.append(da.values)
        a = np.stack(per_time)  # (2, C, H, W)
        return torch.from_numpy(a[None].copy())

    ref = surf_ds["2t"]
    lat = ref[list(ref.data_vars)[0]].latitude.values  # RDA HRES is 90 -> -90
    lon = ref[list(ref.data_vars)[0]].longitude.values

    batch = Batch(
        surf_vars={v: surf_pair(v) for v in SURF_VARS},
        # Per the MS 0.1 example, static is loaded from a HuggingFace pickle AFTER
        # regrid (it is already at 0.1 deg, 1801x3600), not from RDA.
        static_vars={},
        atmos_vars={v: atmos_pair(v) for v in ATMOS_VARS},
        metadata=Metadata(
            lat=torch.from_numpy(lat.copy()),
            lon=torch.from_numpy(lon.copy()),
            time=(t_init,),
            atmos_levels=tuple(LEVELS),
        ),
    )

    batch = batch.regrid(res=0.1)
    batch.static_vars = _load_static_0p1()
    return batch


def _load_static_0p1() -> dict:
    """Static z/slt/lsm at 0.1 deg from HuggingFace (matches the MS 0.1 example)."""
    path = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.1-static.pickle")
    with open(path, "rb") as f:
        static = pickle.load(f)
    return {k: torch.from_numpy(np.asarray(v)) for k, v in static.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["rda", "wb2"], default="rda")
    ap.add_argument("--date", default=C.INIT_DATE, help="init date YYYY-MM-DD")
    ap.add_argument("--init-hour", type=int, default=C.INIT_HOUR_UTC, choices=[6, 12, 18])
    ap.add_argument("--steps", type=int, default=C.HORIZON_HOURS // 6,
                    help="6-hourly steps (default covers the 7/14-7/15 window)")
    ap.add_argument("--half", type=float, default=C.DEFAULT_HALF_DEG)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="test1_highres_wind.csv")
    args = ap.parse_args()

    date = datetime.strptime(args.date, "%Y-%m-%d")

    if args.source == "rda":
        print(f"[Test1] building 0.1 HRES batch from RDA d113001 for {C.SITE_NAME}")
        batch = build_highres_batch_from_rda(date, args.init_hour)
        label = "test1-0.1-highres"
    else:
        print("[Test1] SMOKE: WeatherBench2 HRES-T0 0.25 deg (not the real 0.1 result)")
        from aurora.foundry.demo.hres_t0_data import load_batch
        batch = load_batch(date)
        label = "test1-SMOKE-0.25-hrest0"

    print("[Test1] loading aurora-0.1-finetuned.ckpt ...")
    model = AuroraHighRes()
    model.load_checkpoint()

    rows = C.run_and_extract(
        model, batch,
        steps=args.steps,
        half=args.half,
        device=args.device,
        label=label,
    )
    C.write_csv(rows, args.out)


if __name__ == "__main__":
    main()
