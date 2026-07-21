# Aurora Wind-Forecast Test (Sunscreen / Healdsburg)

Testing Microsoft **[Aurora](https://microsoft.github.io/aurora/)** for short-range
**10 m wind speed + direction** forecasting around a deployment site in the
California Central Valley (Alexander Valley / Geyserville), and comparing three
Aurora model variants against each other and against real observations.

**Goal:** the best 24 h wind forecast (speed + direction) issued *one day prior*
to deployment, benchmarked against the Healdsburg 7/14–7/15 mast data and other
forecast products (HRRR, NBM, RRFS, Wunderground).

Each test builds an Aurora input `Batch` from real initial conditions, rolls the
model forward, extracts a small lat/lon box around the site, derives wind speed &
meteorological direction from the `10u`/`10v` components, and writes a tidy CSV.

## The three tests

| Script | Model | Data source | Step | Notes |
|---|---|---|---|---|
| `Test1:Aurora0.1°Fine-Tuned.py` | `AuroraHighRes` (0.1°, 1801×3600) | IFS-HRES analysis, **NCAR RDA d113001** (GRIB) | 6-hourly | Highest resolution; resolves local terrain. Heaviest — needs a GPU. |
| `Test2:Aurora0.25°Pretrained.py` | `AuroraPretrained` (0.25°, 721×1440) | **ERA5** (Copernicus CDS) | 6-hourly | Un-fine-tuned baseline; the number a local fine-tune must beat. |
| `Test3:Aurora1.5(0.25°).py` | `AuroraV1p5` (0.25°) | **ERA5** (Copernicus CDS) | **hourly** | Newest generation; +22 variables; only one that hits a 2–4 h lead. |

Shared code:
- `aurora_common.py` — site/window config, wind math, box extraction, the rollout
  driver, and CSV writer (used by all three tests).
- `era5_data.py` — ERA5 download (CDS) + batch builder (Tests 2 & 3).
- `smoke_test.py` — minimal random-input sanity check that the model runs.

## Setup

```bash
python3.12 -m venv .venv          # or: uv venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Credentials

- **ERA5 (Tests 2 & 3)** — free Copernicus CDS account. Put your key in
  `~/.cdsapirc` (see https://cds.climate.copernicus.eu/how-to-api) and accept the
  licences for `reanalysis-era5-single-levels` and `reanalysis-era5-pressure-levels`.
- **HRES / NCAR RDA (Test 1)** — the dataset (`d113001`) is CC-BY / unrestricted,
  and the official Microsoft guide downloads it with unauthenticated HTTPS, so no
  account is normally required. If the file server ever gates on login, register at
  https://gdex.ucar.edu and add credentials to `~/.netrc` for `data.rda.ucar.edu`.

Model checkpoints and static fields download automatically from HuggingFace on
first run.

## Running

```bash
# Test 2 — ERA5 baseline (runnable on CPU; slow but works)
python "Test2:Aurora0.25°Pretrained.py" --device cpu

# Test 3 — v1.5 hourly
python "Test3:Aurora1.5(0.25°).py" --device cuda

# Test 1 — 0.1° HRES (needs a GPU)
python "Test1:Aurora0.1°Fine-Tuned.py" --source rda --device cuda
# quick model smoke test without RDA data (uses 0.25° WeatherBench2 HRES-T0):
python "Test1:Aurora0.1°Fine-Tuned.py" --source wb2 --device cuda
```

Common flags: `--date YYYY-MM-DD`, `--init-hour {6,12,18}`, `--device {cpu,cuda}`,
`--out file.csv`. Test 1/2 take `--steps` (×6 h); Test 3 takes `--hours`.

The forecast site, evaluation window, and init time are centralized in
`aurora_common.py` (`SITE_LAT/LON`, `INIT_DATE`, `INIT_HOUR_UTC`, `HORIZON_HOURS`)
so all three tests stay comparable. Defaults issue from **7/13 12:00 UTC** and roll
out **60 h** to cover the 7/14–7/15 window.

## Output

Each run writes a CSV with one row per grid cell per lead time:

```
model, lead_hours, valid_time_utc, lat, lon, is_site_cell, speed_ms, dir_deg_from
```

`dir_deg_from` is the meteorological direction the wind blows *from* (degrees).
Filter `is_site_cell == 1` for the single nearest-cell forecast at the site.

## Important notes

- **GPU:** the 0.25° models want ~40 GB GPU (v1.5 ~32 GB); the 0.1° model is larger
  still. CPU works but is slow. **Apple Silicon MPS does *not* work** — Aurora's
  Fourier encoding uses float64, which MPS doesn't support. Use CPU or CUDA.
- **v1.5 checkpoint** lives on the `ikwessel/aurora-1.5` HuggingFace repo (not
  `microsoft/aurora`); the code uses the package default, so no action needed.
- **ERA5 is reanalysis** (~5-day latency), so Test 2/3 dates must be in the past —
  this reproduces the test window. For a live pre-deployment forecast, swap ERA5
  for IFS-HRES T0.
- **Data is not committed** (see `.gitignore`): `era5_cache/`, `rda_cache/`, and
  `*.nc/*.grib` are regenerated on run. Only code is tracked.

Faithfulness to the official docs was a priority: Tests 1–3 mirror the Microsoft
ERA5 / HRES-0.1° / v1.5 examples (variable sets, pressure levels, static-pickle
handling, `regrid`, `fine_lead_times`).
