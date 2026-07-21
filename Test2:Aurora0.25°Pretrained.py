"""
Test 2 -- Aurora 0.25 Pretrained (the un-fine-tuned generalist base model).

Role in the experiment
  The baseline. This is the exact checkpoint you would fine-tune from, so its
  off-the-shelf error at the site is the number a local fine-tune must beat. Run
  on the SAME ERA5 input as Test 3, so any Test3 - Test2 gap isolates the value
  of the 1.5 upgrade alone.

Model : AuroraPretrained  (microsoft/aurora : aurora-0.25-pretrained.ckpt)
Grid  : 0.25 deg, 721 x 1440 (~28 km)
Data  : ERA5 (Copernicus CDS)
Step  : 6-hourly
Site/window : Alexander Valley 38.6230932, -122.8856340 ; 7/13-7/16

Run (needs CDS creds + ~40 GB GPU; see aurora_common / era5_data headers):
  python "Test2:Aurora0.25°Pretrained.py" --date 2025-07-13 --init-hour 12 \
         --steps 4 --device cuda
"""

from __future__ import annotations

import argparse
from datetime import datetime

from aurora import AuroraPretrained

import aurora_common as C
from era5_data import build_core_batch, download_era5_core


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=C.INIT_DATE, help="init date YYYY-MM-DD")
    ap.add_argument("--init-hour", type=int, default=C.INIT_HOUR_UTC, choices=[0, 6, 12, 18],
                    help="init hour UTC; history step is 6 h earlier")
    ap.add_argument("--steps", type=int, default=C.HORIZON_HOURS // 6,
                    help="6-hourly steps (default covers the 7/14-7/15 window)")
    ap.add_argument("--half", type=float, default=C.DEFAULT_HALF_DEG)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="test2_pretrained_wind.csv")
    args = ap.parse_args()

    date = datetime.strptime(args.date, "%Y-%m-%d")
    times = [f"{args.init_hour - 6:02d}:00", f"{args.init_hour:02d}:00"]

    print(f"[Test2] ERA5 {args.date} {times} for {C.SITE_NAME}")
    static_p, surf_p, atmos_p = download_era5_core(date, times)
    batch = build_core_batch(static_p, surf_p, atmos_p)

    print("[Test2] loading aurora-0.25-pretrained.ckpt ...")
    model = AuroraPretrained()
    model.load_checkpoint()

    rows = C.run_and_extract(
        model, batch,
        steps=args.steps,
        half=args.half,
        device=args.device,
        label="test2-0.25-pretrained",
    )
    C.write_csv(rows, args.out)


if __name__ == "__main__":
    main()
