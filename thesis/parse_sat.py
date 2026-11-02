"""
Satellite shorelines (CoastSat) on the same day index as the forcing of parse_scalar.py

Input:  a folder of transect CSVs named <site>-<number>.csv (first column: date and time in UTC, second: chainage in m)
Output: data/sat.npy with
    sat_days       [M]     day index (from START) of every image date
    sat_values     [M, P]  chainage (m) of each transect on that date, NaN where the transect has no value
    sat_x          [P]     alongshore position of each transect (m)
    sat_transects  [P]     transect numbers

Day convention, as for the surveys: an image on day a is the shoreline at about OFFSET:00 of day a
(the satellite passes at ~09:45 JST), so forcing days a ... b-1 are what moves it to the image on day b.

Run:  python parse_sat.py [folder]      (default: data/coastsat)
"""

### PLACEHOLDER: done just to test satellite data. If used, I will rewrite it in the future

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import common

### Settings
DATA = common.DATA
START = common.START
END = datetime(2019, 1, 31)                 # last day with forcing, as in parse_scalar.py
N_DAYS = (END - START).days
FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "coastsat"
TIMEZONE = "Asia/Tokyo"                     # images are dated in local time, like the surveys
SPACING = 100                               # distance between neighbouring transects (m)
OUTLIER = None                              # m: a value further than this from the median of the neighbouring images is removed.
                                            # None = keep everything. Storm erosion is exactly what we want to predict, so
                                            # a filter can delete real events: check what it removes before using it.

### Read
if not FOLDER.is_dir():
    raise FileNotFoundError(f"{FOLDER.resolve()} is not a folder: pass it as an argument (python parse_sat.py <folder>)")

series = {}
rows = 0
for path in sorted(p for p in FOLDER.iterdir() if p.suffix.lower() == ".csv"):
    if "summary" in path.name.lower():
        continue
    match = re.search(r"-(\d+)\.csv$", path.name)
    if not match:
        print(f"skipped {path.name}: not named like <site>-<number>.csv")
        continue
    table = pd.read_csv(path)
    date = pd.to_datetime(table.iloc[:, 0], utc=True).dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize()
    day = (date - pd.Timestamp(START)).dt.days.to_numpy()
    value = pd.Series(table.iloc[:, 1].to_numpy(float), index=day)
    rows += len(value)
    series[int(match.group(1))] = value[(value.index >= 0) & (value.index < N_DAYS)].groupby(level=0).mean()   # two images on one day: average
if not series:
    raise FileNotFoundError(f"no transect files found in {FOLDER.resolve()}")

table = pd.DataFrame(series).sort_index().dropna(axis=1, how="all").dropna(axis=0, how="all")
table = table[sorted(table.columns)]

if OUTLIER is not None:
    median = table.rolling(7, center=True, min_periods=3).median()
    removed = (table - median).abs() > OUTLIER
    print(f"outlier filter: {int(removed.to_numpy().sum())} values removed (more than {OUTLIER} m from the local median)")
    table = table.mask(removed)

sat_days = table.index.to_numpy()
sat_values = table.to_numpy()
sat_transects = np.asarray(table.columns)
sat_x = sat_transects * float(SPACING)

### Check
coverage = (~np.isnan(sat_values)).mean(axis=1)
gaps = np.diff(sat_days)
print(f"{len(sat_transects)} transects (numbers {sat_transects[0]} to {sat_transects[-1]}"
      + (f", {sat_transects[-1] - sat_transects[0] + 1 - len(sat_transects)} missing in between" if sat_transects[-1] - sat_transects[0] + 1 != len(sat_transects) else "") + ")")
print(f"{rows} rows read, {len(sat_days)} image dates inside the forcing record "
      f"({(START + pd.Timedelta(days=int(sat_days[0]))):%Y-%m-%d} to {(START + pd.Timedelta(days=int(sat_days[-1]))):%Y-%m-%d})")
print(f"transects with a value per image: median {np.median(coverage):.0%}, "
      f"{(coverage >= 0.5).sum()} images have at least half of them, {(coverage < 0.2).sum()} have less than 20%")
print("days between consecutive images, percentiles 10/25/50/75/90: " + ", ".join(f"{q:.0f}" for q in np.percentile(gaps, [10, 25, 50, 75, 90])))
print(f"images 1 day after the previous one: {(gaps == 1).sum()}; gaps longer than 60 days: {(gaps > 60).sum()}")

np.save(DATA / "sat.npy", dict(sat_days=sat_days, sat_values=sat_values, sat_x=sat_x, sat_transects=sat_transects))
print(f"saved {DATA / 'sat.npy'}")
