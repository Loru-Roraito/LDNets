import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import common

import pandas as pd
import numpy as np

from datetime import datetime
from numpy.lib.stride_tricks import sliding_window_view

DAYS_N = 28 # number of days in a sample
HOUR_N = 24 # number of hours in a day 
WAVE_N = 12 # number of expected wave measurements per day
TIDE_N = 24 # number of expected tide measurements per day
OFFSET = 10 # 10 = windows from 10:00 to 10:00 (aligned with the survey), 0 = calendar days
STRIDE = 7  # stride between windows (1 = daily)
# minimum average values for a window to be valid
MIN_SURVEY = 0.5
MIN_WAVE = 9 / 12 
MIN_PERIOD = 3 / 12  
MIN_DIR =  3 / 12    
MIN_TIDE = 6 / 24
REQUIRE_START = True   # day 0 must be a real survey
# split by blocks of years (the windows overlap, so a random split would leak): remainder of (year % YEAR_BLOCK)
YEAR_BLOCK = 5
VALID_REMAINDER = 1   # years 1991, 1996, 2001, ... -> validation
TEST_REMAINDER = 4    # years 1994, 1999, 2004, ... -> test, all others -> training
EXCLUDE_TOP = 0.01    # training and validation windows with the largest change (top 1%, threshold from the training windows) are left out; 0 = keep all. Test is never filtered.

START_DATE = datetime(1987, 1, 5)
END_DATE = datetime(2024, 12, 24)
N_DAYS = (END_DATE - START_DATE).days + 1

def parse_xaxis(value):
    return float(value.rstrip("m"))

def parse_date(value):
    day = datetime.strptime(value, "%Y-%m-%d")
    return (day - START_DATE).total_seconds() / 3600

with open("thesis/data/coast.json", "r", encoding="utf-8") as f:
    coast_file = json.load(f)

with open("thesis/data/wave.csv", "r", encoding="utf-8") as f:
    wave_file = pd.read_csv(f)

with open("thesis/data/tide.csv", "r", encoding="utf-8") as f:
    tide_file = pd.read_csv(f)

xaxis = [parse_xaxis(x) for x in coast_file["xaxis"]]
xaxis = np.asarray(xaxis, dtype=np.float64)
series = coast_file["series"]

delta = N_DAYS * HOUR_N                            
timesteps = np.arange(N_DAYS, dtype=np.float64) * HOUR_N

coast_dates = np.array([parse_date(s[0]) for s in series])
coast_values = np.array([[float(v) for v in s[1:]] for s in series])
coasts = np.column_stack([np.interp(timesteps, coast_dates, coast_values[:, p]) for p in range(coast_values.shape[1])])
coast_quality = np.isin(timesteps, coast_dates).astype(np.float64)

def hourly_measurements(file, column):
    t = np.rint((pd.to_datetime(file["date"], format="%Y/%m/%d %H:%M:%S") - START_DATE).dt.total_seconds() / 3600).astype(int)
    grid = np.full(delta, np.nan)
    ok = (t >= 0) & (t < delta)
    grid[t[ok]] = file[column].to_numpy(dtype=np.float64)[ok]      # a duplicated timestamp keeps the last record
    if OFFSET:
        grid = np.concatenate([grid[OFFSET:], np.full(OFFSET, np.nan)])
    return grid.reshape(N_DAYS, HOUR_N)

def daily_mean(grid):
    valid = ~np.isnan(grid)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(grid, axis=1) / valid.sum(axis=1) 

hs = hourly_measurements(wave_file, "H") ** 2
ts = hourly_measurements(wave_file, "T")
dirs = np.sin(np.deg2rad(hourly_measurements(wave_file, "Dir")))
tides = hourly_measurements(tide_file, "tide")

# the hour of the largest wave of each day; T, Dir and tide are read at that hour
rows = np.arange(N_DAYS)
imax = np.argmax(np.where(np.isnan(hs), -np.inf, hs), axis=1)
no_h = np.isnan(hs).all(axis=1)

def at_max(grid):
    values = grid[rows, imax]
    values[no_h] = np.nan
    return values

hs_mean, hs_max = daily_mean(hs), at_max(hs)
ts_mean, ts_max = daily_mean(ts), at_max(ts)
dirs_mean, dirs_max = daily_mean(dirs), at_max(dirs)
tides_mean, tides_max = daily_mean(tides), at_max(tides)

h_quality = (~np.isnan(hs)).sum(axis=1) / WAVE_N
t_quality = (~np.isnan(ts)).sum(axis=1) / WAVE_N
dir_quality = (~np.isnan(dirs)).sum(axis=1) / WAVE_N
tide_quality = (~np.isnan(tides)).sum(axis=1) / TIDE_N

signals = np.column_stack([hs_mean, hs_max, ts_mean, ts_max, dirs_mean, dirs_max, tides_mean, tides_max])
sign_names = ["hmean", "hmax", "tmean", "tmax", "dirmean", "dirmax", "tidemean", "tidemax"]
print("signal percentiles (1, 99):")
for name, column in zip(sign_names, signals.T):
    print(f"  {name:9s}", np.round(np.nanpercentile(column, [1, 99]), 2))

#to avoid errors, missing signals are carried from previous values
signals = pd.DataFrame(signals).ffill().bfill().to_numpy()

def windows(array):
    ws = sliding_window_view(array, DAYS_N, axis=0)
    return np.moveaxis(ws, -1, 1)[::STRIDE]   

start_days = np.arange(0, N_DAYS - DAYS_N + 1, STRIDE)

coasts = windows(coasts)                       
profile0 = coasts[:, 0, :].copy()        # full day-0 profile [S, P]; common.py turns it into the PCA parameters          
output = coasts - coasts[:, :1, :]
output = np.abs(output)
signals = windows(signals)      

real = windows(coast_quality) > 0 
checks = {
    "survey": windows(coast_quality).mean(axis=1) >= MIN_SURVEY,
    "wave": windows(h_quality).mean(axis=1) >= MIN_WAVE,
    "period": windows(t_quality).mean(axis=1) >= MIN_PERIOD,
    "dir": windows(dir_quality).mean(axis=1) >= MIN_DIR,
    "tide": windows(tide_quality).mean(axis=1) >= MIN_TIDE,
}
if REQUIRE_START:
    checks["start"] = real[:, 0]
valid = np.logical_and.reduce(list(checks.values()))   

print(f"valid windows: {valid.sum()} of {len(valid)}")
for name, ok in checks.items():
    print(f"  passes {name:7s}: {ok.sum()}")

# ---- dataset split: only windows that pass the quality checks (valid) are used ----
first_day = np.datetime64(START_DATE, "D") + start_days.astype("timedelta64[D]")
last_day = first_day + np.timedelta64(DAYS_N - 1, "D")

def block(day):
    remainder = (day.astype("datetime64[Y]").astype(int) + 1970) % YEAR_BLOCK
    return np.where(remainder == VALID_REMAINDER, 1, np.where(remainder == TEST_REMAINDER, 2, 0))

# 0 = train, 1 = validation, 2 = test, -1 = not used (fails the quality checks, or starts and ends in two different blocks),
# -2 = extreme: a train or validation window with an extreme output (see EXCLUDE_TOP)
split = np.where(valid & (block(first_day) == block(last_day)), block(first_day), -1)
if EXCLUDE_TOP > 0:
    observed_days = real.copy()
    observed_days[:, 0] = False                                                    # day 0 is always 0
    strength = np.where(observed_days, output.mean(axis=2), 0).max(axis=1)         # largest mean change over the profile, on a real survey day
    threshold = np.quantile(strength[split == 0], 1 - EXCLUDE_TOP)                 # from the training windows only
    extreme = (strength > threshold) & np.isin(split, (0, 1))
    print(f"extreme windows: threshold {threshold:.1f} m (mean change over the profile), "
          f"{np.sum(extreme & (split == 0))} train and {np.sum(extreme & (split == 1))} validation windows left out")
    split[extreme] = -2

print(f"split: train {np.sum(split == 0)}, validation {np.sum(split == 1)}, test {np.sum(split == 2)}, "
      f"extreme {np.sum(split == -2)}, not used {np.sum(split == -1)} (of which {np.sum(valid & (split == -1))} are good windows that cross two blocks)")

dataset = {
    "x": xaxis,
    "sign": signals,
    "output": output,
    "real": real,
    "valid": valid,
    "split": split,
    "profile0": profile0,
    "sample_dates": start_days * HOUR_N,
    "t": np.arange(DAYS_N, dtype=np.float64) * HOUR_N
}

print(dataset["x"].shape)
print(dataset["t"].shape)
print(dataset["output"].shape)
print(dataset["sign"].shape)
print(dataset["profile0"].shape)
print(dataset["split"].shape)

observed = real[valid].copy()
observed[:, 0] = False

print("output percentiles (1, 99):", np.round(np.percentile(np.abs(output[valid][observed]), [1, 99]), 1))

np.save("thesis/data/data.npy", dataset)
print("data converted to npy")