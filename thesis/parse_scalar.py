"""
Daily forcing features and real survey profiles

Output: thesis/data/data.npy with
    S            [N_DAYS, F]  daily features
    features     [F}          feature names
    survey_days  [M]          day index (from START) of every real survey
    profiles     [M, P]       the surveyed profiles (raw)
    x            [P]          positions

Forcing day k covers OFFSET:00 of day k -> OFFSET:00 of day k+1
"""

import json
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

import common

### Settings
DATA = common.DATA
START = common.START
END = datetime(2019, 1, 31)
OFFSET = 10   # surveys are taken at ~OFFSET:00
N_DAYS = (END - START).days
EXPECTED_WAVES = 12
EXPECTED_DIRS = 12
EXPECTED_TIDES = 24

### Read
def read_hourly(path, columns):
    data = pd.read_csv(path)
    time = pd.to_datetime(data["date"], format="%Y/%m/%d %H:%M:%S")
    hours = ((time - START).dt.total_seconds() / 3600).round().astype(int).to_numpy() - OFFSET
    valid_hours = (hours >= 0) & (hours < N_DAYS * 24)
    output = {}
    for column in columns:
        raw = np.full(N_DAYS * 24, np.nan)
        raw[hours[valid_hours]] = data[column].to_numpy(float)[valid_hours]
        output[column] = raw.reshape(N_DAYS, 24)
    return output

waves = read_hourly(DATA / "wave.csv", ["H", "T", "Dir"])
tides = read_hourly(DATA / "tide.csv", ["tide"])

hs, ts, dirs = waves["H"], waves["T"], waves["Dir"]
tides = tides["tide"]

### Forcing features
h2s = hs ** 2
l0s = 9.81 * ts ** 2 / (2 * np.pi) # Deep water wavelength
steeps = hs / l0s
fluxes = h2s * ts
twls = tides + 0.043 * np.sqrt(hs * l0s) # Total water level: tide + runup (Stockdon"s dissipative limit)

peak = np.where(np.isnan(h2s), -np.inf, h2s).argmax(1)[:, None]
findPeak = lambda a: np.take_along_axis(a, peak, 1)[:, 0]

def create_feat(value):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(value, axis=1)

    
feats = {
    "hmean": create_feat(hs),
    "hmax": findPeak(hs),
    "tmean": create_feat(ts),
    "tmax": findPeak(ts),
    "dirmean": create_feat(dirs),
    "dirmax": findPeak(dirs),
    "tidemean": create_feat(tides),
    "tidemax": findPeak(tides),
    "fluxmean": create_feat(fluxes),
    "fluxmax": findPeak(fluxes),
    "steepmean": create_feat(steeps),
    "steepmax": findPeak(steeps),
    "twlmean": create_feat(twls),
    "twlmax": findPeak(twls)
}

# Fill gaps with interpolation
for name, value in feats.items():
    feats[name] = pd.Series(value).interpolate(limit_direction="both").to_numpy()

# TODO: check EMA validity once more
# Exponential moving average (EMA) with memory tau days to calculate cumulative wave energy
def ema(values, tau):
    lam, acc, out = np.exp(-1 / tau), 0.0, np.zeros_like(values)
    for i, x in enumerate(values):
        acc = lam * acc + (1 - lam) * x
        out[i] = acc
    return out

feats["ecum_3"] = ema(feats["hmean"], 3)
feats["ecum_14"] = ema(feats["hmean"], 14)

valid_waves = ~(np.isnan(hs) | np.isnan(ts))
valid_dirs = ~np.isnan(dirs)
valid_tides = ~np.isnan(tides)
n_waves = valid_waves.sum(1).astype(float)
n_dirs = valid_dirs.sum(1).astype(float)
n_tides = valid_tides.sum(1).astype(float)
wave_hours = n_waves.astype(int)
dir_hours = n_dirs.astype(int)
tide_hours = n_tides.astype(int)

missing_waves = np.clip(1 - n_waves / EXPECTED_WAVES, 0, 1)   
missing_dirs = np.clip(1 - n_dirs / EXPECTED_DIRS, 0, 1)    
missing_tides = np.clip(1 - n_dirs / EXPECTED_TIDES, 0, 1)    
feats["missing_waves"] = missing_waves
feats["missing_dirs"] = missing_dirs
feats["missing_tides"] = missing_tides

# Compress tails
for feat in ["hmean", "hmax", "fluxmean", "fluxmax", "ecum_3", "ecum_14", "twlmean", "twlmax"]:
    feats[feat] = np.log1p(feats[feat])
feats["steepmean"] = np.log(feats["steepmean"] + 1e-3)
feats["steepmax"] = np.log(feats["steepmax"] + 1e-3)

feat_names = list(feats)
feat_values = np.column_stack([feats[feat] for feat in feat_names])

with open(DATA / "coast.json", encoding="utf-8") as file:
    coast = json.load(file)
xaxis = np.array([float(x.rstrip("m")) for x in coast["xaxis"]])
days = np.array([(datetime.strptime(series[0], "%Y-%m-%d") - START).days for series in coast["series"]])
ys = np.array([[float(value) for value in series[1:]] for series in coast["series"]])
valid = (days >= 0) & (days < N_DAYS)
days, ys = days[valid], ys[valid]

np.save(DATA / "data.npy", dict(feat_values=feat_values, feat_names=feat_names, survey_days=days, profiles=ys, x=xaxis,
                                  wave_hours=wave_hours, tide_hours=tide_hours, dir_hours=dir_hours))
print(f'{len(days)} surveys inside the forcing record, {N_DAYS} forcing days, {len(feat_names)} features')