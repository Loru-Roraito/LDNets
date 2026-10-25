"""
Daily forcing features and the real survey profiles

Output: thesis/data/daily.npy with
    S            [N_DAYS, F]  daily features
    features     list of F names
    survey_days  [M]          day index (from START) of every real survey
    profiles     [M, P]       the surveyed profiles (raw)
    x            [P]          positions

Forcing day k covers OFFSET_H:00 of day k -> OFFSET_H:00 of day k+1
"""
import json, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'thesis' / 'data'

START, END = datetime(1997, 4, 1), datetime(2019, 1, 31)
OFFSET_H = 10   # surveys are taken at ~10:00: forcing 'day' k = from 10:00 of day k to 10:00 of day k+1
N_DAYS = (END - START).days
G = 9.81

def read_hourly(path, cols):
    df = pd.read_csv(path)
    t = pd.to_datetime(df['date'], format='%Y/%m/%d %H:%M:%S')
    h = ((t - START).dt.total_seconds() / 3600).round().astype(int).to_numpy() - OFFSET_H
    ok = (h >= 0) & (h < N_DAYS * 24)
    out = {}
    for c in cols:
        g = np.full(N_DAYS * 24, np.nan)
        g[h[ok]] = df[c].to_numpy(float)[ok]
        out[c] = g.reshape(N_DAYS, 24)             # [days, 24 hours]
    return out

w = read_hourly(DATA / 'wave.csv', ['H', 'T', 'Dir'])
td_raw = read_hourly(DATA / 'tide.csv', ['tide'])['tide']
Hs, Tp, Dir = w['H'], w['T'], w['Dir']

# --- tide: fill the gaps with a harmonic fit (the tide is almost purely astronomical).
# Surge is NOT in the fitted part, so filled periods miss the meteorological component.
hours = np.arange(N_DAYS * 24).astype(float)
flat = td_raw.reshape(-1)
ok_t = ~np.isnan(flat)
periods = [12.4206, 12.0, 12.6583, 11.9672, 25.8193, 23.9345, 26.8684, 24.0659]   # M2 S2 N2 K2 O1 K1 Q1 P1 (h)

def design(h):
    cols = [np.ones_like(h)]
    for P in periods:
        wv = 2 * np.pi / P
        cols += [np.cos(wv * h), np.sin(wv * h)]
    return np.column_stack(cols)

cut = np.flatnonzero(ok_t)[int(0.7 * ok_t.sum())]              # quick out-of-sample check of the fit
tr_i = ok_t & (hours < cut)
c_chk, *_ = np.linalg.lstsq(design(hours[tr_i] + OFFSET_H), flat[tr_i], rcond=None)
te_i = ok_t & (hours >= cut)
print(f'tide harmonic fit: out-of-sample RMSE {np.sqrt(np.mean((design(hours[te_i] + OFFSET_H) @ c_chk - flat[te_i]) ** 2)):.3f} '
      f'(std of the tide {np.nanstd(flat):.3f})')
coef, *_ = np.linalg.lstsq(design(hours[ok_t] + OFFSET_H), flat[ok_t], rcond=None)
td = np.where(ok_t, flat, design(hours + OFFSET_H) @ coef).reshape(N_DAYS, 24)

# --- coverage per year (to see where the forcing record has holes)
valid_w = ~(np.isnan(Hs) | np.isnan(Tp) | np.isnan(Dir))
valid_t = ~np.isnan(td_raw)
year_of_day = np.array([(START.year + (START.month - 1) / 12 + d / 365.25) for d in range(N_DAYS)]).astype(int)
n_w = valid_w.sum(1).astype(float)                             # valid wave samples per day
exp_w = (pd.Series(np.where(n_w > 0, n_w, np.nan))             # usual samples per day around each date
         .rolling(91, center=True, min_periods=10).median().bfill().ffill().to_numpy())
wave_hours = n_w.astype(int)                                   # valid wave samples per day (saved for the anchor filter)
print('year | waves: hours valid, samples/day (median), days with >=6 samples, days with no wave data | tide: hours valid')
for yr in sorted(set(year_of_day)):
    m_ = year_of_day == yr
    spd = np.median(n_w[m_][n_w[m_] > 0]) if (n_w[m_] > 0).any() else 0
    print(f'{yr} | {valid_w[m_].mean():10.2f} {spd:10.0f} {np.mean(n_w[m_] >= 6):10.2f} {np.mean(n_w[m_] == 0):10.2f} | {valid_t[m_].mean():10.2f}')

H2 = Hs ** 2
L0 = G * Tp ** 2 / (2 * np.pi)                     # deep-water wavelength
steep = Hs / L0
flux = H2 * Tp                                     # proportional to the wave energy flux
twl = td + 0.043 * np.sqrt(Hs * L0)                # tide + runup (dissipative limit of Stockdon)

def stat(f, a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return f(a, axis=1)

k = np.where(np.isnan(H2), -np.inf, H2).argmax(1)[:, None]
at_peak = lambda a: np.take_along_axis(a, k, 1)[:, 0]

feat = {
    'hmean': stat(np.nanmean, H2),  'hmax': stat(np.nanmax, H2),
    'tmean': stat(np.nanmean, Tp),  'tmax': at_peak(Tp),
    'dirmean': stat(np.nanmean, Dir), 'dirmax': at_peak(Dir),
    'tidemean': stat(np.nanmean, td), 'tidemax': at_peak(td),
    'flux': stat(np.nanmean, flux), 'steep': stat(np.nanmean, steep),
    'twl_max': stat(np.nanmax, twl),
}
miss_w = np.clip(1 - n_w / exp_w, 0, 1)             # fraction of the usual wave samples that are missing (cadence-aware)
miss_t = 1 - valid_t.mean(1)                       # fraction of hours where the tide is filled, not measured

for name, v in feat.items():                       # days with no wave data: interpolate between valid days
    feat[name] = pd.Series(v).interpolate(limit_direction='both').to_numpy()

def ema(v, tau):                                   # cumulative energy with memory tau days
    lam, acc, out = np.exp(-1 / tau), 0.0, np.zeros_like(v)
    for i, x in enumerate(v):
        acc = lam * acc + (1 - lam) * x
        out[i] = acc
    return out

feat['ecum_3'] = ema(feat['hmean'], 3)
feat['ecum_14'] = ema(feat['hmean'], 14)
feat['miss_w'] = miss_w
feat['miss_t'] = miss_t

for f in ['hmean', 'hmax', 'flux', 'ecum_3', 'ecum_14', 'twl_max']:   # skewed features
    feat[f] = np.log1p(feat[f])
feat['steep'] = np.log(feat['steep'] + 1e-3)

FEATURES = list(feat)
S = np.column_stack([feat[f] for f in FEATURES])   # [N_DAYS, F]

with open(DATA / 'coast.json', encoding='utf-8') as f:
    coast = json.load(f)
xaxis = np.array([float(x.rstrip('m')) for x in coast['xaxis']])
sd = np.array([(datetime.strptime(s[0], '%Y-%m-%d') - START).days for s in coast['series']])
Y = np.array([[float(v) for v in s[1:]] for s in coast['series']])
keep = (sd >= 0) & (sd < N_DAYS)
sd, Y = sd[keep], Y[keep]

np.save(DATA / 'daily.npy', dict(S=S, features=FEATURES, survey_days=sd, profiles=Y, x=xaxis,
                                  wave_hours=wave_hours))
print(f'{len(sd)} surveys inside the forcing record, {N_DAYS} forcing days, {len(FEATURES)} features')
import json, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'thesis' / 'data'

START, END = datetime(1997, 4, 1), datetime(2019, 1, 31) # Only period with tide data
OFFSET_H = 10   # surveys are taken at ~10:00
N_DAYS = (END - START).days
G = 9.81

def read_hourly(path, cols):
    df = pd.read_csv(path)
    t = pd.to_datetime(df['date'], format='%Y/%m/%d %H:%M:%S')
    h = ((t - START).dt.total_seconds() / 3600).round().astype(int).to_numpy() - OFFSET_H
    ok = (h >= 0) & (h < N_DAYS * 24)
    out = {}
    for c in cols:
        g = np.full(N_DAYS * 24, np.nan)
        g[h[ok]] = df[c].to_numpy(float)[ok]
        out[c] = g.reshape(N_DAYS, 24)
    return out

w = read_hourly(DATA / 'wave.csv', ['H', 'T', 'Dir'])
td_raw = read_hourly(DATA / 'tide.csv', ['tide'])['tide']
Hs, Tp, Dir = w['H'], w['T'], w['Dir']

# Tide. Gaps are fille with an armonic fit, that misses the outliers
hours = np.arange(N_DAYS * 24).astype(float)
flat = td_raw.reshape(-1)
ok_t = ~np.isnan(flat)
periods = [12.4206, 12.0, 12.6583, 11.9672, 25.8193, 23.9345, 26.8684, 24.0659]   # Periods of the main tidal components: M2 S2 N2 K2 O1 K1 Q1 P1 (h)

def design(h):
    cols = [np.ones_like(h)]
    for P in periods:
        wv = 2 * np.pi / P
        cols += [np.cos(wv * h), np.sin(wv * h)]
    return np.column_stack(cols)

cut = np.flatnonzero(ok_t)[int(0.7 * ok_t.sum())]              # check to see if the fit is valid
tr_i = ok_t & (hours < cut)
c_chk, *_ = np.linalg.lstsq(design(hours[tr_i] + OFFSET_H), flat[tr_i], rcond=None)
te_i = ok_t & (hours >= cut)
print(f'tide harmonic fit: out-of-sample RMSE {np.sqrt(np.mean((design(hours[te_i] + OFFSET_H) @ c_chk - flat[te_i]) ** 2)):.3f} '
      f'(std of the tide {np.nanstd(flat):.3f})')
coef, *_ = np.linalg.lstsq(design(hours[ok_t] + OFFSET_H), flat[ok_t], rcond=None)
td = np.where(ok_t, flat, design(hours + OFFSET_H) @ coef).reshape(N_DAYS, 24)

# Check the coverage of wave and tide data
valid_w = ~(np.isnan(Hs) | np.isnan(Tp) | np.isnan(Dir))
valid_t = ~np.isnan(td_raw)
year_of_day = np.array([(START.year + (START.month - 1) / 12 + d / 365.25) for d in range(N_DAYS)]).astype(int)
n_w = valid_w.sum(1).astype(float)                             # valid wave samples per day
exp_w = (pd.Series(np.where(n_w > 0, n_w, np.nan))             # usual samples per day around each date
         .rolling(91, center=True, min_periods=10).median().bfill().ffill().to_numpy())
wave_hours = n_w.astype(int)                                   # valid wave samples per day (saved for the anchor filter)
print('year | waves: hours valid, samples/day (median), days with >=6 samples, days with no wave data | tide: hours valid')
for yr in sorted(set(year_of_day)): 
    m_ = year_of_day == yr
    spd = np.median(n_w[m_][n_w[m_] > 0]) if (n_w[m_] > 0).any() else 0
    print(f'{yr} | {valid_w[m_].mean():10.2f} {spd:10.0f} {np.mean(n_w[m_] >= 6):10.2f} {np.mean(n_w[m_] == 0):10.2f} | {valid_t[m_].mean():10.2f}')

H2 = Hs ** 2
L0 = G * Tp ** 2 / (2 * np.pi)                     # deep-water wavelength
steep = Hs / L0
flux = H2 * Tp                                     # proportional to the wave energy flux
twl = td + 0.043 * np.sqrt(Hs * L0)                # tide + runup (dissipative limit of Stockton)

def stat(f, a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return f(a, axis=1)

k = np.where(np.isnan(H2), -np.inf, H2).argmax(1)[:, None]
at_peak = lambda a: np.take_along_axis(a, k, 1)[:, 0]

feat = {
    'hmean': stat(np.nanmean, H2),  'hmax': stat(np.nanmax, H2),
    'tmean': stat(np.nanmean, Tp),  'tmax': at_peak(Tp),
    'dirmean': stat(np.nanmean, Dir), 'dirmax': at_peak(Dir),
    'tidemean': stat(np.nanmean, td), 'tidemax': at_peak(td),
    'flux': stat(np.nanmean, flux), 'steep': stat(np.nanmean, steep),
    'twl_max': stat(np.nanmax, twl),
}
miss_w = np.clip(1 - n_w / exp_w, 0, 1)             # fraction of the usual wave samples that are missing (cadence-aware)
miss_t = 1 - valid_t.mean(1)                       # fraction of hours where the tide is filled, not measured

for name, v in feat.items():                       # days with no wave data: interpolate between valid days
    feat[name] = pd.Series(v).interpolate(limit_direction='both').to_numpy()

def ema(v, tau):                                   # cumulative energy with memory tau days
    lam, acc, out = np.exp(-1 / tau), 0.0, np.zeros_like(v)
    for i, x in enumerate(v):
        acc = lam * acc + (1 - lam) * x
        out[i] = acc
    return out

feat['ecum_3'] = ema(feat['hmean'], 3)
feat['ecum_14'] = ema(feat['hmean'], 14)
feat['miss_w'] = miss_w
feat['miss_t'] = miss_t

for f in ['hmean', 'hmax', 'flux', 'ecum_3', 'ecum_14', 'twl_max']:   # skewed features
    feat[f] = np.log1p(feat[f])
feat['steep'] = np.log(feat['steep'] + 1e-3)

FEATURES = list(feat)
S = np.column_stack([feat[f] for f in FEATURES])   # [N_DAYS, F]

with open(DATA / 'coast.json', encoding='utf-8') as f:
    coast = json.load(f)
xaxis = np.array([float(x.rstrip('m')) for x in coast['xaxis']])
sd = np.array([(datetime.strptime(s[0], '%Y-%m-%d') - START).days for s in coast['series']])
Y = np.array([[float(v) for v in s[1:]] for s in coast['series']])
keep = (sd >= 0) & (sd < N_DAYS)
sd, Y = sd[keep], Y[keep]

np.save(DATA / 'daily.npy', dict(S=S, features=FEATURES, survey_days=sd, profiles=Y, x=xaxis,
                                  wave_hours=wave_hours))
print(f'{len(sd)} surveys inside the forcing record, {N_DAYS} forcing days, {len(FEATURES)} features')
