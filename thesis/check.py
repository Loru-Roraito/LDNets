"""Does the day-0 profile help to predict the change, and in which form (few points, many points, PCA)?
Window level, random forest, trained on the training years and scored on the validation years.
Needs the data made by the new parse.py (key 'profile0').  Run it next to common.py:  python check_target.py"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import common

raw = np.load(common.DATA_PATH, allow_pickle=True)[()]       # no OUTPUT_MODE check here: works on any parse
if "profile0" not in raw:
    raise SystemExit("no 'profile0' in the data: run the new parse.py first")
train, valid, _ = common.split_windows(raw)
real = np.asarray(raw["real"])
output = raw["output"]                                       # [window, day, point], change from day 0 (signed, abs or cumulative)
signals = raw["sign"]                                        # [window, day, 8]
profile0 = raw["profile0"]                                   # [window, point]

def last_real_day(w):
    days = np.flatnonzero(real[w][1:]) + 1
    return days[-1] if len(days) else 0

last = np.array([last_real_day(w) for w in range(len(real))])
rows = np.arange(len(real))
forcing = np.concatenate([signals.mean(axis=1), signals.max(axis=1), signals.min(axis=1), last[:, None]], axis=1)
cadence = np.stack([real.sum(axis=1), last], axis=1).astype(float)          # number of real surveys, day of the last one

change = output[rows, last]                                  # [window, point], change from day 0 to the last real day
keep = last > 0
tr, va = train[keep[train]], valid[keep[valid]]

def sampled(n):
    return profile0[:, np.linspace(0, profile0.shape[1] - 1, n).astype(int)]

feature_sets = {"survey count only": cadence, "forcing only": forcing}
for n in (3, 5, 11):
    feature_sets[f"forcing + {n} profile points"] = np.concatenate([forcing, sampled(n)], axis=1)
for k in (2, 3, 5):
    pca = PCA(k).fit(profile0[tr])                           # fitted on the training windows only
    feature_sets[f"forcing + {k} PCA scores ({100 * pca.explained_variance_ratio_.sum():.0f}% of variance)"] = \
        np.concatenate([forcing, pca.transform(profile0)], axis=1)

targets = {"magnitude (mean |change| over the profile)": np.abs(change).mean(axis=1)}
if raw.get("output_mode") == "signed":
    targets["net change (mean over the profile)"] = change.mean(axis=1)

for target_name, y in targets.items():
    print(target_name)
    for set_name, features in feature_sets.items():
        scores = []
        for seed in range(3):
            forest = RandomForestRegressor(200, min_samples_leaf=5, random_state=seed, n_jobs=-1).fit(features[tr], y[tr])
            prediction = forest.predict(features[va])
            scores.append(1 - np.sum((y[va] - prediction) ** 2) / np.sum((y[va] - y[tr].mean()) ** 2))
        print(f"  {set_name:48s} R2 on the validation years: {np.mean(scores):6.2f}")
print(f"({len(tr)} train, {len(va)} valid windows; output_mode in the data: {raw.get('output_mode', 'not saved')})")
print("With ~90 validation windows, differences of 0.05 or less are noise. 'survey count only' is the warning light for the cumulative target.")