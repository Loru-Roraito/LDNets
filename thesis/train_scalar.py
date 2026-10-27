"""
How well can the largest erosion over 7 days be predicted from the forcing

Needs data file
 
Target  erosion = largest erosion between two surveys DELTA_DAYS days apart
Model    random forest on forcing summaries of the days between the two surveys (no profile information)
    + optional extra groups
"""

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score

import common
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

### Settings
START = common.START
DELTA = 7              # days between 2 surveys
MIN_WAVES = 6          # minimum number of hours with wave data for a day to be considered valid
MIN_DIRS = 0
MIN_TIDES = 12
MIN_DAYS = 0.6         # minimum number of good days for an anchor to be kept
MIN_ANCHORS = 30       # minimum number of anchors in a year to consider it
BLOCK_DAYS = 28        # how many days in a resampling block
N_BOOT = 500
SKIP = 7               # number of days to skip between training and test
EXTREME_SHARE = 0.1    # percentage of erosions to be considered extreme
FORWARD_YEAR = 2003    # first year with forward chaining (has enough history)
UNIT = "cm"

TEST = True           # check all models
METHOD = "loyo"       # "loyo" or "forward"

N_TREES= 200
MIN_LEAF = 20

DATA = common.DATA
OUTPUT = common.OUTPUT

MEAN_SIGNALS = ["hmean", "fluxmean", "twlmean", "steepmean", "tmean", "dirmean", "tidemean"]
MAX_SIGNALS = ["hmax", "fluxmax", "twlmax", "steepmax", "tmax", "dirmax", "tidemax"]

NAMES = (MEAN_SIGNALS + MAX_SIGNALS + ["ecum_3", "ecum_14", "season_sin", "season_cos", "state_1", "state_2", "state_3"])

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
    def flush(self):
        for stream in self.streams:
            stream.flush()

log = open(OUTPUT / f"results.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log)

### Data import and setup
data = np.load(DATA / "data.npy", allow_pickle=True).item()
feat_values = data["feat_values"]
feat_names = data["feat_names"]
wave_hours = data["wave_hours"]
dir_hours = data["dir_hours"]
tide_hours = data["tide_hours"]

n_days = len(feat_values)
feat_indexes = {feat: index for index, feat in enumerate(feat_names)}

survey_days, indexes = np.unique(data["survey_days"], return_index=True)
profiles = data["profiles"][indexes]
profiles = {int(d): profiles[i] for i, d in enumerate(survey_days)}

total_feats = []
erosions = []
days_a = []
days_b = []
profiles_a = []
for day in survey_days:
    a = int(day)
    b = a + DELTA
    if b not in profiles or a < 1 or b > n_days:
        continue
    f = slice(a, b)
    feats = feat_values[f]
    if ((wave_hours[f] >= MIN_WAVES) * (dir_hours[f] >= MIN_DIRS) * (tide_hours[f] >= MIN_TIDES)).mean() < MIN_DAYS or np.isnan(feats).any() or np.isnan(feat_values[a - 1]).any():
        continue

    season_angle = 2 * np.pi * (START + timedelta(days=a)).timetuple().tm_yday / 365.25
    total_feats.append([feats[:, feat_indexes[n]].mean() for n in MEAN_SIGNALS]
                     + [feats[:, feat_indexes[n]].max() for n in MAX_SIGNALS]
                     + [feat_values[a - 1, feat_indexes["ecum_3"]], feat_values[a - 1, feat_indexes["ecum_14"]],
                        np.sin(season_angle), np.cos(season_angle),
                        0.0, 0.0, 0.0])
    erosions.append((profiles[b] - profiles[a]).min())
    days_a.append(a)
    days_b.append(b)
    profiles_a.append(profiles[a])

total_feats = np.array(total_feats)
erosions = np.array(erosions)
days_a = np.array(days_a)
days_b = np.array(days_b)
profiles_a = np.array(profiles_a)

# TODO: check PCA
# PCA with 3 components
mu = profiles_a.mean(0)
_, _, vt = np.linalg.svd(profiles_a - mu, full_matrices=False)
total_feats[:, -3:] = (profiles_a - mu) @ vt[:3].T

n_days = len(erosions)

dates = np.array([START + timedelta(days=int(day)) for day in days_a])
years = np.array([d.year for d in dates])
months = np.array([d.month for d in dates])
threshold = np.quantile(erosions, EXTREME_SHARE)
events = erosions < threshold

all_columns = {name: index for index, name in enumerate(NAMES)}
print(f"{n_days} anchors ({DELTA}-day pairs) | erosion: mean {erosions.mean():.1f}, std {erosions.std():.1f} {UNIT} | "
      f"extreme : erosion < {threshold:.1f} {UNIT} ({events.sum()} events)")
 
GROUPS = {
    "wave energy":        ["hmean", "hmax", "fluxmean", "fluxmax"],
    "period / steepness": ["steepmean", "steepmax", "tmean", "tmax"],
    "direction":          ["dirmean", "dirmax"],
    "tide / run-up":      ["twlmean", "twlmax", "tidemean", "tidemax"],
    "antecedent waves":   ["ecum_3", "ecum_14"],
    "season":             ["season_sin", "season_cos"],
    "beach state":        ["state_1", "state_2", "state_3"]
}
CORE = ["wave energy", "period / steepness", "direction", "tide / run-up"]
EXTRA = ["antecedent waves", "season", "beach state"]
columns_of = lambda groups: [all_columns[n] for group in groups for n in GROUPS[group]]

### Models
def make_folds(method):
    folds = []
    for y in sorted(set(years)):
        anchors = np.flatnonzero(years == y)
        if len(anchors) < MIN_ANCHORS:
            continue
        first, last = days_a[anchors].min(), days_b[anchors].max()
        if method == "loyo":
            valid = np.flatnonzero((days_b < first - SKIP) | (days_a > last + SKIP))
        else:
            if y < FORWARD_YEAR:
                continue
            valid = np.flatnonzero(days_b < first - SKIP)
            if len(valid) < 300:
                continue
        folds.append((y, valid, anchors))
    return folds

# TODO: check parameters
def predict(folds, kind="rf", columns=None, classifier=False):
    prediction = np.full(n_days, np.nan)
    base = np.full(n_days, np.nan)
    probability = np.full(n_days, np.nan)
    for _, valid, anchor in folds:
        base[anchor] = erosions[valid].mean()
        if kind == "rf":
            m = RandomForestRegressor(N_TREES, min_samples_leaf=MIN_LEAF, n_jobs=-1, random_state=0)
            prediction[anchor] = m.fit(total_feats[valid][:, columns], erosions[valid]).predict(total_feats[anchor][:, columns])
            if classifier:
                c = RandomForestClassifier(N_TREES, min_samples_leaf=MIN_LEAF, n_jobs=-1, random_state=0)
                c.fit(total_feats[valid][:, columns], (erosions[valid] < threshold).astype(int))
                probability[anchor] = c.predict_proba(total_feats[anchor][:, columns])[:, 1]
        elif kind == "linear_peak":
            c = [all_columns["hmax"]]
            prediction[anchor] = LinearRegression().fit(total_feats[valid][:, c], erosions[valid]).predict(total_feats[anchor][:, c])
        elif kind == "month":
            mm = {k: erosions[valid][months[valid] == k].mean() for k in range(1, 13) if (months[valid] == k).any()}
            prediction[anchor] = [mm.get(k, erosions[valid].mean()) for k in months[anchor]]
    return prediction, base, probability

def resample(evaluation_indexes, seed=0):
    blocks = (days_a[evaluation_indexes] - days_a[evaluation_indexes].min()) // BLOCK_DAYS
    members = [np.flatnonzero(blocks == block) for block in np.unique(blocks)]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
            for _ in range(N_BOOT)]

def metrics(prediction, erosion, event, base):
    auc = roc_auc_score(event, -prediction) if 0 < event.sum() < len(event) else np.nan
    return dict(corr=np.corrcoef(prediction, erosion)[0, 1] if np.std(prediction) > 0 else np.nan,
                skill=1 - ((prediction - erosion) ** 2).sum() / ((base - erosion) ** 2).sum(), auc=auc)

def evaluate(predictions, base, evaluation_indexes, reference=None):
    anchors = resample(evaluation_indexes)
    erosion = erosions[evaluation_indexes]
    event = events[evaluation_indexes]
    base = base[evaluation_indexes]

    metric_results = {name: metrics(prediction[evaluation_indexes], erosion, event, base) for name, prediction in predictions.items()}
    scores = {name: {metric: [] for metric in ("corr", "skill", "auc")} for name in predictions}
    for anchor in anchors:
        if event[anchor].sum() == 0 or event[anchor].sum() == len(anchor):
            continue
        for name, prediction in predictions.items():
            for metric, value in metrics(prediction[evaluation_indexes][anchor], erosion[anchor], event[anchor], base[anchor]).items():
                scores[name][metric].append(value)

    output = {}
    for prediction in predictions:
        output[prediction] = {}
        for metric in ("corr", "skill", "auc"):
            a = np.array(scores[prediction][metric])
            lo, hi = np.nanpercentile(a, [2.5, 97.5])
            d = (a - np.array(scores[reference][metric])) if reference and prediction != reference else None
            dlo, dhi = np.nanpercentile(d, [2.5, 97.5]) if d is not None else (np.nan, np.nan)
            output[prediction][metric] = (metric_results[prediction][metric], lo, hi, dlo, dhi)
    return output

def show(title, results, evaluation_indexes, reference=None):
    event_blocks = len(np.unique(((days_a[evaluation_indexes] - days_a[evaluation_indexes].min()) // BLOCK_DAYS)[events[evaluation_indexes]]))
    print(f"\n=== {title} | {len(evaluation_indexes)} pairs, {events[evaluation_indexes].sum()} events in {event_blocks} separate {BLOCK_DAYS}-day blocks")
    print(f'{"model":28s} | {"corr":^21s} | {"Skill vs training mean":^21s} | {"AUC worst 10%":^21s}')
    for index, result in results.items():
        print(f"{index:28s} | " + " | ".join(f"{result[metric][0]:6.3f} [{result[metric][1]:6.3f},{result[metric][2]:6.3f}]" for metric in ("corr", "skill", "auc")))
    if reference:
        print(f'difference with "{reference}":')
        for index, result in results.items():
            if index == reference:
                continue
            print(f"{index:28s} | " + " | ".join(f'{result[metric][0] - results[reference][metric][0]:+6.3f} [{result[metric][3]:+6.3f},{result[metric][4]:+6.3f}]' for metric in ('corr', 'skill', 'auc')))

core_columns = columns_of(CORE)
if TEST:
    # 1) LOYO (leave one year out)
    loyo = make_folds("loyo")
    prediction_loyo, base_loyo, _ = predict(loyo, "rf", core_columns, True)
    prediction_month_loyo, _, _ = predict(loyo, "month")
    prediction_peak, _, _ = predict(loyo, "linear_peak")
    evaluation_indexes_loyo = np.flatnonzero(~np.isnan(prediction_loyo))
    results_loyo = evaluate({"forest on forcing": prediction_loyo, "monthly climatology": prediction_month_loyo, "linear on storm peak": prediction_peak},
        base_loyo, evaluation_indexes_loyo, reference="monthly climatology")
    show("LOYO", results_loyo, evaluation_indexes_loyo, reference="monthly climatology")

    # 2) Forward chaining (train on all previous years, test on the next year)
    forward = make_folds("forward")
    prediction_forward, base_forward, _ = predict(forward, "rf", core_columns)
    prediction_month_forward, _, _ = predict(forward, "month")
    evaluation_indexes_forward = np.flatnonzero(~np.isnan(prediction_forward))
    results_forward = evaluate({"forest on forcing": prediction_forward, "monthly climatology": prediction_month_forward},
        base_forward, evaluation_indexes_forward, reference="monthly climatology")
    show(f"Forward chaining, ({forward[0][0]} onward)", results_forward, evaluation_indexes_forward, reference="monthly climatology")

    # 3) Ablation (LOYO) (removal of features)
    configs = {"Core": CORE}
    for group in CORE:
        configs[f"Drop {group}"] = [g for g in CORE if g != group]
    for group in CORE:
        configs[f"Only {group}"] = [group]
    for group in EXTRA:
        configs[f"Add {group}"] = CORE + [group]
    prediction_ablation = {name: (prediction_loyo if name == "Core" else predict(loyo, "rf", columns_of(groups))[0])
        for name, groups in configs.items()}
    results_ablation = evaluate(prediction_ablation, base_loyo, evaluation_indexes_loyo, reference="Core")
    show("Ablation (LOYO)", results_ablation, evaluation_indexes_loyo, reference="Core")

folds = make_folds(METHOD)
prediction, base, probability = predict(folds, "rf", core_columns, True)
prediction_month, _, _ = predict(folds, "month")
evaluation_indexes = np.flatnonzero(~np.isnan(prediction))
results = evaluate({"forest on forcing": prediction, "monthly climatology": prediction_month}, base, evaluation_indexes, reference="monthly climatology")

### Figures (done with AI, might redo later)
BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#8a8985'
INK, INK2, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#e6e5e1'
plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.edgecolor': GRID, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'legend.frameon': False, 'lines.linewidth': 1.6,
})

def fmt_dates(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

fig = plt.figure(figsize=(16, 9.2))
gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.26, left=0.055, right=0.985, top=0.93, bottom=0.075)
fig.suptitle(f"Largest erosion over 7 days, predicted from forcing ({METHOD})",
    x=0.055, ha='left', fontsize=12, fontweight='bold')

# A) Time series
ax = fig.add_subplot(gs[0, :2])
o = np.argsort(days_a[evaluation_indexes]); T_ = dates[evaluation_indexes][o]
obs, pr = erosions[evaluation_indexes][o].astype(float), prediction[evaluation_indexes][o].astype(float)
gap = np.r_[False, np.diff(days_a[evaluation_indexes][o]) > 14]
obs[gap] = np.nan; pr[gap] = np.nan
ax.plot(T_, obs, color=GREY, lw=0.8, label="Observed")
ax.plot(T_, pr, color=BLUE, lw=1.0, alpha=0.9, label="Predicted")
e_ = events[evaluation_indexes][o]
ax.scatter(T_[e_], erosions[evaluation_indexes][o][e_], s=14, color=ORANGE, zorder=3, label=f"Extreme (worst {int(EXTREME_SHARE * 100)}%)")
ax.axhline(threshold, color=ORANGE, lw=0.8, ls=':')
ax.axvspan(datetime(2008, 1, 1), datetime(2010, 1, 1), color=GRID, alpha=0.7, lw=0)
ax.text(datetime(2009, 1, 1), ax.get_ylim()[1] * 0.97, "No wave\ndata", ha="center", va="top", fontsize=8.5, color=INK2)
ax.axvline(datetime(2011, 6, 1), color=INK2, lw=0.8, ls='--')
ax.text(datetime(2011, 9, 1), ax.get_ylim()[0] * 0.98, "Weekly surveys ->", fontsize=8.5, color=INK2, va="bottom")
ax.set_ylabel(f"Largest erosion ({UNIT}")
ax.set_title("A: observed vs predicted erosion")
ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, -0.09))
fmt_dates(ax)

# B) Per-year AUC
per_year = []
for y, _, te in folds:
    e_ = events[te]
    a_ = roc_auc_score(e_, -prediction[te]) if 0 < e_.sum() < len(te) else np.nan
    c_ = np.corrcoef(prediction[te], erosions[te])[0, 1]
    per_year.append((y, len(te), int(e_.sum()), c_, a_))
ax = fig.add_subplot(gs[0, 2])
yy = [p_[0] for p_ in per_year]; aa = [p_[4] for p_ in per_year]; ne = [p_[2] for p_ in per_year]
ax.bar(range(len(yy)), np.nan_to_num(aa, nan=0.0), color=BLUE, width=0.62)
ax.axhline(0.5, color=INK2, lw=0.9, ls="--")
for i, (a_, n_) in enumerate(zip(aa, ne)):
    ax.text(i, (0 if np.isnan(a_) else a_) + 0.015, f'{n_}', ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(range(len(yy))); ax.set_xticklabels([str(y)[2:] for y in yy], fontsize=8)
ax.set_xlabel("Year (extreme events)")
ax.set_ylim(0, 1.08); ax.set_ylabel(f"AUC, worst {int(EXTREME_SHARE * 100)}%")
ax.set_title("B: skill by year")
ax.grid(axis="x", visible=False)

# C) Predicted vs observed
ax = fig.add_subplot(gs[1, 0])
pe, re_, ee = prediction[evaluation_indexes], erosions[evaluation_indexes], events[evaluation_indexes]
ax.scatter(pe[~ee], re_[~ee], s=8, color=BLUE, alpha=0.30, lw=0, label="Other")
ax.scatter(pe[ee], re_[ee], s=14, color=ORANGE, alpha=0.85, lw=0, label="Extreme")
lim = [min(pe.min(), re_.min()), max(pe.max(), re_.max())]
ax.plot(lim, lim, color=INK2, lw=0.9, ls="--")
ax.axhline(threshold, color=ORANGE, lw=0.8, ls=":")
c_all = results["forest on forcing"]
ax.text(0.97, 0.04, f'Corr: {c_all["corr"][0]:.2f}; skill: {c_all["skill"][0]:.2f}', transform=ax.transAxes,
        ha="right", fontsize=9, color=INK)
ax.set_xlabel(f"Predicted ({UNIT})"); ax.set_ylabel(f"Observed ({UNIT})")
ax.set_title("C: predicted vs observed"); ax.legend(loc="upper left")

# D) Lift by predicted-risk decile
ax = fig.add_subplot(gs[1, 1])
pp = probability[evaluation_indexes]; ee = events[evaluation_indexes]
from scipy.stats import rankdata
dec = (rankdata(pp, method="ordinal") - 1) * 10 // len(pp)         
rate = [ee[dec == i].mean() for i in range(10)]
ax.bar(range(10), rate, color=BLUE, width=0.7)
ax.axhline(ee.mean(), color=INK2, lw=0.9, ls="--")
ax.text(0.2, ee.mean() + 0.01, f"Base rate: {ee.mean():.2f}", fontsize=8.5, color=INK2, va="bottom")
ax.text(9, rate[-1] + 0.01, f'{rate[-1]:.2f}', ha="center", fontsize=8.5, color=INK)
ax.set_xticks(range(10)); ax.set_xticklabels([str(i + 1) for i in range(10)])
ax.set_xlabel("Predicted-risk decile (10 = highest)"); ax.set_ylabel("Observed extreme rate")
ax.set_title("D: high risk score vs extremes?"); ax.grid(axis="x", visible=False)

# E) Response curve
ax = fig.add_subplot(gs[1, 2])
xs = total_feats[evaluation_indexes, all_columns["hmax"]]
qe = np.quantile(xs, np.linspace(0, 1, 9)); qe[-1] += 1e-9
mid, mo, mp, q1, q3 = [], [], [], [], []
for i in range(8):
    m_ = (xs >= qe[i]) & (xs < qe[i + 1])
    mid.append(np.median(xs[m_])); mo.append(np.median(re_[m_])); mp.append(np.median(pe[m_]))
    q1.append(np.percentile(re_[m_], 25)); q3.append(np.percentile(re_[m_], 75))
ax.fill_between(mid, q1, q3, color=GREY, alpha=0.25, lw=0, label="Observed, middle half")
ax.plot(mid, mo, color=GREY, marker="o", ms=5, label="Observed, median")
ax.plot(mid, mp, color=BLUE, marker="o", ms=5, label="Predicted, median")
ax.set_xlabel("Storm peak of the week (log scale)"); ax.set_ylabel(f"Largest erosion ({UNIT})")
ax.set_title("E: response to the strongest wave"); ax.legend(loc="lower left")
path = METHOD + "_overview.png"
fig.savefig(OUTPUT / path, dpi=150)

if TEST:
    # F) Ablation
    fig2, axs = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    fig2.subplots_adjust(left=0.27, right=0.98, top=0.88, bottom=0.11, wspace= 0.08)
    fig2.suptitle(f"Which variables carry the signal? ({METHOD}, 95% intervals)", x=0.02, ha="left",
                fontsize=12, fontweight="bold")
    keys = list(results_ablation)
    yv = np.arange(len(keys))[::-1]
    for ax, m, ttl in ((axs[0], "auc", "AUC, worst 10%"), (axs[1], "skill", "skill vs training mean")):
        for k, y_ in zip(keys, yv):
            pt_, lo, hi = results_ablation[k][m][:3]
            colr = BLUE if k == "Core" else (ORANGE if "Add" in k else INK2)
            ax.plot([lo, hi], [y_, y_], color=colr, lw=2.2, solid_capstyle="round")
            ax.plot(pt_, y_, 'o', color=colr, ms=7, mec=SURF, mew=1.5)
            ax.text(hi + (0.01 if m == "auc" else 0.012), y_, f'{pt_:.2f}', va="center", fontsize=8.5, color=INK,
                    bbox=dict(fc=SURF, ec="none", pad=1))
        ref_ = results_ablation["Core"][m][0]
        ax.axvline(ref_, color=BLUE, lw=0.9, ls=":")
        if m == "auc":
            ax.axvline(0.5, color=INK2, lw=0.9, ls="--")
        ax.set_title(ttl)
    axs[0].set_yticks(yv); axs[0].set_yticklabels([k.strip() for k in keys])
    for lbl, k in zip(axs[0].get_yticklabels(), keys):
        lbl.set_color(INK); lbl.set_fontweight("bold" if k == "Core" else "normal")
    fig2.text(0.27, 0.025, "Blue = forcing groups; grey = dropping/keeping 1 group; Orange = adding 1 group", fontsize=9, color=INK2)
    fig2.savefig(OUTPUT / "ablation.png", dpi=150)

try:
    plt.show()
except Exception:
    pass

sys.stdout = sys.__stdout__
log.close()