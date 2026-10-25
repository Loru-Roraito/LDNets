"""
How well can the largest erosion over 7 days be predicted from the forcing

Needs thesis/data/daily.npy (run build_daily.py first)
 
Target   r = largest erosion between two surveys H days apart
Model    random forest on forcing summaries of the days between the two surveys (no profile information)
    (+ optional extra groups: antecedent wave energy, season, beach state -- tested one at a time)

"""
import os
from datetime import datetime, timedelta
from pathlib import Path
 
import numpy as np
import matplotlib
if 'MPLBACKEND' not in os.environ:          # interactive backend for the viewer (as in visualize.py)
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
 
# ------------------------------------------------------------------ settings
START = datetime(1997, 4, 1)
H = 7                      # days between the two surveys
SHIFT = 0                  # forcing day k = 10:00 of day k -> 10:00 of day k+1 (build_daily.py), so 0 is exact
MIN_WAVE_HOURS = 6         # a forcing day is "good" with at least this many valid wave samples
MIN_GOOD_FRAC = 0.7        # an anchor is kept only if this fraction of its forcing days is good
EMBARGO = 7                # days removed between training data and the tested year
EVENT_Q = 0.10             # "extreme" = worst 10% of erosions (threshold from all anchors; it is a definition)
FIRST_FORWARD_YEAR = 2003  # forward chaining starts here (needs enough earlier data)
MIN_YEAR_ANCHORS = 30
BLOCK_DAYS = 28
N_BOOT = 500
N_TREES, MIN_LEAF = 200, 20
UNIT = 'cm'                # unit of the profile height
SHOW = True                # open the figures / the viewer at the end
ORDER = 'date'             # viewer order: 'date' or 'storm' (strongest week first)
 
OUT = Path(__file__).resolve().parent / 'figures_scalar'
OUT.mkdir(exist_ok=True)
 
# palette: reference categorical slots 1 and 2 (blue, orange), neutral ink and grey
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
 
# ------------------------------------------------------------------ data
raw = np.load(Path(__file__).resolve().parents[1] / 'thesis' / 'data' / 'daily.npy', allow_pickle=True).item()
S, FEATURES, wave_hours = raw['S'], list(raw['features']), raw['wave_hours']
fi = {f: i for i, f in enumerate(FEATURES)}
N_DAYS = len(S)
xaxis = raw['x']                                                  # positions along the profile (m)
sd, first = np.unique(raw['survey_days'], return_index=True)
PROF = raw['profiles'][first]
prof = {int(d): PROF[i] for i, d in enumerate(sd)}
 
MEAN_SIGNALS = ['hmean', 'hmax', 'flux', 'twl_max', 'steep', 'tmean', 'dirmean', 'tidemean', 'tidemax']
NAMES = ([f'{n}_mean' for n in MEAN_SIGNALS] + ['hmax_peak', 'ecum_3_prior', 'ecum_14_prior',
                                                'season_sin', 'season_cos', 'state_1', 'state_2', 'state_3'])
feat_rows, r_list, da, db, Pa = [], [], [], [], []
for a in sd:
    a = int(a); b = a + H
    if b not in prof or a < 1 or b + SHIFT > N_DAYS:
        continue
    f = slice(a + SHIFT, b + SHIFT)
    F = S[f]
    if (wave_hours[f] >= MIN_WAVE_HOURS).mean() < MIN_GOOD_FRAC or np.isnan(F).any() or np.isnan(S[a - 1]).any():
        continue
    ang = 2 * np.pi * (START + timedelta(days=a)).timetuple().tm_yday / 365.25
    feat_rows.append([F[:, fi[n]].mean() for n in MEAN_SIGNALS]
                     + [F[:, fi['hmax']].max(), S[a - 1, fi['ecum_3']], S[a - 1, fi['ecum_14']],
                        np.sin(ang), np.cos(ang), 0.0, 0.0, 0.0])
    r_list.append((prof[b] - prof[a]).min()); da.append(a); db.append(b); Pa.append(prof[a])
X = np.array(feat_rows); r = np.array(r_list); day_a, day_b = np.array(da), np.array(db)
Pa = np.array(Pa)
mu = Pa.mean(0)
_, _, vt = np.linalg.svd(Pa - mu, full_matrices=False)       # unsupervised beach-state modes (no target involved)
X[:, -3:] = (Pa - mu) @ vt[:3].T
N = len(r)
dates = np.array([START + timedelta(days=int(a)) for a in day_a])
years = np.array([d.year for d in dates])
months = np.array([d.month for d in dates])
thr = np.quantile(r, EVENT_Q)
ev = r < thr
col = {n: i for i, n in enumerate(NAMES)}
print(f'{N} anchors ({H}-day pairs) | erosion r: mean {r.mean():.1f}, std {r.std():.1f} {UNIT} | '
      f'extreme = r < {thr:.1f} {UNIT} ({ev.sum()} events)')
 
GROUPS = {
    'wave energy':        ['hmean_mean', 'hmax_mean', 'flux_mean', 'hmax_peak'],
    'period / steepness': ['steep_mean', 'tmean_mean'],
    'direction':          ['dirmean_mean'],
    'tide / run-up':      ['twl_max_mean', 'tidemean_mean', 'tidemax_mean'],
    'antecedent waves':   ['ecum_3_prior', 'ecum_14_prior'],
    'season':             ['season_sin', 'season_cos'],
    'beach state':        ['state_1', 'state_2', 'state_3'],
}
CORE = ['wave energy', 'period / steepness', 'direction', 'tide / run-up']
EXTRA = ['antecedent waves', 'season', 'beach state']
cols_of = lambda gs: [col[n] for g in gs for n in GROUPS[g]]
 
 
# ------------------------------------------------------------------ cross-validation machinery
def make_folds(scheme):
    folds = []
    for y in sorted(set(years)):
        te = np.flatnonzero(years == y)
        if len(te) < MIN_YEAR_ANCHORS:
            continue
        lo, hi = day_a[te].min(), day_b[te].max()
        if scheme == 'loyo':
            tr = np.flatnonzero((day_b < lo - EMBARGO) | (day_a > hi + EMBARGO))
        else:
            if y < FIRST_FORWARD_YEAR:
                continue
            tr = np.flatnonzero(day_b < lo - EMBARGO)
            if len(tr) < 300:
                continue
        folds.append((y, tr, te))
    return folds
 
 
def cv_predict(folds, kind='rf', cols=None, clf=False):
    pred, base, prob = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    for _, tr, te in folds:
        base[te] = r[tr].mean()
        if kind == 'rf':
            m = RandomForestRegressor(N_TREES, min_samples_leaf=MIN_LEAF, n_jobs=-1, random_state=0)
            pred[te] = m.fit(X[tr][:, cols], r[tr]).predict(X[te][:, cols])
            if clf:
                c = RandomForestClassifier(N_TREES, min_samples_leaf=MIN_LEAF, n_jobs=-1, random_state=0)
                c.fit(X[tr][:, cols], (r[tr] < thr).astype(int))
                prob[te] = c.predict_proba(X[te][:, cols])[:, 1]
        elif kind == 'linear_peak':
            c_ = [col['hmax_peak']]
            pred[te] = LinearRegression().fit(X[tr][:, c_], r[tr]).predict(X[te][:, c_])
        elif kind == 'month':
            mm = {k: r[tr][months[tr] == k].mean() for k in range(1, 13) if (months[tr] == k).any()}
            pred[te] = [mm.get(k, r[tr].mean()) for k in months[te]]
    return pred, base, prob
 
 
def metrics(p, rr, e, b):
    auc = roc_auc_score(e, -p) if 0 < e.sum() < len(e) else np.nan
    return dict(corr=np.corrcoef(p, rr)[0, 1] if np.std(p) > 0 else np.nan,
                skill=1 - ((p - rr) ** 2).sum() / ((b - rr) ** 2).sum(), auc=auc)
 
 
def make_resamples(E, seed=0):
    blocks = (day_a[E] - day_a[E].min()) // BLOCK_DAYS
    members = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
            for _ in range(N_BOOT)]
 
 
def evaluate(preds, base, E, ref=None):
    """preds: name -> prediction vector (length N). Returns name -> metric -> (point, lo, hi, dlo, dhi)."""
    rs = make_resamples(E)
    rE, eE, bE = r[E], ev[E], base[E]
    pt = {k: metrics(p[E], rE, eE, bE) for k, p in preds.items()}
    bt = {k: {m: [] for m in ('corr', 'skill', 'auc')} for k in preds}
    for idx in rs:
        if eE[idx].sum() == 0 or eE[idx].sum() == len(idx):
            continue
        for k, p in preds.items():
            for m, v in metrics(p[E][idx], rE[idx], eE[idx], bE[idx]).items():
                bt[k][m].append(v)
    out = {}
    for k in preds:
        out[k] = {}
        for m in ('corr', 'skill', 'auc'):
            a_ = np.array(bt[k][m])
            lo, hi = np.nanpercentile(a_, [2.5, 97.5])
            d = (a_ - np.array(bt[ref][m])) if ref and k != ref else None
            dlo, dhi = np.nanpercentile(d, [2.5, 97.5]) if d is not None else (np.nan, np.nan)
            out[k][m] = (pt[k][m], lo, hi, dlo, dhi)
    return out
 
 
def show_table(title, res, E, ref=None):
    n_blocks = len(np.unique((day_a[E] - day_a[E].min()) // BLOCK_DAYS))
    ev_blocks = len(np.unique(((day_a[E] - day_a[E].min()) // BLOCK_DAYS)[ev[E]]))
    print(f'\n=== {title} | {len(E)} pairs, {ev[E].sum()} events in {ev_blocks} separate {BLOCK_DAYS}-day blocks')
    print(f"{'model':28s} | {'corr':^21s} | {'skill vs training mean':^21s} | {'AUC worst 10%':^21s}")
    for k, v in res.items():
        print(f'{k:28s} | ' + ' | '.join(f'{v[m][0]:6.3f} [{v[m][1]:6.3f},{v[m][2]:6.3f}]' for m in ('corr', 'skill', 'auc')))
    if ref:
        print(f'difference to "{ref}" (interval excluding 0 = a real difference):')
        for k, v in res.items():
            if k == ref:
                continue
            print(f'{k:28s} | ' + ' | '.join(
                f'{v[m][0] - res[ref][m][0]:+6.3f} [{v[m][3]:+6.3f},{v[m][4]:+6.3f}]' for m in ('corr', 'skill', 'auc')))
 
 
# ------------------------------------------------------------------ 1) LOYO: main model, baselines
loyo = make_folds('loyo')
core_cols = cols_of(CORE)
p_main, base_l, prob_main = cv_predict(loyo, 'rf', core_cols, clf=True)
p_month, _, _ = cv_predict(loyo, 'month')
p_lin, _, _ = cv_predict(loyo, 'linear_peak')
E_l = np.flatnonzero(~np.isnan(p_main))
res = evaluate({'forest on forcing': p_main, 'monthly climatology': p_month, 'linear on storm peak': p_lin},
               base_l, E_l, ref='monthly climatology')
show_table('LEAVE-ONE-YEAR-OUT (every year predicted without itself)', res, E_l, ref='monthly climatology')
 
per_year = []
print('\nper year (LOYO):  year  pairs  events   corr    AUC')
for y, _, te in loyo:
    e_ = ev[te]
    a_ = roc_auc_score(e_, -p_main[te]) if 0 < e_.sum() < len(te) else np.nan
    c_ = np.corrcoef(p_main[te], r[te])[0, 1]
    per_year.append((y, len(te), int(e_.sum()), c_, a_))
    print(f'                  {y}  {len(te):5d}  {int(e_.sum()):6d}  {c_:6.2f}  {a_:6.3f}')
 
# ------------------------------------------------------------------ 2) forward chaining
fwd = make_folds('forward')
p_f, base_f, _ = cv_predict(fwd, 'rf', core_cols)
p_fm, _, _ = cv_predict(fwd, 'month')
E_f = np.flatnonzero(~np.isnan(p_f))
show_table(f'FORWARD CHAINING (each year predicted from earlier years only, {fwd[0][0]} onward)',
           evaluate({'forest on forcing': p_f, 'monthly climatology': p_fm}, base_f, E_f, ref='monthly climatology'),
           E_f, ref='monthly climatology')
E_w = E_f[years[E_f] >= 2012]
if len(E_w) > 30:
    show_table('FORWARD CHAINING, weekly-survey era only (2012 onward)',
               evaluate({'forest on forcing': p_f, 'monthly climatology': p_fm}, base_f, E_w, ref='monthly climatology'),
               E_w, ref='monthly climatology')
 
# ------------------------------------------------------------------ 3) ablation (LOYO)
configs = {'core: 4 forcing groups': CORE}
for g in CORE:
    configs[f'  drop {g}'] = [h for h in CORE if h != g]
for g in CORE:
    configs[f'  only {g}'] = [g]
for g in EXTRA:
    configs[f'  core + {g}'] = CORE + [g]
abl_pred = {k: (p_main if k.startswith('core:') else cv_predict(loyo, 'rf', cols_of(gs))[0])
            for k, gs in configs.items()}
abl = evaluate(abl_pred, base_l, E_l, ref='core: 4 forcing groups')
show_table('ABLATION (LOYO): which groups of variables carry the signal', abl, E_l, ref='core: 4 forcing groups')
 
# ------------------------------------------------------------------ figures
def fmt_dates(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
 
 
fig = plt.figure(figsize=(16, 9.2))
gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.26, left=0.055, right=0.985, top=0.93, bottom=0.075)
fig.suptitle('Largest erosion over 7 days, predicted from the forcing alone '
             '(leave-one-year-out, every point predicted without its own year)', x=0.055, ha='left',
             fontsize=12, fontweight='bold')
 
# A: time series
ax = fig.add_subplot(gs[0, :2])
o = np.argsort(day_a[E_l]); T_ = dates[E_l][o]
obs, pr = r[E_l][o].astype(float), p_main[E_l][o].astype(float)
gap = np.r_[False, np.diff(day_a[E_l][o]) > 14]
obs[gap] = np.nan; pr[gap] = np.nan
ax.plot(T_, obs, color=GREY, lw=0.8, label='observed')
ax.plot(T_, pr, color=BLUE, lw=1.0, alpha=0.9, label='predicted')
e_ = ev[E_l][o]
ax.scatter(T_[e_], r[E_l][o][e_], s=14, color=ORANGE, zorder=3, label=f'extreme (worst {int(EVENT_Q * 100)}%)')
ax.axhline(thr, color=ORANGE, lw=0.8, ls=':')
ax.axvspan(datetime(2008, 1, 1), datetime(2010, 1, 1), color=GRID, alpha=0.7, lw=0)
ax.text(datetime(2009, 1, 1), ax.get_ylim()[1] * 0.97, 'no wave\ndata', ha='center', va='top', fontsize=8.5, color=INK2)
ax.axvline(datetime(2011, 6, 1), color=INK2, lw=0.8, ls='--')
ax.text(datetime(2011, 9, 1), ax.get_ylim()[0] * 0.98, 'weekly surveys →', fontsize=8.5, color=INK2, va='bottom')
ax.set_ylabel(f'largest erosion ({UNIT}; lower = more)')
ax.set_title('A  Observed and predicted erosion through time')
ax.legend(loc='upper left', ncol=3, bbox_to_anchor=(0.0, -0.09))
fmt_dates(ax)
 
# B: per-year AUC
ax = fig.add_subplot(gs[0, 2])
yy = [p_[0] for p_ in per_year]; aa = [p_[4] for p_ in per_year]; ne = [p_[2] for p_ in per_year]
ax.bar(range(len(yy)), np.nan_to_num(aa, nan=0.0), color=BLUE, width=0.62)
ax.axhline(0.5, color=INK2, lw=0.9, ls='--')
for i, (a_, n_) in enumerate(zip(aa, ne)):
    ax.text(i, (0 if np.isnan(a_) else a_) + 0.015, f'{n_}', ha='center', va='bottom', fontsize=7.5, color=INK2)
ax.set_xticks(range(len(yy))); ax.set_xticklabels([str(y)[2:] for y in yy], fontsize=8)
ax.set_xlabel('year (numbers above bars = extreme events)')
ax.set_ylim(0, 1.08); ax.set_ylabel('AUC, worst 10%')
ax.set_title('B  Skill by year (dashed = no skill)')
ax.grid(axis='x', visible=False)
 
# C: predicted vs observed
ax = fig.add_subplot(gs[1, 0])
pe, re_, ee = p_main[E_l], r[E_l], ev[E_l]
ax.scatter(pe[~ee], re_[~ee], s=8, color=BLUE, alpha=0.30, lw=0, label='other')
ax.scatter(pe[ee], re_[ee], s=14, color=ORANGE, alpha=0.85, lw=0, label='extreme')
lim = [min(pe.min(), re_.min()), max(pe.max(), re_.max())]
ax.plot(lim, lim, color=INK2, lw=0.9, ls='--')
ax.axhline(thr, color=ORANGE, lw=0.8, ls=':')
c_all = res['forest on forcing']
ax.text(0.97, 0.04, f"corr {c_all['corr'][0]:.2f}   skill {c_all['skill'][0]:.2f}", transform=ax.transAxes,
        ha='right', fontsize=9, color=INK)
ax.set_xlabel(f'predicted ({UNIT})'); ax.set_ylabel(f'observed ({UNIT})')
ax.set_title('C  Predicted vs observed'); ax.legend(loc='upper left')
 
# D: lift by predicted-risk decile
ax = fig.add_subplot(gs[1, 1])
pp = prob_main[E_l]; ee = ev[E_l]
from scipy.stats import rankdata
dec = (rankdata(pp, method='ordinal') - 1) * 10 // len(pp)           # equal-sized risk deciles (ties broken by order)
rate = [ee[dec == i].mean() for i in range(10)]
ax.bar(range(10), rate, color=BLUE, width=0.7)
ax.axhline(ee.mean(), color=INK2, lw=0.9, ls='--')
ax.text(0.2, ee.mean() + 0.01, f'base rate {ee.mean():.2f}', fontsize=8.5, color=INK2, va='bottom')
ax.text(9, rate[-1] + 0.01, f'{rate[-1]:.2f}', ha='center', fontsize=8.5, color=INK)
ax.set_xticks(range(10)); ax.set_xticklabels([str(i + 1) for i in range(10)])
ax.set_xlabel('predicted-risk decile (10 = highest)'); ax.set_ylabel('observed extreme rate')
ax.set_title('D  Does a high risk score mean more extremes?'); ax.grid(axis='x', visible=False)
 
# E: response curve
ax = fig.add_subplot(gs[1, 2])
xs = X[E_l, col['hmax_peak']]
qe = np.quantile(xs, np.linspace(0, 1, 9)); qe[-1] += 1e-9
mid, mo, mp, q1, q3 = [], [], [], [], []
for i in range(8):
    m_ = (xs >= qe[i]) & (xs < qe[i + 1])
    mid.append(np.median(xs[m_])); mo.append(np.median(re_[m_])); mp.append(np.median(pe[m_]))
    q1.append(np.percentile(re_[m_], 25)); q3.append(np.percentile(re_[m_], 75))
ax.fill_between(mid, q1, q3, color=GREY, alpha=0.25, lw=0, label='observed, middle half')
ax.plot(mid, mo, color=GREY, marker='o', ms=5, label='observed, median')
ax.plot(mid, mp, color=BLUE, marker='o', ms=5, label='predicted, median')
ax.set_xlabel('storm peak of the week (log scale)'); ax.set_ylabel(f'largest erosion ({UNIT})')
ax.set_title('E  Response to the strongest wave of the week'); ax.legend(loc='lower left')
fig.savefig(OUT / 'overview.png', dpi=150)
 
# F: ablation
fig2, axs = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
fig2.subplots_adjust(left=0.27, right=0.98, top=0.88, bottom=0.11, wspace=0.08)
fig2.suptitle('Which variables carry the signal? (leave-one-year-out, 95% intervals)', x=0.02, ha='left',
              fontsize=12, fontweight='bold')
keys = list(abl)
yv = np.arange(len(keys))[::-1]
for ax, m, ttl in ((axs[0], 'auc', 'AUC, worst 10% (0.5 = no skill)'), (axs[1], 'skill', 'skill vs training mean')):
    for k, y_ in zip(keys, yv):
        pt_, lo, hi = abl[k][m][:3]
        colr = BLUE if k.startswith('core:') else (ORANGE if 'core +' in k else INK2)
        ax.plot([lo, hi], [y_, y_], color=colr, lw=2.2, solid_capstyle='round')
        ax.plot(pt_, y_, 'o', color=colr, ms=7, mec=SURF, mew=1.5)
        ax.text(hi + (0.01 if m == 'auc' else 0.012), y_, f'{pt_:.2f}', va='center', fontsize=8.5, color=INK,
                bbox=dict(fc=SURF, ec='none', pad=1))
    ref_ = abl['core: 4 forcing groups'][m][0]
    ax.axvline(ref_, color=BLUE, lw=0.9, ls=':')
    if m == 'auc':
        ax.axvline(0.5, color=INK2, lw=0.9, ls='--')
    ax.set_title(ttl)
axs[0].set_yticks(yv); axs[0].set_yticklabels([k.strip() for k in keys])
for lbl, k in zip(axs[0].get_yticklabels(), keys):
    lbl.set_color(INK); lbl.set_fontweight('bold' if k.startswith('core:') else 'normal')
fig2.text(0.27, 0.025, 'blue = the 4 forcing groups together;  grey = dropping or keeping only one group;  '
          'orange = adding an extra group to the core', fontsize=9, color=INK2)
fig2.savefig(OUT / 'ablation.png', dpi=150)
 
# ------------------------------------------------------------------ viewer: slide through the weeks and the models
MODELS = {'forest on forcing': p_main, 'monthly climatology': p_month, 'linear on storm peak': p_lin}
MODELS.update({k.strip(): v for k, v in abl_pred.items() if not k.startswith('core:')})
 
 
def viewer(order=ORDER):
    sel = E_l[np.argsort(-X[E_l, col['hmax_peak']])] if order == 'storm' else E_l[np.argsort(day_a[E_l])]
    PA = np.array([prof[int(day_a[i])] for i in sel])
    PB = np.array([prof[int(day_b[i])] for i in sel])
    names = list(MODELS)
    ylim_c = (1.08 * r.min(), 1.08 * np.percentile((PB - PA).max(1), 99))
    ylim_f = (S[:, fi['hmax']].min(), np.nanmax(S[:, fi['hmax']]) * 1.05)
    ylim_e = (1.08 * r.min(), 0.05 * abs(r.min()))
 
    fig, ((ax_p, ax_c), (ax_f, ax_m)) = plt.subplots(2, 2, figsize=(14, 8.5))
    fig.subplots_adjust(bottom=0.17, hspace=0.40, wspace=0.52, top=0.90, left=0.07, right=0.97)
    s_win = Slider(fig.add_axes([0.15, 0.07, 0.7, 0.025]), 'week', 0, len(sel) - 1, valinit=0, valstep=1)
    s_mod = Slider(fig.add_axes([0.15, 0.025, 0.7, 0.025]), 'model', 0, len(names) - 1, valinit=0, valstep=1)
 
    def update(_=None):
        j = int(min(s_win.val, len(sel) - 1)); m = int(s_mod.val)
        i = sel[j]; a = int(day_a[i]); name = names[m]
        d = PB[j] - PA[j]
        p_m = MODELS[name][i]
        fig.suptitle(f'{dates[i]:%Y-%m-%d}  +{H} days  |  observed largest erosion {r[i]:.0f} {UNIT}  |  '
                     f'{name}: {p_m:.0f} {UNIT}{"  |  EXTREME week" if ev[i] else ""}', fontsize=11.5, fontweight='bold')
 
        ax_p.cla()
        ax_p.plot(xaxis, PA[j], color=GREY, ls='--', label='survey, start')
        ax_p.plot(xaxis, PB[j], color=INK, label=f'survey, +{H} days')
        lo_, hi_ = min(PA[j].min(), PB[j].min()), max(PA[j].max(), PB[j].max())
        ax_p.set_ylim(lo_ - 0.1 * (hi_ - lo_), hi_ + 0.1 * (hi_ - lo_)); ax_p.set_xlim(xaxis.min(), xaxis.max())
        ax_p.set_xlabel('x (m)'); ax_p.set_ylabel(f'height ({UNIT})'); ax_p.set_title('Real profiles')
        ax_p.legend(loc='lower right', fontsize=8.5)
 
        ax_c.cla()
        ax_c.axhline(0, color=GREY, lw=0.8)
        ax_c.plot(xaxis, d, color=INK, label='real change')
        ax_c.plot(xaxis[np.argmin(d)], d.min(), 'o', color=INK, ms=7)
        ax_c.axhline(p_m, color=BLUE, ls='--', lw=2, label=f'predicted largest erosion ({name})')
        ax_c.axhline(thr, color=ORANGE, ls=':', lw=1.2, label='extreme threshold')
        ax_c.set_ylim(*ylim_c); ax_c.set_xlim(xaxis.min(), xaxis.max())
        ax_c.set_xlabel('x (m)'); ax_c.set_ylabel(f'change ({UNIT}; below 0 = erosion)')
        ax_c.set_title('Real change and predicted largest erosion'); ax_c.legend(loc='lower right', fontsize=8.5)
 
        ax_f.cla()
        ax_f.bar(np.arange(1, H + 1), S[a + SHIFT:a + H + SHIFT, fi['hmax']], color=BLUE, width=0.7)
        ax_f.set_ylim(*ylim_f); ax_f.set_xticks(np.arange(1, H + 1))
        ax_f.set_xlabel('day after the survey'); ax_f.set_ylabel('storm index (log scale)')
        ax_f.set_title('Forcing of the week (daily maximum wave energy)')
 
        ax_m.cla()
        yv = np.arange(len(names))[::-1]
        for n_, y_ in zip(names, yv):
            sel_ = n_ == name
            ax_m.plot(MODELS[n_][i], y_, 'o', color=BLUE, ms=11 if sel_ else 6, mec=INK if sel_ else SURF, mew=1.5)
        ax_m.axvline(r[i], color=INK, ls='--', lw=1.4)
        ax_m.axvline(thr, color=ORANGE, ls=':', lw=1.2)
        ax_m.text(r[i], len(names) - 0.35, ' observed', ha='left', va='bottom', fontsize=8.5, color=INK)
        ax_m.set_yticks(yv); ax_m.set_yticklabels(names, fontsize=8.5)
        ax_m.set_xlim(*ylim_e); ax_m.set_ylim(-0.7, len(names) - 0.2)
        ax_m.set_xlabel(f'predicted largest erosion ({UNIT})')
        ax_m.set_title('Every model on this week (selected: large dot)')
        fig.canvas.draw_idle()
 
    s_win.on_changed(update); s_mod.on_changed(update)
 
    def on_key(e):
        if e.key in ('right', 'left'):
            s_win.set_val(min(max(s_win.val + (1 if e.key == 'right' else -1), 0), len(sel) - 1))
        elif e.key in ('up', 'down'):
            s_mod.set_val(min(max(s_mod.val + (1 if e.key == 'up' else -1), 0), len(names) - 1))
    fig.canvas.mpl_connect('key_press_event', on_key)
    update()
    fig.sliders = (s_win, s_mod)
    return fig
 
 
fig_v = viewer()
 
print(f'\nfigures saved in {OUT}')
if SHOW:
    try:
        plt.show()
    except Exception:
        pass