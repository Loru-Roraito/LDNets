"""
scalar_lead.py -- how far ahead can the scalar model see?  (+ an interactive viewer)

Needs thesis/data/daily.npy (run build_daily.py first). No TensorFlow. Run:  python scalar_lead.py

Question: the scalar model predicts the largest erosion over the 7 days after a survey from the forcing of those
7 days. If only the first k days of that forcing are known (k = 0..7), how good is the prediction?
  k = 0 : nothing known except the season (sin/cos of the day of the year)  -> climatology
  k = 7 : the whole week known                                              -> the model of scalar_cv.py
This is an UPPER BOUND for a forecast: the first k days are the real, observed forcing (a perfect forecast).
A real wave forecast has errors, so the skill of a real forecast with lead k is lower than the value shown here.

For each k a separate random forest is trained (features: mean of each forcing signal over the first k days, the
peak of hmax over those days, the season). Evaluation is leave-one-year-out; intervals are bootstrap over blocks
of BLOCK_DAYS days (see scalar_cv.py).

Outputs: a table, figures_scalar/lead_curve.png, and a viewer (anchor slider + 'days of forcing known' slider).
"""
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
if 'MPLBACKEND' not in os.environ:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------ settings
START = datetime(1997, 4, 1)
H = 7                      # days between the two surveys
SHIFT = 0                  # forcing day k = 10:00 of day k -> 10:00 of day k+1, so 0 is exact (see build_daily.py)
MIN_WAVE_HOURS = 6
MIN_GOOD_FRAC = 0.7
EMBARGO = 7
EVENT_Q = 0.10             # "extreme" = worst 10% of erosions (a definition, taken over all anchors)
MIN_YEAR_ANCHORS = 30
BLOCK_DAYS = 28
N_BOOT = 500
N_TREES, MIN_LEAF = 200, 20
UNIT = 'cm'
SHOW = True                # open the viewer at the end
ORDER = 'date'             # viewer order: 'date' or 'storm' (strongest week first)

OUT = Path(__file__).resolve().parent / 'figures_scalar'
OUT.mkdir(exist_ok=True)
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

# ------------------------------------------------------------------ data: anchors (same rules as scalar_cv.py)
raw = np.load(Path(__file__).resolve().parents[1] / 'thesis' / 'data' / 'daily.npy', allow_pickle=True).item()
S, FEATURES, wave_hours = raw['S'], list(raw['features']), raw['wave_hours']
fi = {f: i for i, f in enumerate(FEATURES)}
N_DAYS = len(S)
xaxis = raw['x']
sd, first = np.unique(raw['survey_days'], return_index=True)
PROF_ALL = raw['profiles'][first]
prof = {int(d): PROF_ALL[i] for i, d in enumerate(sd)}

SIGNALS = ['hmean', 'hmax', 'flux', 'twl_max', 'steep', 'tmean', 'dirmean', 'tidemean', 'tidemax']
IH = SIGNALS.index('hmax')
Fwin, r_list, da, ang = [], [], [], []
for a in sd:
    a = int(a); b = a + H
    if b not in prof or a < 1 or b + SHIFT > N_DAYS:
        continue
    f = slice(a + SHIFT, b + SHIFT)
    F = S[f]
    if (wave_hours[f] >= MIN_WAVE_HOURS).mean() < MIN_GOOD_FRAC or np.isnan(F).any() or np.isnan(S[a - 1]).any():
        continue
    Fwin.append(F[:, [fi[n] for n in SIGNALS]])                      # [H, n_signals], day 0 = first day after the survey
    r_list.append((prof[b] - prof[a]).min()); da.append(a)
    ang.append(2 * np.pi * (START + timedelta(days=a)).timetuple().tm_yday / 365.25)
Fwin, r, day_a, ang = np.array(Fwin), np.array(r_list), np.array(da), np.array(ang)
N = len(r)
years = np.array([(START + timedelta(days=int(a))).year for a in day_a])
thr = np.quantile(r, EVENT_Q)
ev = r < thr
print(f'{N} anchors ({H}-day pairs) | erosion r: mean {r.mean():.1f}, std {r.std():.1f} {UNIT} | '
      f'extreme = r < {thr:.1f} {UNIT} ({ev.sum()} events)')


def features(k):
    season = np.column_stack([np.sin(ang), np.cos(ang)])
    if k == 0:
        return season
    return np.column_stack([Fwin[:, :k, :].mean(1), Fwin[:, :k, IH].max(1), season])


# ------------------------------------------------------------------ leave-one-year-out for every k
folds = []
for y in sorted(set(years)):
    te = np.flatnonzero(years == y)
    if len(te) < MIN_YEAR_ANCHORS:
        continue
    lo, hi = day_a[te].min(), day_a[te].max() + H
    tr = np.flatnonzero((day_a + H < lo - EMBARGO) | (day_a > hi + EMBARGO))
    folds.append((tr, te))

P = np.full((H + 1, N), np.nan)               # P[k, i] = prediction for anchor i when k days of forcing are known
base = np.full(N, np.nan)
for k in range(H + 1):
    Xk = features(k)
    for tr, te in folds:
        base[te] = r[tr].mean()
        m = RandomForestRegressor(N_TREES, min_samples_leaf=MIN_LEAF, n_jobs=-1, random_state=0)
        P[k, te] = m.fit(Xk[tr], r[tr]).predict(Xk[te])
    print(f'  trained k = {k}')
E = np.flatnonzero(~np.isnan(P[0]))


def metrics(p, rr, e, b):
    return dict(corr=np.corrcoef(p, rr)[0, 1] if np.std(p) > 0 else np.nan,
                skill=1 - ((p - rr) ** 2).sum() / ((b - rr) ** 2).sum(),
                auc=roc_auc_score(e, -p) if 0 < e.sum() < len(e) else np.nan)


blocks = (day_a[E] - day_a[E].min()) // BLOCK_DAYS
members = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
rng = np.random.default_rng(0)
resamples = [np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))]) for _ in range(N_BOOT)]
rE, eE, bE = r[E], ev[E], base[E]
pt = [metrics(P[k, E], rE, eE, bE) for k in range(H + 1)]
bt = [{m: [] for m in ('corr', 'skill', 'auc')} for _ in range(H + 1)]
for idx in resamples:
    if eE[idx].sum() == 0 or eE[idx].sum() == len(idx):
        continue
    for k in range(H + 1):
        for m, v in metrics(P[k, E][idx], rE[idx], eE[idx], bE[idx]).items():
            bt[k][m].append(v)
bt = [{m: np.array(v) for m, v in d.items()} for d in bt]
ci = lambda a: np.nanpercentile(a, [2.5, 97.5])

print(f'\n=== SKILL vs DAYS OF FORCING KNOWN (leave-one-year-out) | {len(E)} pairs, {int(eE.sum())} events')
print('days known |         corr          | skill vs training mean |    AUC worst 10%     | AUC minus full week')
for k in range(H + 1):
    cells = [f"{pt[k][m]:6.3f} [{ci(bt[k][m])[0]:6.3f},{ci(bt[k][m])[1]:6.3f}]" for m in ('corr', 'skill', 'auc')]
    d = bt[k]['auc'] - bt[H]['auc']
    dl, dh = ci(d)
    print(f'{k:6d}     | ' + ' | '.join(cells) + f' | {pt[k]["auc"] - pt[H]["auc"]:+6.3f} [{dl:+6.3f},{dh:+6.3f}]')
print('Read: an interval for "AUC minus full week" that contains 0 means that k days are already as good as the whole '
      'week. These are upper bounds for a forecast (the known days are observed, not forecast).')

# ------------------------------------------------------------------ lead curve figure
fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.15, wspace=0.28)
fig.suptitle('How much of the week must be known to predict its largest erosion?  '
             '(upper bound: the known days are observed, not forecast)', x=0.06, ha='left', fontsize=11.5,
             fontweight='bold')
ks = np.arange(H + 1)
for ax, m, ttl, ref in ((axs[0], 'auc', 'AUC, worst 10%', 0.5), (axs[1], 'skill', 'skill vs training mean', 0.0),
                        (axs[2], 'corr', 'correlation with observed', 0.0)):
    v = np.array([pt[k][m] for k in ks])
    lo = np.array([ci(bt[k][m])[0] for k in ks]); hi = np.array([ci(bt[k][m])[1] for k in ks])
    ax.fill_between(ks, lo, hi, color=BLUE, alpha=0.18, lw=0)
    ax.plot(ks, v, color=BLUE, marker='o', ms=6, mec=SURF, mew=1.5)
    ax.axhline(ref, color=INK2, lw=0.9, ls='--')
    ax.text(H, v[-1], f' {v[-1]:.2f}', va='center', fontsize=8.5, color=INK)
    ax.text(0, v[0], f'{v[0]:.2f} ', va='center', ha='right', fontsize=8.5, color=INK)
    ax.set_xticks(ks); ax.set_xlabel('days of forcing known (0 = season only)')
    ax.set_title(ttl); ax.set_xlim(-0.6, H + 0.7)
fig.savefig(OUT / 'lead_curve.png', dpi=150)
print(f'\nfigure saved in {OUT / "lead_curve.png"}')


# ------------------------------------------------------------------ viewer
def viewer(order=ORDER):
    sel = E[np.argsort(-Fwin[E][:, :, IH].max(1))] if order == 'storm' else E[np.argsort(day_a[E])]
    PA = np.array([prof[int(a)] for a in day_a[sel]])
    PB = np.array([prof[int(a) + H] for a in day_a[sel]])
    ylim_c = (1.08 * (PB - PA).min(), 1.08 * np.percentile((PB - PA).max(1), 99))   # full range: the worst week must fit
    ylim_f = (Fwin[:, :, IH].min(), Fwin[:, :, IH].max() * 1.05)
    ylim_e = (1.1 * r.min(), 0.05 * abs(r.min()))

    fig, ((ax_p, ax_c), (ax_f, ax_k)) = plt.subplots(2, 2, figsize=(14, 8.5))
    fig.subplots_adjust(bottom=0.17, hspace=0.40, wspace=0.24, top=0.90)
    s_win = Slider(fig.add_axes([0.15, 0.07, 0.7, 0.025]), 'week', 0, len(sel) - 1, valinit=0, valstep=1)
    s_k = Slider(fig.add_axes([0.15, 0.025, 0.7, 0.025]), 'days known', 0, H, valinit=H, valstep=1)

    def update(_=None):
        j = int(min(s_win.val, len(sel) - 1)); k = int(s_k.val)
        i = sel[j]; a = int(day_a[i])
        d = PB[j] - PA[j]
        p_k = P[k, i]
        title_ev = '  |  EXTREME week' if ev[i] else ''
        fig.suptitle(f'{START + timedelta(days=a):%Y-%m-%d}  +{H} days  |  observed largest erosion {r[i]:.0f} {UNIT}  |  '
                     f'predicted with {k} day(s) known: {p_k:.0f} {UNIT}{title_ev}', fontsize=11.5, fontweight='bold')

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
        ax_c.axhline(p_k, color=BLUE, ls='--', lw=2, label=f'predicted largest erosion ({k} day(s) known)')
        ax_c.axhline(thr, color=ORANGE, ls=':', lw=1.2, label='extreme threshold')
        ax_c.set_ylim(*ylim_c); ax_c.set_xlim(xaxis.min(), xaxis.max())
        ax_c.set_xlabel('x (m)'); ax_c.set_ylabel(f'change ({UNIT}; below 0 = erosion)')
        ax_c.set_title('Real change and predicted largest erosion'); ax_c.legend(loc='lower right', fontsize=8.5)

        ax_f.cla()
        known = np.arange(H) < k
        ax_f.bar(np.arange(1, H + 1), Fwin[i][:, IH], color=[BLUE if kn else GRID for kn in known], width=0.7)
        ax_f.set_ylim(*ylim_f); ax_f.set_xticks(np.arange(1, H + 1))
        ax_f.set_xlabel('day after the survey'); ax_f.set_ylabel('storm index (log scale)')
        ax_f.set_title('Forcing: known days in blue, unknown in grey')

        ax_k.cla()
        ax_k.plot(ks, P[:, i], color=BLUE, marker='o', ms=5, mec=SURF, mew=1.2)
        ax_k.plot(k, p_k, 'o', color=BLUE, ms=11, mec=INK, mew=1.5)
        ax_k.axhline(r[i], color=INK, ls='--', lw=1.4)
        ax_k.axhline(thr, color=ORANGE, ls=':', lw=1.2)
        ax_k.text(H, r[i], 'observed ', ha='right', va='bottom', fontsize=8.5, color=INK)
        ax_k.set_ylim(*ylim_e); ax_k.set_xticks(ks)
        ax_k.set_xlabel('days of forcing known'); ax_k.set_ylabel(f'largest erosion ({UNIT})')
        ax_k.set_title('Prediction as more of the week becomes known')
        fig.canvas.draw_idle()

    s_win.on_changed(update); s_k.on_changed(update)

    def on_key(e):
        if e.key in ('right', 'left'):
            s_win.set_val(min(max(s_win.val + (1 if e.key == 'right' else -1), 0), len(sel) - 1))
        elif e.key in ('up', 'down'):
            s_k.set_val(min(max(s_k.val + (1 if e.key == 'up' else -1), 0), H))
    fig.canvas.mpl_connect('key_press_event', on_key)
    update()
    fig.sliders = (s_win, s_k)                     # keep references (also lets tests drive the viewer)
    return fig


if __name__ == '__main__':
    fig_v = viewer()
    if SHOW:
        try:
            plt.show()
        except Exception:
            pass
