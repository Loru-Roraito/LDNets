# %%
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import matplotlib
matplotlib.use(os.environ.get("LDNETS_BACKEND", "TkAgg"))
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import common
import numpy as np
from scipy import stats
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

reconstruction = True

# %%
raw = common.load_raw()
dataset_train, dataset_valid, dataset_tests = common.load_datasets(raw)
model = common.LDNetModel.load(
    common.problem,
    common.normalization,
    common.num_latent_states,
    common.dt,
    common.dt_base,
    common.MODEL_DIR,
)

# %%
history = np.load(common.HISTORY_PATH)
iterations_history = history['iterations_history']
loss_train_history = history['loss_train_history']
loss_valid_history = history['loss_valid_history']
num_epochs_Adam = int(history['num_epochs_Adam'])

fig_loss, axs = plt.subplots(1, 1)
axs.loglog(np.maximum(iterations_history, 1), loss_train_history, 'o-', label='training loss')
axs.loglog(np.maximum(iterations_history, 1), loss_valid_history, 'o-', label='validation loss')
if 'best_epoch' in history.files:
    axs.axvline(max(int(history['best_epoch']), 1), color='gray', linestyle=':', label='best epoch')
axs.axvline(max(num_epochs_Adam, 1), color='gray', linestyle='--', label='Adam | BFGS')
axs.set_xlabel('epochs'), axs.set_ylabel('loss')
axs.legend()
plt.show(block=False)

# %% test years: the scores count only the observed days (real survey on the day and on day 0)
out_fields_ROM, out_fields_FOM = common.predict(model, dataset_tests)
mask = dataset_tests['mask'].numpy() > 0
rom_observed, fom_observed = out_fields_ROM[mask], out_fields_FOM[mask]

NRMSE = np.sqrt(np.mean(np.square(rom_observed - fom_observed))) / (np.max(fom_observed) - np.min(fom_observed))
R_coeff = stats.pearsonr(rom_observed.reshape(-1), fom_observed.reshape(-1))
print('Normalized RMSE:       %1.3e' % NRMSE)
print('Pearson dissimilarity: %1.3e' % (1 - R_coeff[0]))
common.report(model, dataset_train, dataset_valid, dataset_tests)

# %% every valid window, in date order, for the viewer
valid_idx = np.flatnonzero(raw["valid"])
dataset_all = common.make_dataset(raw, valid_idx)
out_fields_ROM, out_fields_FOM = common.predict(model, dataset_all)     # [window, day, point]
signals_normalized = dataset_all['inp_signals'].numpy()                 # [window, day, signal]
observed = dataset_all['mask'].numpy() > 0                              # [window, day]
real = np.asarray(raw["real"])[valid_idx]                               # [window, day]  real survey that day
sample_dates = np.asarray(raw["sample_dates"])[valid_idx]
start_date = datetime.strptime(raw["start_date"], "%Y-%m-%d")
x_values = raw["x"]

split_of_window = np.full(len(raw["valid"]), "dropped", dtype=object)   # valid windows that cross two splits
for name, indices in zip(("train", "validation", "test"), common.split_windows(raw)):
    split_of_window[indices] = name
dataset_splits = split_of_window[valid_idx]

# %%
num_samples, num_days = out_fields_FOM.shape[:2]
rmse_values = np.sqrt(np.mean(np.square(out_fields_ROM - out_fields_FOM), axis=2))
value_range = np.max(out_fields_FOM) - np.min(out_fields_FOM)
nrmse_values = rmse_values / max(float(value_range), 1e-12)
nrmse_max = max(float(nrmse_values.max()), 1e-12)

fig, (ax, signal_ax, rmse_ax) = plt.subplots(
    1, 3,
    figsize=(11, 5),
    gridspec_kw={"width_ratios": [4, 1, 0.65]},
)
fom, = ax.plot(
    x_values,
    out_fields_FOM[0, 0, :],
    label=f"original ({common.OUTPUT_MODE} cumulative change)",
)
if reconstruction:
    rom, = ax.plot(
        x_values,
        out_fields_ROM[0, 0, :],
        label="reconstruction",
    )

ax.set_ylim(min(out_fields_FOM.min(), out_fields_ROM.min()), max(out_fields_FOM.max(), out_fields_ROM.max()))
ax.set_xlim(x_values.min(), x_values.max())
ax.axhline(0, color="gray", linewidth=0.5)
ax.legend()

signal_labels = list(raw["sign_names"])
signal_bars = signal_ax.barh(
    signal_labels,
    signals_normalized[0, 0],
)
signal_ax.set_xlim(-1.5, 1.5)
signal_ax.axvline(0, color="gray", linewidth=0.5)
signal_ax.set_xlabel("signals (normalized)")
signal_ax.invert_yaxis()

rmse_bar = rmse_ax.bar([0], [nrmse_values[0, 0]], width=0.6)
rmse_ax.set_ylim(0, nrmse_max * 1.05)
rmse_ax.set_xlim(-0.6, 0.6)
rmse_ax.set_xticks([])
rmse_ax.set_ylabel("NRMSE")

date_text = fig.suptitle("")

num_frames = num_samples * num_days
current_frame = 0
paused = False

fig.subplots_adjust(bottom=0.22)

slider_ax = fig.add_axes([0.18, 0.08, 0.62, 0.04])
time_slider = Slider(
    slider_ax,
    "Frame",
    0,
    num_frames - 1,
    valinit=0,
    valstep=1,
)

button_ax = fig.add_axes([0.82, 0.065, 0.12, 0.07])
pause_button = Button(button_ax, "Pause")

def draw_frame(frame):
    sample_idx, day_idx = divmod(int(frame), num_days)

    fom.set_ydata(out_fields_FOM[sample_idx, day_idx, :])
    fom.set_alpha(1.0 if real[sample_idx, day_idx] else 0.3)     # faded: interpolated day, not a real survey

    if reconstruction:
        rom.set_ydata(out_fields_ROM[sample_idx, day_idx, :])
        rmse_bar[0].set_height(nrmse_values[sample_idx, day_idx])

    for bar, value in zip(signal_bars, signals_normalized[sample_idx, day_idx]):
        bar.set_width(value)

    window_start = start_date + timedelta(hours=float(sample_dates[sample_idx]))
    day_date = window_start + timedelta(days=day_idx)
    state = "real survey" if real[sample_idx, day_idx] else "interpolated"
    counted = "scored" if observed[sample_idx, day_idx] else "not scored"       # scored in the report = real survey on the day and on day 0

    date_text.set_text(
        f"window {window_start:%Y-%m-%d} | {day_date:%Y-%m-%d} (day {day_idx + 1}/{num_days}) | "
        f"{dataset_splits[sample_idx]} | {state}, {counted}"
    )


def update(frame):
    global current_frame

    current_frame = int(frame)
    draw_frame(current_frame)

    if time_slider.val != current_frame:
        time_slider.set_val(current_frame)


def slider_changed(value):
    global current_frame

    current_frame = int(value)
    draw_frame(current_frame)

    if "animation" in globals() and paused:
        animation.frame_seq = iter(range(current_frame, num_frames))

    fig.canvas.draw_idle()

def step_day(direction):
    global current_frame

    target_frame = current_frame + direction
    target_frame = max(0, min(target_frame, num_frames - 1))

    current_frame = target_frame
    draw_frame(current_frame)
    time_slider.set_val(current_frame)

    if paused:
        animation.frame_seq = iter(range(current_frame, num_frames))

    fig.canvas.draw_idle()

def toggle_pause(_):
    global paused

    if paused:
        animation.frame_seq = iter(range(current_frame, num_frames))
        animation.event_source.start()
        pause_button.label.set_text("Pause")
    else:
        animation.event_source.stop()
        pause_button.label.set_text("Play")

    paused = not paused
    fig.canvas.draw_idle()

back_button_ax = fig.add_axes([0.04, 0.065, 0.06, 0.07])
back_button = Button(back_button_ax, "<")
back_button.on_clicked(lambda _: step_day(-1))

forward_button_ax = fig.add_axes([0.12, 0.065, 0.06, 0.07])
forward_button = Button(forward_button_ax, ">")
forward_button.on_clicked(lambda _: step_day(1))

time_slider.on_changed(slider_changed)
pause_button.on_clicked(toggle_pause)

animation = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=50,
    blit=False,
    repeat=False
)

plt.show()
# %%