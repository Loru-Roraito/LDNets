# %%
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import timedelta

import common
import numpy as np  
from network import utils
from scipy import stats
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# %%
data_path = common.DATA_PATH
dataset_train, dataset_valid, dataset_tests = common.load_datasets()
model_dt = common.dt
problem = common.problem
normalization = common.normalization
model_dir = common.MODEL_DIR
history_path = common.HISTORY_PATH

model = common.LDNetModel.load(
    problem,
    normalization,
    common.num_latent_states,
    model_dt,
    common.dt_base,
    model_dir,
)

# %%
history = np.load(history_path)
iterations_history = history['iterations_history']
loss_train_history = history['loss_train_history']
loss_valid_history = history['loss_valid_history']
num_epochs_Adam = int(history['num_epochs_Adam'])

fig_loss, axs = plt.subplots(1, 1)
axs.loglog(iterations_history, loss_train_history, 'o-', label='training loss')
axs.loglog(iterations_history, loss_valid_history, 'o-', label='validation loss')
axs.axvline(num_epochs_Adam)
axs.set_xlabel('epochs'), axs.set_ylabel('loss')
axs.legend()
plt.show()

# %%
out_fields = model(dataset_tests)
 
# Since the LDNet works with normalized data, we map back the outputs into the original ranges.
out_fields_FOM = utils.denormalize_output(dataset_tests['out_fields'], common.problem, common.normalization).numpy()
out_fields_ROM = utils.denormalize_output(out_fields, common.problem, common.normalization).numpy()
 
NRMSE = np.sqrt(np.mean(np.square(out_fields_ROM - out_fields_FOM))) / (np.max(out_fields_FOM) - np.min(out_fields_FOM))
R_coeff = stats.pearsonr(np.reshape(out_fields_ROM, (-1,)), np.reshape(out_fields_FOM, (-1,)))
 
print('Normalized RMSE:       %1.3e' % NRMSE)
print('Pearson dissimilarity: %1.3e' % (1 - R_coeff[0]))

raw_dataset = np.load(data_path, allow_pickle=True).item()
x_values = raw_dataset["x"]
split_of_window = np.asarray(raw_dataset["split"])
valid_indices = np.flatnonzero(raw_dataset["valid"] | (split_of_window == 3))     # the windows that pass the quality checks, plus the low-quality test windows, in date order
real = np.asarray(raw_dataset["real"])[valid_indices]                              # a real survey on that day (otherwise the original is interpolated)
check_names = list(raw_dataset["check_names"])
failed_checks = [", ".join(name for name, ok in zip(check_names, row) if not ok) for row in np.asarray(raw_dataset["checks"])[valid_indices]]

dataset_all = common.make_dataset(raw_dataset, valid_indices)

out_fields = model(dataset_all)

out_fields_FOM = utils.denormalize_output(
    dataset_all["out_fields"], problem, normalization
).numpy()

out_fields_ROM = utils.denormalize_output(
    out_fields, problem, normalization
).numpy()
sample_dates = raw_dataset["sample_dates"][valid_indices]
signals_normalized = dataset_all["inp_signals"].numpy()

split_names = np.array(["extreme (left out)", "not used", "train", "validation", "test", "test, LOW QUALITY"])      # split: -2, -1, 0, 1, 2, 3 (written by parse.py)
dataset_splits = split_names[split_of_window[valid_indices] + 2]

# %%
num_samples, num_days = out_fields_FOM.shape[:2]
rmse_values = np.sqrt(
    np.mean(np.square(out_fields_ROM - out_fields_FOM), axis=(2, 3))
)
value_range = np.max(out_fields_FOM) - np.min(out_fields_FOM)
nrmse_values = rmse_values / max(float(value_range), 1e-12)
nrmse_max = max(float(nrmse_values.max()), 1e-12)

fig, (ax, quality_ax, rmse_ax) = plt.subplots(
    1, 3,
    figsize=(11, 5),
    gridspec_kw={"width_ratios": [4, 1, 0.65]},
)
fom, = ax.plot(
    x_values,
    out_fields_FOM[0, 0, :, 0],
    label="original",
)
rom, = ax.plot(
    x_values,
    out_fields_ROM[0, 0, :, 0],
    label="reconstruction",
)

ax.set_ylim(out_fields_FOM.min(), out_fields_FOM.max())
ax.set_xlim(x_values.min(), x_values.max())
ax.legend()

signal_labels = [signal["name"] for signal in problem["input_signals"]]

signal_bars = quality_ax.barh(
    signal_labels,
    signals_normalized[0, 0],
)

quality_ax.set_xlim(-1.5, 1.5)
quality_ax.axvline(0, color="gray", linewidth=0.5)
quality_ax.set_xlabel("signals (normalized)")
quality_ax.invert_yaxis()

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

    fom.set_ydata(out_fields_FOM[sample_idx, day_idx, :, 0])
    fom.set_alpha(1.0 if real[sample_idx, day_idx] else 0.3)         # faded: interpolated day, not a real survey

    rom_frame = out_fields_ROM[sample_idx, day_idx, :, 0]
    rom.set_ydata(rom_frame)
    rmse_bar[0].set_height(nrmse_values[sample_idx, day_idx])

    for bar, value in zip(signal_bars, signals_normalized[sample_idx, day_idx]):
        bar.set_width(value)

    sample_start_hours = sample_dates[sample_idx]
    date = common.START_DATE + timedelta(hours=float(sample_start_hours))

    dataset_split = dataset_splits[sample_idx]

    survey = "real survey" if real[sample_idx, day_idx] else "interpolated"
    failed = f" | failed: {failed_checks[sample_idx]}" if failed_checks[sample_idx] else ""
    date_text.set_text(
        f"{date:%Y-%m-%d} | day {day_idx + 1}/{num_days} | {dataset_split} | {survey}{failed}"
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