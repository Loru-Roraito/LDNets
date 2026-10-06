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

hourly = True
reconstruction = True

# %%
if hourly:
    data_path = common.DATA_PATH_HOURLY
    dataset_train, dataset_valid, dataset_tests = common.load_datasets_hourly()
    num_input_signals = np.load(data_path, allow_pickle=True).item()['sign'].shape[2]
    problem, normalization = common.configuration_for_signal_count(num_input_signals)
else:
    data_path = common.DATA_PATH
    dataset_train, dataset_valid, dataset_tests = common.load_datasets()
    problem = common.problem
    normalization = common.normalization

model = common.LDNetModel.load(problem, normalization, common.num_latent_states, common.dt, common.dt_base, common.MODEL_DIR)

# %%
history = np.load(common.HISTORY_PATH)
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
out_fields = model(dataset_tests, autoregressive=True)
 
# Since the LDNet works with normalized data, we map back the outputs into the original ranges.
out_fields_FOM = utils.denormalize_output(dataset_tests['out_fields'], common.problem, common.normalization).numpy()
out_fields_ROM = utils.denormalize_output(out_fields, common.problem, common.normalization).numpy()
 
NRMSE = np.sqrt(np.mean(np.square(out_fields_ROM - out_fields_FOM))) / (np.max(out_fields_FOM) - np.min(out_fields_FOM))
R_coeff = stats.pearsonr(np.reshape(out_fields_ROM, (-1,)), np.reshape(out_fields_FOM, (-1,)))
 
print('Normalized RMSE:       %1.3e' % NRMSE)
print('Pearson dissimilarity: %1.3e' % (1 - R_coeff[0]))

raw_dataset = np.load(data_path, allow_pickle=True).item()
x_values = raw_dataset["x"]
all_indices = np.arange(len(raw_dataset["output"]))
raw_sample_ids = raw_dataset["sample_ids"]
restore_order = np.argsort(raw_sample_ids)
sample_ids = raw_sample_ids[restore_order]

dataset_all = utils.MY_create_dataset(data_path, all_indices[restore_order])
utils.process_dataset(dataset_all, problem, normalization)

out_fields = model(dataset_all, autoregressive=True)

out_fields_FOM = utils.denormalize_output(
    dataset_all["out_fields"], problem, normalization
).numpy()

out_fields_ROM = utils.denormalize_output(
    out_fields, problem, normalization
).numpy()
output_real = dataset_all["output_real"].numpy()

sample_dates = raw_dataset["sample_dates"][restore_order]
quality = raw_dataset["quality"][restore_order]
quality *= np.array([24, 2, 2, 2, 1])

dataset_splits_by_row = np.full(len(raw_dataset["output"]), "test", dtype=object)
dataset_splits_by_row[dataset_train["sample_indices"]] = "train"
dataset_splits_by_row[dataset_valid["sample_indices"]] = "validation"
dataset_splits = dataset_splits_by_row[restore_order]

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
if reconstruction:
    rom, = ax.plot(
        x_values,
        out_fields_ROM[0, 0, :, 0],
        label="reconstruction",
    )

ax.set_ylim(out_fields_FOM.min(), out_fields_FOM.max())
ax.set_xlim(x_values.min(), x_values.max())
ax.legend()

quality_labels = ["output", "height", "period", "direction", "tide"]
quality_values = quality[0]

quality_bars = quality_ax.barh(
    quality_labels,
    quality_values,
)

quality_ax.set_xlim(0, 28 * 24)
quality_ax.set_xlabel("valid hours")
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

    if output_real[sample_idx, day_idx]:
        fom.set_ydata(out_fields_FOM[sample_idx, day_idx, :, 0])

    if reconstruction:
        rom_frame = out_fields_ROM[sample_idx, day_idx, :, 0]
        rom.set_ydata(rom_frame)
        rmse_bar[0].set_height(nrmse_values[sample_idx, day_idx])

    for bar, value in zip(quality_bars, quality[sample_idx]):
        bar.set_width(value)

    sample_start_hours = sample_dates[sample_idx]
    date = common.start_date + timedelta(hours=float(sample_start_hours))

    dataset_split = dataset_splits[sample_idx]

    date_text.set_text(
        f"{date:%Y-%m-%d} | day {day_idx + 1}/{num_days} | {dataset_split}"
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
