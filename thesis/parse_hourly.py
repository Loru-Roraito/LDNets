import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
import pandas as pd
import numpy as np
from datetime import datetime
import common

dt = 28 # number of days in a sample
wave_n = 24
tide_n = 24
train_threshold = 0.9
test_threshold = 0.5
start_date = datetime(1997, 4, 1)
end_date = datetime(2019, 1, 31 )
delta = (end_date - start_date).total_seconds() / 3600

def parse_xaxis(value):
    return float(value.rstrip("m"))

def parse_coast_date(value):
    dt = datetime.strptime(value, "%Y-%m-%d")
    return (dt - start_date).total_seconds() / 3600

def parse_date(dt):
    return (dt - start_date).total_seconds() / 3600

with open("thesis/data/coast.json", "r", encoding="utf-8") as f:
    coast = json.load(f)

xaxis = [parse_xaxis(x) for x in coast["xaxis"]]
xaxis = np.asarray(xaxis, dtype=np.float64)
series = coast["series"]

coast_dates = []
values = []

for value in series:
    date = parse_coast_date(value[0])
    coast_dates.append(date)
    measurements = [float(x) for x in value[1:]]
    values.append(measurements)

coast_dates = np.asarray(coast_dates, dtype=np.float64)
values = np.asarray(values, dtype=np.float64)


with open("thesis/data/wave.csv", "r", encoding="utf-8") as f:
    wave = pd.read_csv(f)

wave_dates = pd.to_datetime(wave["date"], format="%Y/%m/%d %H:%M:%S")
wave_dates = [parse_date(t) for t in wave_dates]
wave_dates = np.asarray(wave_dates, dtype=np.float64)

hs = np.asarray(wave["H"], dtype=np.float64)
ts = np.asarray(wave["T"], dtype=np.float64)
dirs = np.asarray(wave["Dir"], dtype=np.float64)


with open("thesis/data/tide.csv", "r", encoding="utf-8") as f:
    tide = pd.read_csv(f)

tide_dates = pd.to_datetime(tide["date"], format="%Y/%m/%d %H:%M:%S")
tide_dates = [parse_date(t) for t in tide_dates]
tide_dates = np.asarray(tide_dates, dtype=np.float64)

tides = np.asarray(tide["tide"], dtype=np.float64)


def split_values(values, dates, dt):
    dt = dt * 24
    output = []
    output_real = []
    qualities = [] # check how much data is real and how much comes from interpolation
    i = 0
    time = 0
    while dates[i] < time:
        i += 1

    sample_dates = []

    while dates[i] < delta:
        sample_dates.append(time)
        quality = dt
        samples = []
        real_samples = []
        initial_time = time

        while time - initial_time < dt and dates[i] < delta:
            if dates[i] == time:
                samples.append(values[i])
                real_samples.append(True)
                i += 1
            else:
                # linear interpolation
                quality -= 1
                samples.append((values[i - 1] + (values[i] - values[i - 1]) * ((time - dates[i - 1]) / (dates[i] - dates[i - 1]))))
                real_samples.append(False)
            time += 1

        output.append(samples)
        output_real.append(real_samples)
        qualities.append(quality - int((dt - (time - initial_time)) / 24))

    if len(output[-1]) < len(output[-2]):
        output.pop()
        output_real.pop()
    return output, output_real, qualities, sample_dates

def split_signals(hs, ts, dirs, wave_dates, tides, tide_dates, dt):
    dt = dt * 24
    signals = []
    i = 0
    time = 0
    while wave_dates[i] < time:
        i += 1
    j = 0
    while tide_dates[j] < time:
        j += 1
    h_qualities = []
    t_qualities = []
    dir_qualities = []
    tide_qualities = []
    a = 0
    while time < delta:
        current_time = 0
        signal = []
        h_quality = dt
        t_quality = dt
        dir_quality = dt
        tide_quality = dt

        b = 0
        while current_time < dt and time < delta:
            subtime = 0
            h = []
            t = []
            dir = []
            tide = []
            while subtime < 1:
                if i < len(wave_dates) and wave_dates[i] == time:
                    if not np.isnan(hs[i]):
                        h += [hs[i] ** 2]
                    else:
                        h_quality -= 1
                        if current_time > 0:
                            h += [signal[-1][0]]
                        elif time > 0 and subtime == 0:
                            h += [signals[-1][-1][0]]
                        else:
                            h += [0.78 ** 2]
                    if not np.isnan(ts[i]):
                        t += [ts[i]]
                    else:
                        t_quality -= 1
                        if current_time > 0:
                            t += [signal[-1][4]]
                        elif time > 0 and subtime == 0:
                            t += [signals[-1][-1][4]]
                        else:
                            t += [6.9]
                    if not np.isnan(dirs[i]):
                        dir += [dirs[i]]
                    else:
                        dir_quality -= 1
                        if current_time > 0:
                            dir += [signal[-1][8]]
                        elif time > 0 and subtime == 0:
                            dir += [signals[-1][-1][8]]
                        else:
                            dir += [6.81]
                    i += 1
                else:
                    h_quality -= 1
                    t_quality -= 1
                    dir_quality -= 1
                    if current_time > 0:
                        h += [signal[-1][0]]
                        t += [signal[-1][4]]
                        dir += [signal[-1][8]]
                    elif time > 0 and subtime == 0:
                        h += [signals[-1][-1][0]]
                        t += [signals[-1][-1][4]]
                        dir += [signals[-1][-1][8]]
                    else:
                        h += [0.78 ** 2]
                        t += [6.9]
                        dir += [6.81]
                if j < len(tide_dates) and tide_dates[j] == time:
                    if not np.isnan(tides[j]):
                        tide += [tides[j]]
                    else:
                        tide_quality -= 1
                        if current_time > 0:
                            tide += [signal[-1][12]]
                        elif time > 0 and subtime == 0:
                            tide += [signals[-1][-1][12]]
                        else:
                            tide += [1.72]
                    j += 1
                else:
                    tide_quality -= 1
                    if current_time > 0:
                        tide += [signal[-1][12]]
                    elif time > 0 and subtime == 0:
                        tide += [signals[-1][-1][12]]
                    else:
                        tide += [1.72]
                subtime += 1
                time += 1

            current_time += 1

            hmean = np.mean(h)
            hmax = np.max(h)
            hmin = np.min(h)
            hdev = np.std(h)

            tmean = np.mean(t)
            tmax = np.max(t)
            tmin = np.min(t)
            tdev = np.std(t)

            dirmean = np.mean(dir)
            dirmax = np.max(dir)
            dirmin = np.min(dir)
            dirdev = np.std(dir)

            tidemean = np.mean(tide)
            tidemax = np.max(tide)
            tidemin = np.min(tide)
            tidedev = np.std(tide)

            signal.append([hmean, hmax, hmin, hdev, tmean, tmax, tmin, tdev, dirmean, dirmax, dirmin, dirdev, tidemean, tidemax, tidemin, tidedev])  

            old_change = 0
            for x in range(common.n_points):
                if a < len(output):
                    if b > 0:
                        height = output[a][b - 1][x * len(xaxis) // common.n_points]
                        if b > 1:
                            previous = output[a][b - 2][x * len(xaxis) // common.n_points]
                        elif a > 0:
                            previous = output[a - 1][-1][x * len(xaxis) // common.n_points]
                        else:
                            previous = height
                    elif a > 0:
                        height = output[a - 1][-1][x * len(xaxis) // common.n_points]
                        previous = output[a - 1][-2][x * len(xaxis) // common.n_points]
                    else:
                        height = output[a][0][x * len(xaxis) // common.n_points]
                        previous = height

                    change = height - previous
                    signal[-1].append(height)
                    if output_real[a][b]:
                        signal[-1].append(change)
                        old_change = change
                    else:
                        signal[-1].append(old_change)

            b += 1

        signals.append(signal)
        h_qualities.append(h_quality - (dt - current_time))
        t_qualities.append(t_quality - (dt - current_time))
        dir_qualities.append(dir_quality - (dt - current_time))
        tide_qualities.append(tide_quality - (dt - current_time))
        a += 1

    if len(signals[-1]) < len(signals[-2]):
        signals.pop()
    
    return signals, h_qualities, t_qualities, dir_qualities, tide_qualities

output, output_real, a, sample_dates = split_values(values, coast_dates, dt)

signals, b, c, d, e = split_signals(hs, ts, dirs, wave_dates, tides, tide_dates, dt)

scores = []
for i in range(len(output)):
    if a[i] <= 14:
        scores.append(0)
    else:
        scores.append((24 * a[i] + 2 * b[i] + 2 * c[i] + 2 * d[i] + e[i]) / (24*dt*5))

sample_ids = np.arange(len(output))
quality = np.column_stack([a, b, c, d, e])

order = np.argsort(np.asarray(scores))[::-1]

sample_ids = sample_ids[order]
sample_dates = np.asarray(sample_dates)[order]
quality = quality[order]

scores = np.asarray(scores, dtype=np.float64)[order]
signals = np.asarray(signals, dtype=np.float64)[order]
output = np.asarray(output, dtype=np.float64)[order]
output_real = np.asarray(output_real, dtype=bool)[order]

for i, score in enumerate(scores):
    if score <= train_threshold:
        print(f"{i}/{len(scores)} scores above {train_threshold}")
        break
for i, score in enumerate(scores):
    if score <= test_threshold:
        print(f"{i}/{len(scores)} scores above {test_threshold}")
        break

dataset = {
    "x": xaxis,
    "sign": signals,
    "output": output,
    "output_real": output_real,
    "sample_ids": sample_ids,
    "sample_dates": sample_dates,
    "quality": quality,
    "t": np.arange(dt * 24, dtype=np.float64)
}

print(dataset["x"].shape)
print(dataset["t"].shape)
print(dataset["output"].shape)
print(dataset["sign"].shape)

np.save("thesis/data/data_hourly.npy", dataset)
print("data converted to npy")