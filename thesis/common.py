import os
import json
from pathlib import Path
from network import utils
import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx('float64')

SEED = int(os.environ.get("LDNETS_SEED", "42"))
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

dt = 24
dt_hourly = 1
dt_base = 24
# ---- settings that run_experiments.py varies -----------------------------------------------------------------
# Order of priority: environment variable LDNETS_<NAME>  >  config.json of the run LDNETS_RUN=<run>  >  the default below.
# So the plain "python train.py" uses the defaults, and "LDNETS_RUN=pca5_s0 python visualize.py" rebuilds the settings of that run.
RUN_NAME = os.environ.get("LDNETS_RUN", "")
_thesis_dir = Path(__file__).resolve().parents[1] / 'thesis'
_run_dir = _thesis_dir / 'model' / 'runs' / RUN_NAME if RUN_NAME else _thesis_dir / 'model'
try:
    _config = json.loads((_run_dir / "config.json").read_text()) if RUN_NAME else {}
except FileNotFoundError:
    _config = {}

def _setting(name, default):
    return type(default)(os.environ.get("LDNETS_" + name, _config.get(name, default)))

num_latent_states = _setting("num_latent_states", 3)
N_PROFILE_POINTS = _setting("N_PROFILE_POINTS", 11)      # inputs describing the day-0 profile (0 = none)
PROFILE_FEATURES = _setting("PROFILE_FEATURES", "points")  # "points": N evenly spaced points of the profile;  "pca": its first N principal components
OUTPUT_MODE = "abs"

problem = {
    "space": {
        "dimension" : 1
    },
    "input_parameters": [{"name": f"profile_{i}"} for i in range(N_PROFILE_POINTS)],
    "input_signals": [
        { "name": "hmean" },
        { "name": "hmax" },

        { "name": "tmean" },
        { "name": "tmax" },

        { "name": "dirmean" },
        { "name": "dirmax" },

        { "name": "tidemean" },
        { "name": "tidemax" }
    ],
    "output_fields": [
        { "name": "change" }
    ]
}

normalization = {
    'space': { 'min' : [-115], 'max' : [+385]},
    'time': {
        'time_constant' : dt_base
    },
    "input_parameters": {},     # filled by load_raw, from the training windows (see set_profile_normalization)
    'input_signals': {
        'hmean': { 'min': 0.32, 'max': 6.92 },
        'hmax': { 'min': 0.44, 'max': 10.89 },

        'tmean': { 'min': 5.9, 'max': 11 },
        'tmax': { 'min': 5.3, 'max': 11.9 },

        'dirmean': { 'min': 0.67, 'max': 0.99 },
        'dirmax': { 'min': 0.48, 'max': 1 },

        'tidemean': { 'min': 1.55, 'max': 1.92 },
        'tidemax': { 'min': 1.07, 'max': 2.21 }
    },
    'output_fields': {
        'change': { 'min': 0, 'max': +98 }
    }
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / 'thesis' / 'data' /'data.npy'
RUN_DIR = _run_dir
MODEL_DIR = RUN_DIR / 'trained_model'
HISTORY_PATH = RUN_DIR / 'training_history.npz'
RESULTS_PATH = RUN_DIR / 'results.json'

YEAR_BLOCK = 5                         
VALID_REMAINDER = 1
TEST_REMAINDER = 4
 
def split_windows(raw):
    days = (np.asarray(raw["sample_dates"]) / 24).astype(int)
    first = np.datetime64(raw["start_date"]) + days.astype("timedelta64[D]")
    last = first + np.timedelta64(raw["sign"].shape[1] - 1, "D")
 
    def split_of(day):
        remainder = (day.astype("datetime64[Y]").astype(int) + 1970) % YEAR_BLOCK
        return np.where(remainder == VALID_REMAINDER, 1, np.where(remainder == TEST_REMAINDER, 2, 0))
 
    split_first, split_last = split_of(first), split_of(last)
    ok = np.asarray(raw["valid"], dtype=bool) & (split_first == split_last)
    return tuple(np.flatnonzero(ok & (split_first == k)) for k in range(3))

_pca = {}    # mean profile and principal components (PROFILE_FEATURES = "pca"), computed from the training windows by set_profile_normalization

def profile_points(raw):
    """Indices of the profile points used as parameters: N_PROFILE_POINTS, evenly spaced from the first to the last."""
    return np.linspace(0, raw['profile0'].shape[1] - 1, N_PROFILE_POINTS).astype(int)

def profile_features(raw, indices):
    """Day-0 profile inputs of the windows: [len(indices), N_PROFILE_POINTS], in metres (before normalization)."""
    profile = raw['profile0'][indices]
    if PROFILE_FEATURES == "pca":
        return (profile - _pca['mean']) @ _pca['components'].T
    return profile[:, profile_points(raw)]

def set_profile_normalization(raw):
    """Fit what the profile inputs need, from the training windows only (nothing from valid/test enters):
    the principal components if PROFILE_FEATURES = "pca", and the min and max of every input."""
    if N_PROFILE_POINTS == 0:
        return
    if PROFILE_FEATURES not in ("points", "pca"):
        raise ValueError(f"PROFILE_FEATURES must be 'points' or 'pca', not {PROFILE_FEATURES!r}")
    train, _, _ = split_windows(raw)
    profile = raw['profile0']
    if N_PROFILE_POINTS > min(profile.shape[1], len(train) if PROFILE_FEATURES == "pca" else profile.shape[1]):
        raise ValueError(f"N_PROFILE_POINTS = {N_PROFILE_POINTS} is more than the {profile.shape[1]} profile points "
                         f"or the {len(train)} training windows")
    if PROFILE_FEATURES == "pca":
        mean = profile[train].mean(axis=0)
        _, singular, components = np.linalg.svd(profile[train] - mean, full_matrices=False)
        components = components[:N_PROFILE_POINTS]
        components = components * np.sign(components[np.arange(len(components)), np.abs(components).argmax(axis=1)])[:, None]   # fixed sign: reproducible
        _pca.update(mean=mean, components=components)
        explained = 100 * np.sum(singular[:N_PROFILE_POINTS] ** 2) / np.sum(singular ** 2)
        print(f"profile parameters: {N_PROFILE_POINTS} principal components ({explained:.0f}% of the variance of the training profiles)")
    else:
        print(f"profile parameters: {N_PROFILE_POINTS} points, x = {np.round(np.asarray(raw['x'])[profile_points(raw)], 1).tolist()}")
    values = profile_features(raw, train)
    low, high = values.min(axis=0), values.max(axis=0)
    high = np.maximum(high, low + 1e-6)
    normalization['input_parameters'] = {
        f"profile_{i}": {'min': float(l), 'max': float(h)} for i, (l, h) in enumerate(zip(low, high))
    }

def make_dataset(raw, indices):
    dataset = {
        'points': raw['x'][:, None],                      
        'times': raw['t'],                                
        'inp_parameters': profile_features(raw, indices) if N_PROFILE_POINTS > 0 else None,
        'inp_signals': raw['sign'][indices],                
        'out_fields': raw['output'][indices][..., None],    
    }
    utils.process_dataset(dataset, problem, normalization)
    real = np.asarray(raw['real'])[indices]
    mask = real & real[:, :1]
    mask[:, 0] = False
    dataset['mask'] = tf.convert_to_tensor(mask, tf.float64)
    dataset['sample_indices'] = np.asarray(indices)
    return dataset
 
REQUIRED_KEYS = ("x", "sign", "profile0", "sign_names", "output", "output_mode", "real", "valid", "sample_dates", "start_date", "t")

def load_raw():
    raw = np.load(DATA_PATH, allow_pickle=True)[()]
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise KeyError(f"{DATA_PATH} has no {missing}: run parse.py again")
    if raw["output_mode"] != OUTPUT_MODE:
        raise ValueError(f"the data was parsed with OUTPUT_MODE = {raw['output_mode']!r} but common.py has {OUTPUT_MODE!r}")
    expected = [signal['name'] for signal in problem['input_signals']]
    if list(raw["sign_names"]) != expected:
        raise ValueError(f"signals in the data {list(raw['sign_names'])} differ from problem['input_signals'] {expected}")
    set_profile_normalization(raw)
    return raw

def load_datasets(raw=None):
    raw = load_raw() if raw is None else raw
    train, valid, test = split_windows(raw)
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError(f"empty split (train {len(train)}, valid {len(valid)}, test {len(test)} windows) out of "
                         f"{int(np.sum(raw['valid']))} valid windows: check the thresholds in parse.py or the year blocks")
    datasets = tuple(make_dataset(raw, indices) for indices in (train, valid, test))
    for name, dataset in zip(("train", "valid", "test"), datasets):
        targets = dataset['out_fields'].numpy()[..., 0][dataset['mask'].numpy() > 0]
        print(f"{name:5s}: {dataset['num_samples']:5d} windows, {int(dataset['mask'].numpy().sum()):6d} observed days, "
              f"{100 * np.mean(np.abs(targets) > 1):.1f}% of the targets outside the normalized range [-1, 1]")
    return datasets

class LDNetModel:
    def __init__(
        self,
        problem,
        normalization,
        num_latent_states,
        dt,
        dt_base,
        reconstruction_batch_size=4,
    ):
        self.problem = problem
        self.normalization = normalization
        self.num_latent_states = num_latent_states
        self.dt = dt
        self.dt_base = dt_base
        self.reconstruction_batch_size = reconstruction_batch_size

        input_shape_dyn = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)
        self.NNdyn = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape_dyn),
            tf.keras.layers.Dense(32, activation=tf.nn.tanh),
            tf.keras.layers.Dense(32, activation=tf.nn.tanh),
            tf.keras.layers.Dense(num_latent_states)
        ])

        # the reconstruction sees only the latent state and the position: the forcing acts through the dynamics,
        # so the cumulative change cannot react directly to the current day's waves
        input_shape_rec = (None, None, num_latent_states + problem['space']['dimension'])
        self.NNrec = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape_rec),
            tf.keras.layers.Dense(24, activation=tf.nn.tanh),
            tf.keras.layers.Dense(24, activation=tf.nn.tanh),
            tf.keras.layers.Dense(24, activation=tf.nn.tanh),
            tf.keras.layers.Dense(24, activation=tf.nn.tanh),
            tf.keras.layers.Dense(len(problem['output_fields']))
        ])
 
    def summary(self):
        self.NNdyn.summary()
        self.NNrec.summary()
 
    @property
    def trainable_variables(self):
        return self.NNdyn.variables + self.NNrec.variables
 
    def evolve_dynamics(self, dataset):
        if len(self.problem['input_parameters']) > 0:
            parameters = dataset['inp_parameters']          
        else:
            parameters = tf.zeros((dataset['num_samples'], 0), dtype=tf.float64)
        state = tf.zeros((dataset['num_samples'], self.num_latent_states), dtype=tf.float64)
        state_history = tf.TensorArray(tf.float64, size=dataset['num_times'])
        state_history = state_history.write(0, state)
        dt_ref = self.normalization['time']['time_constant']
 
        for i in tf.range(dataset['num_times'] - 1):
            state = state + self.dt / dt_ref * self.NNdyn(tf.concat([state, parameters, dataset['inp_signals'][:, i, :]], axis=-1))
            state_history = state_history.write(i + 1, state)
 
        return tf.transpose(state_history.stack(), perm=(1, 0, 2))

    @staticmethod
    def _output_transform(output):
        # nonlinear transformation to compress long tails
        alpha = 0.05
        return (output ** 3 + alpha * output) / (1 + alpha)
 
    def reconstruct_output(self, dataset, states):
        outputs = []
        for start in range(0, dataset['num_samples'], self.reconstruction_batch_size):
            stop = min(start + self.reconstruction_batch_size, dataset['num_samples'])
            states_batch = states[start:stop]
            points_batch = dataset['points_full'][start:stop]

            states_expanded = tf.broadcast_to(
                tf.expand_dims(states_batch, axis=2),
                [stop - start, dataset['num_times'], dataset['num_points'], self.num_latent_states],
            )
            outputs.append(self.NNrec(tf.concat([
                states_expanded,
                points_batch,
            ], axis=3)))

        output = tf.concat(outputs, axis=0)
        return self._output_transform(output)
 
    def _reconstruct_timestep(self, dataset, state):
        states_expanded = tf.broadcast_to(
            tf.expand_dims(tf.expand_dims(state, axis=1), axis=1),
            [dataset['num_samples'], 1, dataset['num_points'], self.num_latent_states],
        )
        output = self.NNrec(tf.concat([
            states_expanded,
            dataset['points_full'][:, 0:1, :, :],
        ], axis=3))
        return self._output_transform(output)[:, 0, :, :]
    
    def __call__(self, dataset):
        states = self.evolve_dynamics(dataset)
        return self.reconstruct_output(dataset, states)
 
    def save(self, model_dir=None):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        # Keras 3 native format requires the .keras extension.
        self.NNdyn.save(str(model_dir / 'NNdyn.keras'))
        self.NNrec.save(str(model_dir / 'NNrec.keras'))
 
    @classmethod
    def load(
        cls,
        problem,
        normalization,
        num_latent_states,
        dt,
        dt_base,
        model_dir=None,
        reconstruction_batch_size=4,
    ):
        model_dir = Path(model_dir)
        model = cls(
            problem,
            normalization,
            num_latent_states,
            dt,
            dt_base,
            reconstruction_batch_size,
        )
        model.NNdyn = tf.keras.models.load_model(str(model_dir / 'NNdyn.keras'))
        model.NNrec = tf.keras.models.load_model(str(model_dir / 'NNrec.keras'))
        return model

weight_direction = 0.1 
epsilon = 1e-4

def get_direction(velocity):
    return tf.math.divide(velocity, (epsilon + tf.expand_dims(tf.norm(velocity, axis=3), axis=-1)))
 
def make_loss_fn(
    model,
    dataset,
    target_velocity,
    target_direction,
):
    def loss_fn():
        velocity = model(dataset)
        MSE_velocity = tf.reduce_mean(tf.square(velocity - target_velocity))
        direction = get_direction(velocity)
        MSE_direction = tf.reduce_mean(tf.square(direction - target_direction))
        return MSE_velocity + weight_direction * MSE_direction
    return loss_fn

# ---- evaluation (used by train.py and visualize.py; not part of the training) ----
PREDICT_CHUNK = 25      # windows per forward pass when predicting

def to_units(field):
    """Normalized field [S, T, P, 1] -> physical units [S, T, P]."""
    return utils.denormalize_output(field, problem, normalization).numpy()[..., 0]

def predict(model, dataset, chunk=PREDICT_CHUNK):
    """Model output and target in physical units, both [S, T, P]."""
    parts = []
    parameters = dataset.get('inp_parameters')
    for start in range(0, int(dataset['num_samples']), chunk):
        signals = dataset['inp_signals'][start:start + chunk]
        sub = dict(num_samples=int(signals.shape[0]), num_times=dataset['num_times'], num_points=dataset['num_points'],
                   inp_signals=signals, points_full=dataset['points_full'][start:start + chunk])
        if parameters is not None:
            sub['inp_parameters'] = parameters[start:start + chunk]
        parts.append(model(sub))
    return to_units(tf.concat(parts, axis=0)), to_units(dataset['out_fields'])

def rmse(error, mask):
    """Root mean square of error [S, T, P] over the days of mask [S, T] (all points)."""
    return float(np.sqrt(np.mean(np.square(error[mask])))) if mask.any() else float('nan')

def report(model, dataset_train, dataset_valid, dataset_tests, top_share=0.1):
    """RMSE (physical units, observed days only) of the model against two baselines that use no forcing:
    'zero' (nothing changes) and 'mean' (the average change of the training windows, by day and point).
    'top' = the windows with the largest mean |change| (top_share of them): the extreme events."""
    mask_train = dataset_train['mask'].numpy() > 0
    _, target_train = predict(model, dataset_train)
    mean_change = np.stack([target_train[:, k][mask_train[:, k]].mean(axis=0) if mask_train[:, k].any() else np.zeros(target_train.shape[2])
                            for k in range(target_train.shape[1])])                       # [T, P]
    rows = []
    for name, dataset in (("train", dataset_train), ("valid", dataset_valid), ("test", dataset_tests)):
        prediction, target = predict(model, dataset)
        mask = dataset['mask'].numpy() > 0
        strength = (np.abs(target) * mask[..., None]).sum(axis=(1, 2)) / np.maximum(mask.sum(axis=1) * target.shape[2], 1)
        top = mask & (strength >= np.quantile(strength[mask.any(axis=1)], 1 - top_share))[:, None]
        errors = [prediction - target, -target, mean_change[None] - target]               # model, zero, mean
        rows.append((name, *[rmse(e, mask) for e in errors], *[rmse(e, top) for e in errors[:2]]))
    print(f"{'':6s}{'model':>9s}{'zero':>9s}{'mean':>9s}{'vs zero':>9s}{'vs mean':>9s} | top {int(top_share * 100)}%:{'model':>8s}{'zero':>9s}{'vs zero':>9s}")
    for name, m, z, mean, mt, zt in rows:
        print(f"{name:6s}{m:9.2f}{z:9.2f}{mean:9.2f}{1 - m**2 / z**2:9.2f}{1 - m**2 / mean**2:9.2f} |{'':9s}{mt:8.2f}{zt:9.2f}{1 - mt**2 / zt**2:9.2f}")
    print("vs zero / vs mean = 1 - MSE(model) / MSE(baseline), above 0 means better than the baseline. 'mean' = the average change of the training windows (day by day, point by point), no forcing.")
    return {name: {"rmse_model": m, "rmse_zero": z, "rmse_mean": mean, "vs_zero": 1 - m**2 / z**2, "vs_mean": 1 - m**2 / mean**2,
                   "top_rmse_model": mt, "top_rmse_zero": zt, "top_vs_zero": 1 - mt**2 / zt**2}
            for name, m, z, mean, mt, zt in rows}