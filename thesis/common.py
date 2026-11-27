import os
from pathlib import Path
from network import utils
import numpy as np
import tensorflow as tf
from datetime import datetime

tf.keras.backend.set_floatx('float64')

SEED = int(os.environ.get("LDNETS_SEED", "42"))
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

START_DATE = datetime(1987, 1, 5)
END_DATE = datetime(2024, 12, 24)
SEASON = True
MAX = True # measured at max H
MEAN = True
PEAK = True # measured at max tide
TIDE = True
PERCENTILE = 99.9
dt = 24
dt_base = 24
num_latent_states = 5
n_PCA = 10
enc_width = 24
dyn_width = 32
rec_width = 24
num_epochs_Adam = 100
num_epochs_BFGS = 500   
weight_direction = 0.1
learning_rate = 1e-3   

SIGNAL_TABLE = [     # name, switched on, {PERCENTILE: (min, max)}
    ("hmean",    MEAN,          {99: (0.22, 13.72), 99.9: (0.16, 26.07), 100: (0.11, 35.54)}),
    ("htide",    PEAK,          {99: (0.19, 16.87), 99.9: (0.14, 30.90), 100: (0.12, 39.44)}),
    ("hmax",     MAX,           {99: (0.28, 22.18), 99.9: (0.20, 42.44), 100: (0.14, 63.04)}),
    ("tmean",    MEAN,          {99: (5.37, 12.53), 99.9: (4.84, 14.53), 100: (4.62, 15.73)}),
    ("ttide",    PEAK,          {99: (5.00, 13.20), 99.9: (4.20, 14.94), 100: (3.90, 15.90)}),
    ("tmax",     MAX,           {99: (4.70, 13.80), 99.9: (4.00, 15.90), 100: (3.70, 17.60)}),
    ("dirmean",  MEAN,          {99: (0.55,  1.00), 99.9: (0.42,  1.00), 100: (0.33,  1.00)}),
    ("dirtide",  PEAK,          {99: (0.37,  1.00), 99.9: (0.24,  1.00), 100: (0.21,  1.00)}),
    ("dirmax",   MAX,           {99: (0.31,  1.00), 99.9: (0.21,  1.00), 100: (0.21,  1.00)}),
    ("tidemean", MEAN and TIDE, {99: (1.48,  2.02), 99.9: (1.41,  2.21), 100: (1.39,  2.54)}),
    ("tidetide", PEAK and TIDE, {99: (1.80,  2.50), 99.9: (1.68,  2.67), 100: (1.05,  3.04)}),
    ("tidemax",  MAX and TIDE,  {99: (0.87,  2.36), 99.9: (0.71,  2.55), 100: (0.65,  3.04)}),
]

SIGNAL_NAMES = [name for name, on, _ in SIGNAL_TABLE if on]
SIGNAL_RANGES = {name: ranges[PERCENTILE] for name, on, ranges in SIGNAL_TABLE if on}

OUTPUT_TABLE = [     # name, switched on, {PERCENTILE: (min, max)}
    ("y", True, {99: (0.0, 157), 99.9: (-608, 697), 100: (0.0, 331)}),
]
OUTPUT_RANGES = {name: ranges[PERCENTILE] for name, on, ranges in OUTPUT_TABLE if on}

problem = {
    "space": {
        "dimension" : 1
    },
    "input_parameters": [
        *[
            {"name": f"PCA_{point}"}
            for point in range(n_PCA)
        ],
        *([
            {"name": "season_sin"},
            {"name": "season_cos"}
        ] if SEASON else [])
    ],
    "input_signals": [{"name": n} for n in SIGNAL_NAMES],
    "output_fields": [
        { "name": "y" }
    ]
}

normalization = {
    'space': { 'min' : [-115], 'max' : [+385]},
    'time': {
        'time_constant' : dt_base
    },
    'input_signals': {n: {'min': lo, 'max': hi} for n, (lo, hi) in SIGNAL_RANGES.items()},
    'input_parameters': {},     # filled by load_datasets
    'output_fields': {n: {'min': lo, 'max': hi} for n, (lo, hi) in OUTPUT_RANGES.items()}
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / 'thesis' / 'data' /'data.npy'

RUN_TAG = "noPCA"  # optional tag to distinguish runs with the same settings
season = "-".join(p['name'][len("season_"):] for p in problem['input_parameters'] if p['name'].startswith("season_")) or "none"
RUN_NAME = (f"lat{num_latent_states}_pca{n_PCA}_season-{season}_w{dyn_width}x{rec_width}_lr{learning_rate:g}_dir{weight_direction:g}"
            f"_adam{num_epochs_Adam}_bfgs{num_epochs_BFGS}_seed{SEED}_mean{MEAN}_peak{PEAK}_max{MAX}_tide{TIDE}_percentile{PERCENTILE}" + (f"_{RUN_TAG}" if RUN_TAG else ""))
RUN_DIR = REPO_ROOT / 'thesis' / 'model' / RUN_NAME
MODEL_DIR = RUN_DIR / 'trained_model'   
HISTORY_PATH = RUN_DIR / 'training_history.npz'

REQUIRED_KEYS = ("x", "t", "sign", "output", "profile0", "split", "sample_dates")


# TODO: check PCA
pca = {}

def fit_pca(profile, n_components):
    """Principal components of the profiles [windows, points]: mean, components [n, points], % of variance explained."""
    mean = profile.mean(axis=0)
    _, singular, components = np.linalg.svd(profile - mean, full_matrices=False)
    components = components[:n_components]  
    components = components * np.sign(components[np.arange(n_components), np.abs(components).argmax(axis=1)])[:, None]   # fixed sign: same result every run
    return mean, components, 100 * np.sum(singular[:n_components] ** 2) / np.sum(singular ** 2)

SEASON_COLUMNS = {"season_sin": 0, "season_cos": 1} 

def parameters(raw, idx):
    scores = (raw['profile0'][idx] - pca['mean']) @ pca['components'].T
    season = [raw['season'][idx][:, SEASON_COLUMNS[p['name']]] for p in problem['input_parameters'] if p['name'] in SEASON_COLUMNS]
    return np.concatenate([scores, np.stack(season, axis=1)], axis=1) if SEASON else scores

def make_dataset(raw, idx):
    params = parameters(raw, idx)
    dataset = {
        'points': raw['x'][:, None],                                            # [num_points x num_coordinates]
        'times': raw['t'],                                                      # [num_times]
        'inp_parameters': params if params.shape[1] > 0 else None,              # [num_samples x (n_PCA + 2 if use_season)]
        'inp_signals': raw['sign'][idx],                                        # [num_samples x num_times x num_signals]
        'out_fields': raw['output'][idx][..., None],                            # [num_samples x num_times x num_points x num_fields]
    }
    utils.process_dataset(dataset, problem, normalization)
    dataset['sample_indices'] = idx
    real = np.asarray(raw['real'])[idx]
    mask = real & real[:, :1]        
    mask[:, 0] = False
    dataset['mask'] = tf.convert_to_tensor(mask, tf.float64)
    return dataset

def load_datasets():
    raw = np.load(DATA_PATH, allow_pickle=True)[()]
    expected = [s['name'] for s in problem['input_signals']]
    if list(raw.get('sign_names', [])) != expected:
        raise ValueError(f"data.npy has the signals {list(raw.get('sign_names', []))} but common.py expects {expected}: run parse.py again")
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise KeyError(f"{DATA_PATH} has no {missing}: run parse.py again")

    indices = [np.flatnonzero(np.asarray(raw["split"]) == k) for k in (0, 1, 2)]
    if min(len(i) for i in indices) == 0:
        raise ValueError(f"empty split (train {len(indices[0])}, valid {len(indices[1])}, test {len(indices[2])}): check the thresholds and the year blocks in parse.py")
    if n_PCA > min(len(indices[0]), raw["profile0"].shape[1]):
        raise ValueError(f"n_PCA = {n_PCA} is more than the training windows or the profile points")

    mean, components, explained = fit_pca(raw["profile0"][indices[0]], n_PCA)
    pca.update(mean=mean, components=components)
    train_scores = (raw["profile0"][indices[0]] - mean) @ components.T
    low, high = train_scores.min(axis=0), train_scores.max(axis=0)
    high = np.maximum(high, low + 1e-6)
    normalization['input_parameters'] = {f"PCA_{i}": {'min': float(l), 'max': float(h)} for i, (l, h) in enumerate(zip(low, high))}
    if SEASON:
        normalization['input_parameters'].update(season_sin={'min': -1.0, 'max': 1.0}, season_cos={'min': -1.0, 'max': 1.0})
    print(f"PCA of the day-0 profile: {n_PCA} components, {explained:.0f}% of the variance of the training profiles")

    datasets = []
    for name, idx in zip(("train", "valid", "test"), indices):
        datasets.append(make_dataset(raw, idx))
        print(f"{name:5s}: {len(idx)} windows")
    return tuple(datasets)

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

        input_shape_enc = (len(problem['input_parameters']),)
        self.NNenc = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape_enc),
            tf.keras.layers.Dense(enc_width, activation=tf.nn.tanh),
            tf.keras.layers.Dense(enc_width, activation=tf.nn.tanh),
            tf.keras.layers.Dense(num_latent_states)
        ])

        input_shape_dyn = (num_latent_states + len(problem['input_parameters']) - n_PCA + len(problem['input_signals']),)
        self.NNdyn = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape_dyn),
            tf.keras.layers.Dense(dyn_width, activation=tf.nn.tanh),
            tf.keras.layers.Dense(dyn_width, activation=tf.nn.tanh),
            tf.keras.layers.Dense(num_latent_states)
        ])

        input_shape_rec = (None, None, num_latent_states + len(problem['input_signals']) + problem['space']['dimension'])
        self.NNrec = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape_rec),
            tf.keras.layers.Dense(rec_width, activation=tf.nn.tanh),
            tf.keras.layers.Dense(rec_width, activation=tf.nn.tanh),
            tf.keras.layers.Dense(rec_width, activation=tf.nn.tanh),
            tf.keras.layers.Dense(rec_width, activation=tf.nn.tanh),
            tf.keras.layers.Dense(len(problem['output_fields']))
        ])
 
    def summary(self):
        self.NNenc.summary()
        self.NNdyn.summary()
        self.NNrec.summary()
 
    @property
    def trainable_variables(self):
        return self.NNenc.variables + self.NNdyn.variables + self.NNrec.variables
 
    def evolve_dynamics(self, dataset):
        parameters = dataset['inp_parameters'][:, n_PCA:]
        state = self.NNenc(dataset['inp_parameters'])
        state_history = tf.TensorArray(tf.float64, size=dataset['num_times'])
        state_history = state_history.write(0, state)
        dt_ref = self.normalization['time']['time_constant']
 
        for i in tf.range(dataset['num_times'] - 1):
            state = state + self.dt / dt_ref * self.NNdyn(tf.concat([state, parameters, dataset['inp_signals'][:, i, :]], axis=-1))
            state_history = state_history.write(i + 1, state)
 
        return tf.transpose(state_history.stack(), perm=(1, 0, 2))
 
    def reconstruct_output(self, dataset, states):
        outputs = []
        for start in range(0, dataset['num_samples'], self.reconstruction_batch_size):
            stop = min(start + self.reconstruction_batch_size, dataset['num_samples'])
            states_batch = states[start:stop]
            signals_batch = dataset['inp_signals'][start:stop]
            points_batch = dataset['points_full'][start:stop]

            states_expanded = tf.broadcast_to(
                tf.expand_dims(states_batch, axis=2),
                [stop - start, dataset['num_times'], dataset['num_points'], self.num_latent_states],
            )
            inp_signals_expanded = tf.broadcast_to(
                tf.expand_dims(signals_batch, axis=2),
                [stop - start, dataset['num_times'], dataset['num_points'], len(self.problem['input_signals'])],
            )
            outputs.append(self.NNrec(tf.concat([
                states_expanded,
                inp_signals_expanded,
                points_batch,
            ], axis=3)))

        output = tf.concat(outputs, axis=0)
        # nonlinear transformation to compress long tails
        alpha = 0.05
        output = (output ** 3 + alpha * output) / (1 + alpha)
        return output
    
    def __call__(self, dataset):
        states = self.evolve_dynamics(dataset)
        return self.reconstruct_output(dataset, states)
    
    # TODO: check save
 
    def save(self, model_dir=None):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        # Keras 3 native format requires the .keras extension.
        self.NNenc.save(str(model_dir / 'NNenc.keras'))
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
        model.NNenc = tf.keras.models.load_model(str(model_dir / 'NNenc.keras'))
        model.NNdyn = tf.keras.models.load_model(str(model_dir / 'NNdyn.keras'))
        model.NNrec = tf.keras.models.load_model(str(model_dir / 'NNrec.keras'))
        return model

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