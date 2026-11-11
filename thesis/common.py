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

start_date = datetime(1987, 1, 5)     # same as START_DATE in parse.py (sample_dates are hours since this day)
dt = 24
dt_hourly = 1
dt_base = 24
num_latent_states = 5
n_PCA = 10

problem = {
    "space": {
        "dimension" : 1
    },
    "input_parameters": [
        *[
            {"name": f"PCA_{point}"}
            for point in range(n_PCA)
        ]
    ],
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
        { "name": "y" }
    ]
}

normalization = {
    'space': { 'min' : [-115], 'max' : [385]},
    'time': {
        'time_constant' : dt_base
    },
    'input_signals': {
        'hmean':    { 'min': 0.22, 'max': 13.72 },
        'hmax':     { 'min': 0.28, 'max': 22.18 },

        'tmean':    { 'min': 5.37, 'max': 12.53 },
        'tmax':     { 'min': 4.7, 'max': 13.8 },

        'dirmean':  { 'min': 0.55, 'max': 1.00 },
        'dirmax':   { 'min': 0.31, 'max': 1.00 },

        'tidemean': { 'min': 1.48, 'max': 2.02 },
        'tidemax':  { 'min': 0.87, 'max': 2.36 }
    },
    'input_parameters': {},     # PCA_0 ... PCA_n: filled by load_datasets from the training windows
    'output_fields': {
        'y': { 'min': 0, 'max': 150 }
    }
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / 'thesis' / 'data' /'data.npy'
MODEL_DIR = REPO_ROOT / 'thesis' / 'model' / 'trained_model'
HISTORY_PATH = REPO_ROOT / 'thesis' / 'model' / 'training_history.npz'

REQUIRED_KEYS = ("x", "t", "sign", "output", "profile0", "split", "sample_dates")

_pca = {}    # mean profile and principal components, set by load_datasets (from the training windows)

def fit_pca(profile, n_components):
    """Principal components of the profiles [windows, points]: mean, components [n, points], % of variance explained."""
    mean = profile.mean(axis=0)
    _, singular, components = np.linalg.svd(profile - mean, full_matrices=False)
    components = components[:n_components]  
    components = components * np.sign(components[np.arange(n_components), np.abs(components).argmax(axis=1)])[:, None]   # fixed sign: same result every run
    return mean, components, 100 * np.sum(singular[:n_components] ** 2) / np.sum(singular ** 2)

def make_dataset(raw, idx):
    """Dataset of the windows idx (rows of the data file). Needs load_datasets to have run first (PCA and normalization)."""
    dataset = {
        'points': raw['x'][:, None],                                            # [num_points x num_coordinates]
        'times': raw['t'],                                                      # [num_times]
        'inp_parameters': (raw['profile0'][idx] - _pca['mean']) @ _pca['components'].T,   # [num_samples x n_PCA]
        'inp_signals': raw['sign'][idx],                                        # [num_samples x num_times x num_signals]
        'out_fields': raw['output'][idx][..., None],                            # [num_samples x num_times x num_points x num_fields]
    }
    utils.process_dataset(dataset, problem, normalization)
    dataset['sample_indices'] = idx
    return dataset

def load_datasets():
    raw = np.load(DATA_PATH, allow_pickle=True)[()]
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise KeyError(f"{DATA_PATH} has no {missing}: run parse.py again")

    # the split comes from parse.py: 0 train, 1 validation, 2 test, -1 not used (low quality)
    indices = [np.flatnonzero(np.asarray(raw["split"]) == k) for k in (0, 1, 2)]
    if min(len(i) for i in indices) == 0:
        raise ValueError(f"empty split (train {len(indices[0])}, valid {len(indices[1])}, test {len(indices[2])}): check the thresholds and the year blocks in parse.py")
    if n_PCA > min(len(indices[0]), raw["profile0"].shape[1]):
        raise ValueError(f"n_PCA = {n_PCA} is more than the training windows or the profile points")

    # PCA of the day-0 profile, fitted on the training windows only; each score is normalized with its training range
    mean, components, explained = fit_pca(raw["profile0"][indices[0]], n_PCA)
    _pca.update(mean=mean, components=components)
    train_scores = (raw["profile0"][indices[0]] - mean) @ components.T
    low, high = train_scores.min(axis=0), train_scores.max(axis=0)
    high = np.maximum(high, low + 1e-6)
    normalization['input_parameters'] = {f"PCA_{i}": {'min': float(l), 'max': float(h)} for i, (l, h) in enumerate(zip(low, high))}
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

        input_shape_dyn = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)
        self.NNdyn = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape_dyn),
            tf.keras.layers.Dense(32, activation=tf.nn.tanh),
            tf.keras.layers.Dense(32, activation=tf.nn.tanh),
            tf.keras.layers.Dense(num_latent_states)
        ])

        input_shape_rec = (None, None, num_latent_states + len(problem['input_signals']) + problem['space']['dimension'])
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
        state = tf.zeros((dataset['num_samples'], self.num_latent_states), dtype=tf.float64)
        state_history = tf.TensorArray(tf.float64, size=dataset['num_times'])
        state_history = state_history.write(0, state)
        dt_ref = self.normalization['time']['time_constant']
 
        for i in tf.range(dataset['num_times'] - 1):
            state = state + self.dt / dt_ref * self.NNdyn(tf.concat([state, dataset['inp_parameters'], dataset['inp_signals'][:, i, :]], axis=-1))
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
 
    def _reconstruct_timestep(self, dataset, state, inp_signals):
        states_expanded = tf.broadcast_to(
            tf.expand_dims(tf.expand_dims(state, axis=1), axis=1),
            [dataset['num_samples'], 1, dataset['num_points'], self.num_latent_states],
        )
        inp_signals_expanded = tf.broadcast_to(
            tf.expand_dims(tf.expand_dims(inp_signals, axis=1), axis=1),
            [dataset['num_samples'], 1, dataset['num_points'], len(self.problem['input_signals'])],
        )
        output = self.NNrec(tf.concat([
            states_expanded,
            inp_signals_expanded,
            dataset['points_full'][:, 0:1, :, :],
        ], axis=3))
        alpha = 0.05
        output = (output ** 3 + alpha * output) / (1 + alpha)
        return output[:, 0, :, :]
    
    def __call__(self, dataset):
        states = self.evolve_dynamics(dataset)
        return self.reconstruct_output(dataset, states)
    
    # TODO: check from here on
 
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