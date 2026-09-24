from pathlib import Path
from network import utils
import numpy as np
import tensorflow as tf
from datetime import datetime

tf.keras.backend.set_floatx('float64')

start_date = datetime(1987, 1, 5)
dt = 24
dt_base = 24
num_latent_states = 30
n_points = 20

problem = {
    "space": {
        "dimension" : 1
    },
    "input_parameters": [],
    "input_signals": [
        { "name": "h" },
        { "name": "t" },
        { "name": "dir" },
        { "name": "tide" },
        *[
            {"name": f"{signal}_{point}"}
            for point in range(n_points)
            for signal in ("height", "difference")
        ],
    ],
    "output_fields": [
        { "name": "y" }
    ]
}

normalization = {
    'space': { 'min' : [-115], 'max' : [+385]},
    'time': {
        'time_constant' : dt_base
    },
    'input_signals': {
        'h': { 'min': 0, 'max': 8 },
        't': { 'min': 2.5, 'max': 18 },
        'dir': { 'min': 12, 'max': 168 },
        'tide': { 'min': 0.5, 'max': 3 },
        **{
            f'height_{point}': {'min': -700, 'max': 900}
            for point in range(n_points)
        },
        **{
            f'difference_{point}': {'min': -50, 'max': 50}
            for point in range(n_points)
        },
    },
    'output_fields': {
        'y': { 'min': -700, 'max': +900 }
    }
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / 'thesis' / 'data' /'data.npy'
MODEL_DIR = REPO_ROOT / 'thesis' / 'model' / 'trained_model'
HISTORY_PATH = REPO_ROOT / 'thesis' / 'model' / 'training_history.npz'

def load_datasets():
    dataset_train = utils.MY_create_dataset(DATA_PATH, np.arange(0, 120))
    dataset_valid = utils.MY_create_dataset(DATA_PATH, np.arange(120, 200))
    dataset_tests = utils.MY_create_dataset(DATA_PATH, np.arange(200, 240))

    utils.process_dataset(dataset_train, problem, normalization)
    utils.process_dataset(dataset_valid, problem, normalization)
    utils.process_dataset(dataset_tests, problem, normalization)

    return dataset_train, dataset_valid, dataset_tests

class LDNetModel:
    def __init__(self, problem, normalization, num_latent_states, dt, dt_base):
        self.problem = problem
        self.normalization = normalization
        self.num_latent_states = num_latent_states
        self.dt = dt
        self.dt_base = dt_base

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
            state = state + self.dt / dt_ref * self.NNdyn(tf.concat([state, dataset['inp_signals'][:, i, :]], axis=-1))
            state_history = state_history.write(i + 1, state)
 
        return tf.transpose(state_history.stack(), perm=(1, 0, 2))
 
    def reconstruct_output(self, dataset, states):
        states_expanded = tf.broadcast_to(tf.expand_dims(states, axis=2),
            [dataset['num_samples'], dataset['num_times'], dataset['num_points'], self.num_latent_states])
        inp_signals_expanded = tf.broadcast_to(tf.expand_dims(dataset['inp_signals'], axis=2),
            [dataset['num_samples'], dataset['num_times'], dataset['num_points'], len(self.problem['input_signals'])])
        output = self.NNrec(tf.concat([states_expanded, inp_signals_expanded, dataset['points_full']], axis=3))
        # nonlinear transformation to compress long tails
        alpha = 0.05
        output = (output ** 3 + alpha * output) / (1 + alpha)
        return output
 
    def __call__(self, dataset):
        states = self.evolve_dynamics(dataset)
        return self.reconstruct_output(dataset, states)
    
    # TODO: check from here on
 
    def save(self, model_dir=None):
        model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        # Keras 3 native format requires the .keras extension.
        self.NNdyn.save(str(model_dir / 'NNdyn.keras'))
        self.NNrec.save(str(model_dir / 'NNrec.keras'))
 
    @classmethod
    def load(cls, problem, normalization, num_latent_states, dt, dt_base, model_dir=None):
        model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
        model = cls(problem, normalization, num_latent_states, dt, dt_base)
        model.NNdyn = tf.keras.models.load_model(str(model_dir / 'NNdyn.keras'))
        model.NNrec = tf.keras.models.load_model(str(model_dir / 'NNrec.keras'))
        return model

weight_direction = 0.1
epsilon = 1e-4

def get_direction(velocity):
    return tf.math.divide(velocity, (epsilon + tf.expand_dims(tf.norm(velocity, axis=3), axis=-1)))
 
def make_loss_fn(model, dataset, target_velocity, target_direction):
    def loss_fn():
        velocity = model(dataset)
        MSE_velocity = tf.reduce_mean(tf.square(velocity - target_velocity))
        direction = get_direction(velocity)
        MSE_direction = tf.reduce_mean(tf.square(direction - target_direction))
        return MSE_velocity + weight_direction * MSE_direction
    return loss_fn