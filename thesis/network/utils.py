import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import interpolate

def normalize_forw(v, v_min, v_max, axis = None):
    v_min, v_max = reshape_min_max(len(v.shape), v_min, v_max, axis)
    return (2.0*v - v_min - v_max) / (v_max - v_min)

def normalize_back(v, v_min, v_max, axis = None):
    v_min, v_max = reshape_min_max(len(v.shape), v_min, v_max, axis)
    return 0.5*(v_min + v_max + (v_max - v_min) * v)

def reshape_min_max(n, v_min, v_max, axis = None):
    if axis is not None:
        shape_min = [1] * n
        shape_max = [1] * n
        shape_min[axis] = len(v_min)
        shape_max[axis] = len(v_max)
        v_min = np.reshape(v_min, shape_min)
        v_max = np.reshape(v_max, shape_max)
    return v_min, v_max
    
def analyze_normalization(problem, normalization_definition):
    normalization = dict()
    normalization['dt_base'] = normalization_definition['time']['time_constant']
    normalization['x_min'] = np.array(normalization_definition['space']['min'])
    normalization['x_max'] = np.array(normalization_definition['space']['max'])
    if len(problem.get('input_parameters', [])) > 0:
        normalization['inp_parameters_min'] = np.array([normalization_definition['input_parameters'][v['name']]['min'] for v in problem['input_parameters']])
        normalization['inp_parameters_max'] = np.array([normalization_definition['input_parameters'][v['name']]['max'] for v in problem['input_parameters']])
    if len(problem.get('input_signals', [])) > 0:
        normalization['inp_signals_min'] = np.array([normalization_definition['input_signals'][v['name']]['min'] for v in problem['input_signals']])
        normalization['inp_signals_max'] = np.array([normalization_definition['input_signals'][v['name']]['max'] for v in problem['input_signals']])
    normalization['out_fields_min'] = np.array([normalization_definition['output_fields'][v['name']]['min'] for v in problem['output_fields']])
    normalization['out_fields_max'] = np.array([normalization_definition['output_fields'][v['name']]['max'] for v in problem['output_fields']])
    return normalization

def dataset_normalize(dataset, problem, normalization_definition):
    normalization = analyze_normalization(problem, normalization_definition)
    dataset['times']              = dataset['times'] / normalization['dt_base']
    dataset['points']             = normalize_forw(dataset['points']        , normalization['x_min']             , normalization['x_max']             , axis = 1)
    dataset['points_full']        = normalize_forw(dataset['points_full']   , normalization['x_min']             , normalization['x_max']             , axis = 3)
    if dataset['inp_parameters'] is not None:
        dataset['inp_parameters'] = normalize_forw(dataset['inp_parameters'], normalization['inp_parameters_min'], normalization['inp_parameters_max'], axis = 1)
    if dataset['inp_signals'] is not None:
        dataset['inp_signals']    = normalize_forw(dataset['inp_signals']   , normalization['inp_signals_min']   , normalization['inp_signals_max']   , axis = 2)
    dataset['out_fields']         = normalize_forw(dataset['out_fields']    , normalization['out_fields_min']    , normalization['out_fields_max']    , axis = 3)
    
def denormalize_output(out_fields, problem, normalization_definition):
    normalization = analyze_normalization(problem, normalization_definition)
    return normalize_back(out_fields , normalization['out_fields_min'], normalization['out_fields_max'], axis = 3)
    
def process_dataset(dataset, problem, normalization_definition, dt = None, num_points_subsample = None):
    if dt is not None:
        times = np.arange(dataset['times'][0], dataset['times'][-1] + dt * 1e-10, step = dt)
        if dataset['inp_signals'] is not None:
            dataset['inp_signals'] = interpolate.interp1d(dataset['times'], dataset['inp_signals'], axis = 1)(times)
        dataset['out_fields'] = interpolate.interp1d(dataset['times'], dataset['out_fields'], axis = 1)(times)
        dataset['output_real'] = interpolate.interp1d(
            dataset['times'],
            dataset['output_real'].astype(np.float64),
            axis=1,
            kind='nearest',
        )(times).astype(bool)
        dataset['times'] = times

    num_samples = dataset['out_fields'].shape[0]
    num_times = dataset['times'].shape[0]
    num_points = dataset['points'].shape[0]
    num_x = dataset['points'].shape[1]

    points_full = np.broadcast_to(dataset['points'][None,None,:,:], [num_samples, num_times, num_points, num_x])
    
    if num_points_subsample is None:
        dataset['points_full'] = points_full
    else:
        idxs = np.array([[np.random.choice(num_points, num_points_subsample) for j in range(num_times)] for i in range(num_samples)])
        dataset['points_full'] = np.array([[points_full          [i,j,idxs[i,j,:],:] for j in range(num_times)] for i in range(num_samples)])
        dataset['out_fields']  = np.array([[dataset['out_fields'][i,j,idxs[i,j,:],:] for j in range(num_times)] for i in range(num_samples)])

    dataset['num_points'] = dataset['points_full'].shape[2]
    dataset['num_times'] = num_times
    dataset['num_samples'] = num_samples

    dataset_normalize(dataset, problem, normalization_definition)

    if dataset['inp_parameters'] is not None:
        dataset['inp_parameters'] = tf.convert_to_tensor(dataset['inp_parameters'], tf.float64)
    if dataset['inp_signals'] is not None:
        dataset['inp_signals'] = tf.convert_to_tensor(dataset['inp_signals'], tf.float64)
    dataset['out_fields'] = tf.convert_to_tensor(dataset['out_fields'], tf.float64)
    dataset['output_real'] = tf.convert_to_tensor(dataset['output_real'], tf.bool)

def plot_output_1D(dataset, out_fields_ref, out_fields_app, n_row, n_col, title_ROM = 'ROM'):
    fig = plt.figure(figsize=(10, 8), constrained_layout=False)
    outer_grid = fig.add_gridspec(n_row, n_col, wspace=1e-1, hspace=3e-1)

    t = dataset['times']
    x = dataset['points']
    X, T = np.meshgrid(x,t)

    vmin, vmax = np.min(out_fields_ref), np.max(out_fields_ref)
    for i_sample in range(n_col*n_row):
        idx_col = i_sample % n_col
        idx_row = int((i_sample - idx_col) / n_col)

        inner_grid = outer_grid[idx_row, idx_col].subgridspec(1, 2, wspace=0, hspace=0)
        axs = inner_grid.subplots()

        axs[0].pcolormesh(X, T, out_fields_ref[i_sample,:,:,0], shading='auto', vmin = vmin, vmax = vmax)
        axs[1].pcolormesh(X, T, out_fields_app[i_sample,:,:,0], shading='auto', vmin = vmin, vmax = vmax)
        axs[0].set_title('FOM')
        axs[1].set_title(title_ROM)

        for ax in axs.flatten():
            ax.set(xticks=[], yticks=[])
            for spine in ['top', 'bottom', 'left', 'right']: ax.spines[spine].set_visible(True)
    return fig

def plot_time_slices_1D(dataset, out_fields_ref, out_fields_app, n_row, n_col, sample_idx = 0, time_indices = None, title_ROM = 'ROM'):
    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    outer_grid = fig.add_gridspec(n_row, n_col, wspace=0.25, hspace=0.35)

    x = dataset['points']
    t = dataset['times']
    if time_indices is None:
        time_indices = np.linspace(0, len(t) - 1, n_row * n_col, dtype=int)

    for i_subplot, time_idx in enumerate(time_indices[:n_row * n_col]):
        idx_col = i_subplot % n_col
        idx_row = int((i_subplot - idx_col) / n_col)
        ax = fig.add_subplot(outer_grid[idx_row, idx_col])

        ax.plot(x, out_fields_ref[sample_idx, time_idx, :, 0], label='Original', color='black', linewidth=1.5)
        ax.plot(x, out_fields_app[sample_idx, time_idx, :, 0], label=title_ROM, color='tab:orange', linewidth=1.5, linestyle='--')
        ax.set_title(f't = {t[time_idx]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('height')
        ax.grid(True, alpha=0.25)

        if i_subplot == 0:
            ax.legend(loc='best')

    return fig

def MY_create_dataset(dataset_path, idxs):
    print('loading dataset %s' % dataset_path)
    dataset = np.load(dataset_path, allow_pickle = True)[()]
    print('loaded dataset')
    output_real = dataset.get(
        'output_real',
        np.ones(dataset['output'].shape[:2], dtype=bool),
    )

    new_dataset = {
        'points' : dataset['x'][:, None], # [num_points x num_coordinates]
        'times' : dataset['t'], # [num_times]
        'inp_parameters' : None, # [num_samples x num_par]
        'inp_signals' : dataset['sign'][idxs,:,:], # [num_samples x num_times x num_signals]
        'out_fields' : dataset['output'][idxs,:,:,None], # [num_samples x num_times x num_points x num_fields]
        'output_real' : output_real[idxs,:], # [num_samples x num_times]
    }

    return new_dataset