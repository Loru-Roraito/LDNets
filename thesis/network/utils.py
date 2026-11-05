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
        if dataset['inp_parameters'] is not None:
            dataset['inp_parameters'] = interpolate.interp1d(dataset['times'], dataset['inp_parameters'], axis = 1)(times)
        dataset['out_fields'] = interpolate.interp1d(dataset['times'], dataset['out_fields'], axis = 1)(times)
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