import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import common
from network import optimization
import tensorflow as tf
import numpy as np

dataset_train, dataset_valid, dataset_tests = common.load_datasets()
normalization = common.normalization
model_dt = common.dt
model_dir = common.MODEL_DIR
history_path = common.HISTORY_PATH

model = common.LDNetModel(common.problem, normalization, common.num_latent_states, model_dt, common.dt_base)
model.summary()

target_direction_train = common.get_direction(dataset_train['out_fields'])
target_direction_valid = common.get_direction(dataset_valid['out_fields'])
loss_train = common.make_loss_fn(
    model,
    dataset_train,
    dataset_train['out_fields'],
    target_direction_train
)
loss_valid = common.make_loss_fn(
    model,
    dataset_valid,
    dataset_valid['out_fields'],
    target_direction_valid
)

opt = optimization.OptimizationProblem(model.trainable_variables, loss_train, loss_valid)

num_epochs_Adam = 3000
num_epochs_BFGS = 1200

print('training (Adam)...')
opt.optimize_keras(num_epochs_Adam, tf.keras.optimizers.Adam(learning_rate=1e-4))
print('training (BFGS)...')
opt.optimize_BFGS(num_epochs_BFGS)

model.save(model_dir)

# TODO: check
np.savez(
history_path,
    iterations_history=np.array(opt.iterations_history),
    loss_train_history=np.array([float(v) for v in opt.loss_train_history]),
    loss_valid_history=np.array([float(v) for v in opt.loss_valid_history]),
    num_epochs_Adam=num_epochs_Adam
)

print(f'Model saved to {model_dir}')
print(f'History saved to {history_path}')