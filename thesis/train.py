import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import common
from network import optimization
import tensorflow as tf
import numpy as np

dataset_train, dataset_valid, dataset_tests = common.load_datasets()

model = common.LDNetModel(common.problem, common.normalization, common.num_latent_states, common.dt, common.dt_base)
model.summary()

target_direction_train = common.get_direction(dataset_train['out_fields'])
target_direction_valid = common.get_direction(dataset_valid['out_fields'])
autoregressive_training = True
loss_train = common.make_loss_fn(
    model,
    dataset_train,
    dataset_train['out_fields'],
    target_direction_train,
    autoregressive=autoregressive_training,
)
loss_valid = common.make_loss_fn(
    model,
    dataset_valid,
    dataset_valid['out_fields'],
    target_direction_valid,
    autoregressive=autoregressive_training,
)

opt = optimization.OptimizationProblem(model.trainable_variables, loss_train, loss_valid)

num_epochs_Adam = 3000
num_epochs_BFGS = 1200

print('training (Adam)...')
opt.optimize_keras(num_epochs_Adam, tf.keras.optimizers.Adam(learning_rate=1e-4))
print('training (BFGS)...')
opt.optimize_BFGS(num_epochs_BFGS)

model.save(common.MODEL_DIR)

# TODO: check
np.savez(
    common.HISTORY_PATH,
    iterations_history=np.array(opt.iterations_history),
    loss_train_history=np.array([float(v) for v in opt.loss_train_history]),
    loss_valid_history=np.array([float(v) for v in opt.loss_valid_history]),
    num_epochs_Adam=num_epochs_Adam
)

print(f'Model saved to {common.MODEL_DIR}')
print(f'History saved to {common.HISTORY_PATH}')