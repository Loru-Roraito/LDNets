import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import common
from network import optimization
import tensorflow as tf
import numpy as np
import json
import time

start_time = time.time()
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

num_epochs_Adam = common.num_epochs_Adam        # all of these are settings at the top of common.py
num_epochs_BFGS = common.num_epochs_BFGS

print('training (Adam)...')
opt.optimize_keras(num_epochs_Adam, tf.keras.optimizers.Adam(learning_rate=common.learning_rate))
if num_epochs_BFGS > 0:
    print('training (BFGS)...')
    opt.optimize_BFGS(num_epochs_BFGS)

model.save(model_dir)

# the settings of this run, next to the model (the folder name has the main ones too)
settings = {name: getattr(common, name) for name in ("num_latent_states", "n_PCA", "dyn_width", "rec_width", "weight_direction",
                                                     "learning_rate", "num_epochs_Adam", "num_epochs_BFGS", "SEED")}
settings["input_parameters"] = [p["name"] for p in common.problem["input_parameters"]]
(common.RUN_DIR / "settings.json").write_text(json.dumps(settings, indent=2))

# TODO: check
history_path.parent.mkdir(parents=True, exist_ok=True)
loss_valid_history = [float(v) for v in opt.loss_valid_history]
best = int(np.argmin(loss_valid_history))
np.savez(
history_path,
    iterations_history=np.array(opt.iterations_history),
    loss_train_history=np.array([float(v) for v in opt.loss_train_history]),
    loss_valid_history=np.array([float(v) for v in opt.loss_valid_history]),
    num_epochs_Adam=num_epochs_Adam,
    best_epoch=opt.iterations_history[best]
)

print(f'Model saved to {model_dir}')
print(f'History saved to {history_path}')
print(f'Run folder: {common.RUN_DIR}')

report = common.report(model, dataset_train, dataset_valid, dataset_tests)