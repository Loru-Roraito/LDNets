import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import common
from network import optimization
import tensorflow as tf
import numpy as np
import json
import time

NUM_EPOCHS_ADAM = int(os.environ.get("LDNETS_EPOCHS_ADAM", 1000))
NUM_EPOCHS_BFGS = int(os.environ.get("LDNETS_EPOCHS_BFGS", 200))
LEARNING_RATE = 1e-4

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

print('training (Adam)...')
opt.optimize_keras(NUM_EPOCHS_ADAM, tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
print('training (BFGS)...')
opt.optimize_BFGS(NUM_EPOCHS_BFGS)

model.save(model_dir)

history_path.parent.mkdir(parents=True, exist_ok=True)
loss_valid_history = [float(v) for v in opt.loss_valid_history]
best = int(np.argmin(loss_valid_history))
np.savez(
    history_path,
    iterations_history=np.array(opt.iterations_history),
    loss_train_history=np.array([float(v) for v in opt.loss_train_history]),
    loss_valid_history=np.array([float(v) for v in opt.loss_valid_history]),
    num_epochs_Adam=NUM_EPOCHS_ADAM,
    best_epoch=opt.iterations_history[best]
)
print(f'Model saved to {model_dir}')
print(f'History saved to {history_path}')
 
report = common.report(model, dataset_train, dataset_valid, dataset_tests)

# everything run_experiments.py needs to compare runs
common.RESULTS_PATH.write_text(json.dumps({
    "run": common.RUN_NAME, "seed": common.SEED, "num_latent_states": common.num_latent_states,
    "N_PROFILE_POINTS": common.N_PROFILE_POINTS, "PROFILE_FEATURES": common.PROFILE_FEATURES, "output_mode": common.OUTPUT_MODE,
    "epochs_adam": NUM_EPOCHS_ADAM, "epochs_bfgs": NUM_EPOCHS_BFGS,
    "final_train_loss": float(opt.loss_train_history[-1]), "final_valid_loss": loss_valid_history[-1],
    "best_valid_loss": loss_valid_history[best], "best_epoch": int(opt.iterations_history[best]),
    "minutes": round((time.time() - start_time) / 60, 1), "report": report,
}, indent=2))
print(f'Results saved to {common.RESULTS_PATH}')