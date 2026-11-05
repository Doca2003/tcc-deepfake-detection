import json
import os
import tensorflow as tf

from utils.dataset import get_datasets
from models.resnet_model import build_resnet
from models.cnn_custom import build_custom_cnn
from evaluation.metrics import evaluate_model

CONFIGS = {
    "A":{"folder": "faces","builder": build_resnet},
    "B":{"folder": "ela", "builder": build_resnet},
    "C":{"folder":"ela","builder":build_custom_cnn},
}

def train_model(config, data_root="data/samples", epochs=5, batch_size=16,seed=42, output_dir="outputs"):
    tf.keras.utils.set_random_seed(seed)
    cfg = CONFIGS[config]

    data_dir = os.path.join(data_root, cfg["folder"])
    train_ds, val_ds, test_ds = get_datasets(data_dir, batch_size=batch_size, seed=seed)

    model = cfg["builder"]()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    results = evaluate_model(model, test_ds)

    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    model.save_weights(os.path.join(output_dir, "checkpoints", f"config_{config}.weights.h5"))
    with open(os.path.join(output_dir, "logs", f"config_{config}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results