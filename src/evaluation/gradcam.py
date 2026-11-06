import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from models.resnet_model import build_resnet
from models.cnn_custom import build_custom_cnn
from utils.dataset import load_image, IMG_SIZE


def load_trained_model(config, ckpt_dir="outputs/checkpoints"):
    if config in ("A", "B"):
        model = build_resnet(weights=None)
    else:
        model = build_custom_cnn()
    model.load_weights(os.path.join(ckpt_dir, f"config_{config}.weights.h5"))
    return model


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.Model)):
            return layer.name
    raise ValueError("Nenhuma camada convolucional encontrada")


def make_gradcam_heatmap(model, img_batch):
    conv_name = find_last_conv_layer(model)

    with tf.GradientTape() as tape:
        x = img_batch
        conv_outputs = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x)
            if layer.name == conv_name:
                conv_outputs = x
                tape.watch(conv_outputs)
        preds = x
        score = preds[:, 0]

    grads = tape.gradient(score, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    heatmap = tf.reduce_sum(conv_outputs * weights[:, tf.newaxis, tf.newaxis, :], axis=-1)[0]

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(preds[0, 0])

def overlay_heatmap(base_rgb, heatmap, alpha=0.4):
    h, w = base_rgb.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base_rgb, 1 - alpha, colored, alpha, 0)


def save_comparison(model, face_path, ela_path, input_path, out_path, true_label):
    img, _ = load_image(input_path, 0.0)
    heatmap, prob = make_gradcam_heatmap(model, tf.expand_dims(img, 0))

    face = np.array(Image.open(face_path).convert("RGB").resize(IMG_SIZE))
    ela = np.array(Image.open(ela_path).convert("RGB").resize(IMG_SIZE))
    overlay = overlay_heatmap(img.numpy().astype("uint8"), heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    panels = [(face, "Rosto original"), (ela, "Mapa ELA"), (overlay, "Grad-CAM")]
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    real_or_fake = "fake" if true_label == 1 else "real"
    fig.suptitle(f"Verdadeiro: {real_or_fake} | P(fake) = {prob:.2f}")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)