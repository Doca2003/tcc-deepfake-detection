import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from evaluation.gradcam import load_trained_model, make_gradcam_heatmap, overlay_heatmap
from utils.dataset import IMG_SIZE, build_dataset, list_split_files, load_image

CONFIGS = ["A", "B", "C"]
INPUT_FOLDER = {"A": "faces", "B": "ela", "C": "ela"}
CONFIG_NAMES = {
    "A": "Config A: ResNet + rostos",
    "B": "Config B: ResNet + ELA",
    "C": "Config C: CNN própria + ELA",
}


def test_split(data_root, config, seed):
    data_dir = os.path.join(data_root, INPUT_FOLDER[config])
    paths, labels = list_split_files(data_dir, seed=seed)["test"]
    rels = [os.path.relpath(p, data_dir) for p in paths]
    return paths, labels, rels


def predict_all(model, paths, labels, batch_size=32):
    ds = build_dataset(paths, labels, batch_size, shuffle=False, seed=0)
    return model.predict(ds, verbose=0).ravel()


def choose_examples(models, data_root, seed, per_category):
    correct = {}
    rels, labels = None, None

    for cfg in CONFIGS:
        paths, cfg_labels, cfg_rels = test_split(data_root, cfg, seed)
        if rels is None:
            rels, labels = cfg_rels, cfg_labels
        assert cfg_rels == rels, "faces/ e ela/ precisam ter os mesmos arquivos de teste"
        probs = predict_all(models[cfg], paths, cfg_labels)
        correct[cfg] = (probs >= 0.5).astype(float) == np.array(cfg_labels)
        print(f"Config {cfg}: {correct[cfg].mean():.3f} de acertos no teste")

    masks = {
        "todos_acertam": correct["A"] & correct["B"] & correct["C"],
        "todos_erram": ~correct["A"] & ~correct["B"] & ~correct["C"],
        "so_A_acerta": correct["A"] & ~correct["B"] & ~correct["C"],
        "ELA_ajuda": ~correct["A"] & correct["B"],
    }

    rng = random.Random(seed)
    items = []
    for name, mask in masks.items():
        print(f"{name}: {int(mask.sum())} imagens no teste")
        candidates = [int(i) for i in np.flatnonzero(mask)]
        rng.shuffle(candidates)

        by_label = {0.0: [], 1.0: []}
        for i in candidates:
            by_label[labels[i]].append(i)
        queues = [by_label[0.0], by_label[1.0]]

        chosen, seen_videos, turn = [], set(), 0
        while len(chosen) < per_category and any(queues):
            queue = queues[turn % 2]
            turn += 1
            while queue:
                i = queue.pop()
                video = os.path.dirname(rels[i])
                if video not in seen_videos:
                    seen_videos.add(video)
                    chosen.append(i)
                    break

        for i in chosen:
            items.append({"category": name, "rel": rels[i], "label": labels[i]})

    return items


def render_example(models, item, data_root, out_path):
    rel, label = item["rel"], item["label"]
    face = np.array(Image.open(os.path.join(data_root, "faces", rel)).convert("RGB").resize(IMG_SIZE))
    ela = np.array(Image.open(os.path.join(data_root, "ela", rel)).convert("RGB").resize(IMG_SIZE))

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.9))
    axes[0].imshow(face)
    axes[0].set_title("Rosto original", fontsize=9)
    axes[1].imshow(ela)
    axes[1].set_title("Mapa ELA", fontsize=9)

    for ax, cfg in zip(axes[2:], CONFIGS):
        input_path = os.path.join(data_root, INPUT_FOLDER[cfg], rel)
        img, _ = load_image(input_path, 0.0)
        heatmap, prob = make_gradcam_heatmap(models[cfg], tf.expand_dims(img, 0))
        predicted = 1.0 if prob >= 0.5 else 0.0
        result = "acertou" if predicted == label else "errou"
        ax.imshow(overlay_heatmap(face, heatmap))
        ax.set_title(f"{CONFIG_NAMES[cfg]}\nP(fake) = {prob:.2f} ({result})", fontsize=9)

    for ax in axes:
        ax.axis("off")

    true_name = "fake" if label == 1 else "real"
    fig.suptitle(f"{item['category']} | Verdadeiro: {true_name} | {rel}", fontsize=10)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--out_dir", default="outputs/gradcam_selected")
    parser.add_argument("--per_category", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection", default=None)
    args = parser.parse_args()

    models = {cfg: load_trained_model(cfg, ckpt_dir=args.ckpt_dir) for cfg in CONFIGS}

    if args.selection:
        with open(args.selection) as f:
            items = json.load(f)
    else:
        items = choose_examples(models, args.data_root, args.seed, args.per_category)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "selection.json"), "w") as f:
        json.dump(items, f, indent=2)

    for item in items:
        name = os.path.splitext(item["rel"].replace(os.sep, "_"))[0]
        out_path = os.path.join(args.out_dir, item["category"], f"{name}.png")
        render_example(models, item, args.data_root, out_path)

    print(f"{len(items)} figuras salvas em {args.out_dir}")


if __name__ == "__main__":
    main()