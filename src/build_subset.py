import argparse
import json
import os
import random
import tempfile
import urllib.request

from preprocessing.extract_frames import extract_frames
from preprocessing.face_detection import detect_and_crop_face
from preprocessing.ela import apply_ela

METHODS = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
SPLITS = ["train", "val", "test"]
SPLIT_URL = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{}.json"


def get_pairs_by_split(base, seed):
    try:
        pairs = {}
        for split in SPLITS:
            with urllib.request.urlopen(SPLIT_URL.format(split), timeout=30) as r:
                pairs[split] = [tuple(p) for p in json.load(r)]
        print("Split OFICIAL do FF++ carregado.")
        return pairs
    except Exception as e:
        print(f"ATENCAO: nao consegui baixar o split oficial ({e}). "
              "Usando split alternativo, NAO oficial.")

    names = os.listdir(os.path.join(base, "Deepfakes"))
    all_pairs = sorted({tuple(sorted(n[:-4].split("_"))) for n in names if n.endswith(".mp4")})
    random.Random(seed).shuffle(all_pairs)
    n_eval = round(len(all_pairs) * 0.14)
    return {
        "val": all_pairs[:n_eval],
        "test": all_pairs[n_eval:2 * n_eval],
        "train": all_pairs[2 * n_eval:],
    }


def build_jobs(base, pairs_by_split, n_pairs, seed):
    rng = random.Random(seed)
    jobs = []
    fake_i = 0
    for split in SPLITS:
        pairs = sorted(pairs_by_split[split])
        rng.shuffle(pairs)
        for a, b in pairs[:n_pairs[split]]:
            for vid in (a, b):
                jobs.append((os.path.join(base, "original", f"{vid}.mp4"),
                             split, "real", f"{vid}_real"))
            for src, dst in ((a, b), (b, a)):
                method = METHODS[fake_i % len(METHODS)]
                fake_i += 1
                name = f"{src}_{dst}"
                jobs.append((os.path.join(base, method, f"{name}.mp4"),
                             split, "fake", f"{method}_{name}"))
    return jobs


def process_video(video_path, split, label, name, out_root, frames):
    face_dir = os.path.join(out_root, "faces", split, label, name)
    ela_dir = os.path.join(out_root, "ela", split, label, name)
    if os.path.isdir(ela_dir) and os.listdir(ela_dir):
        return 0, 0

    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(ela_dir, exist_ok=True)
    total, ok = 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        extract_frames(video_path, tmp, num_frames=frames)
        for frame_file in sorted(os.listdir(tmp)):
            total += 1
            face_path = os.path.join(face_dir, frame_file)
            if detect_and_crop_face(os.path.join(tmp, frame_file), face_path):
                apply_ela(face_path, os.path.join(ela_dir, frame_file))
                ok += 1
    return total, ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="/kaggle/input/datasets/xdxd003/ff-c23/FaceForensics++_C23")
    parser.add_argument("--out", default="/kaggle/working/data")
    parser.add_argument("--train_pairs", type=int, default=90)
    parser.add_argument("--val_pairs", type=int, default=15)
    parser.add_argument("--test_pairs", type=int, default=15)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = get_pairs_by_split(args.base, args.seed)
    n_pairs = {"train": args.train_pairs, "val": args.val_pairs, "test": args.test_pairs}
    jobs = build_jobs(args.base, pairs, n_pairs, args.seed)
    print(f"{len(jobs)} videos para processar")

    all_total, all_ok = 0, 0
    for i, (video_path, split, label, name) in enumerate(jobs, 1):
        if not os.path.exists(video_path):
            print(f"Video nao encontrado, pulando: {video_path}")
            continue
        total, ok = process_video(video_path, split, label, name, args.out, args.frames)
        all_total += total
        all_ok += ok
        if i % 20 == 0:
            print(f"{i}/{len(jobs)} videos processados")

    print(f"Rostos detectados: {all_ok}/{all_total}")
    for split in SPLITS:
        for label in ["real", "fake"]:
            d = os.path.join(args.out, "ela", split, label)
            n = sum(len(files) for _, _, files in os.walk(d)) if os.path.isdir(d) else 0
            print(f"{split}/{label}: {n} imagens")


if __name__ == "__main__":
    main()