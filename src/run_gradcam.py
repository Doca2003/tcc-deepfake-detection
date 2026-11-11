import argparse
import os

from evaluation.gradcam import load_trained_model, save_comparison
from utils.dataset import list_split_files

parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=["A", "B", "C"], required=True)
parser.add_argument("--n_per_class", type=int, default=3)
parser.add_argument("--per_video", type=int, default=1)
parser.add_argument("--data_root", default="data/samples")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ckpt_dir", default="outputs/checkpoints")
args = parser.parse_args()

folder = "faces" if args.config == "A" else "ela"
data_dir = os.path.join(args.data_root, folder)

paths, labels = list_split_files(data_dir, seed=args.seed)["test"]
model = load_trained_model(args.config, ckpt_dir=args.ckpt_dir)

out_dir = os.path.join("outputs", "gradcam_results", f"config_{args.config}")
class_count = {0.0: 0, 1.0: 0}
video_count = {}

for path, label in zip(paths, labels):
    rel = os.path.relpath(path, data_dir)
    video = os.path.dirname(rel)

    if class_count[label] >= args.n_per_class:
        continue
    if video_count.get(video, 0) >= args.per_video:
        continue

    out_name = os.path.splitext(rel.replace(os.sep, "_"))[0] + ".png"
    save_comparison(
        model,
        face_path=os.path.join(args.data_root, "faces", rel),
        ela_path=os.path.join(args.data_root, "ela", rel),
        input_path=path,
        out_path=os.path.join(out_dir, out_name),
        true_label=label,
    )
    class_count[label] += 1
    video_count[video] = video_count.get(video, 0) + 1

print(f"Imagens salvas em {out_dir}")