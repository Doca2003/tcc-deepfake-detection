import argparse
from training.train import train_model

parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=["A", "B", "C"], required=True)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--data_root", default="data/samples")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", default="outputs")
args = parser.parse_args()

results = train_model(args.config, args.data_root, args.epochs, args.batch_size,
                      seed=args.seed, output_dir=args.output_dir)
print(results)