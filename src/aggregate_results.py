import glob
import json
import os
import re

import numpy as np

METRICS = ["accuracy", "precision", "recall", "f1", "auc"]


def main(root="outputs"):
    runs = {}
    pattern = os.path.join(root, "**", "config_*_results.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        config = re.search(r"config_([ABC])_results", os.path.basename(path)).group(1)
        with open(path) as f:
            result = json.load(f)
        if not all(m in result for m in METRICS):
            print(f"Ignorando (formato antigo): {path}")
            continue
        runs.setdefault(config, []).append(result)

    lines = ["| Config | n | " + " | ".join(METRICS) + " |",
             "|---|---|" + "---|" * len(METRICS)]
    for config in sorted(runs):
        results = runs[config]
        cells = []
        for metric in METRICS:
            values = np.array([r[metric] for r in results])
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            cells.append(f"{values.mean():.3f} ± {std:.3f}")
        lines.append(f"| {config} | {len(results)} | " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    print(table)
    with open(os.path.join(root, "resumo_metricas.md"), "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()