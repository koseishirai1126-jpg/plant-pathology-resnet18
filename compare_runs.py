"""同じ条件で完了した実験の、最良AUCと学習時間を一覧にする。"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


# モデル名以外の、今回の比較で揃える条件。
COMPARISON_KEYS = (
    "pretrained_weights", "class_names", "epochs", "batch_size", "learning_rate", "optimizer", "seed",
    "preprocessing", "augmentation", "split_sha256", "train_csv_sha256",
    "train_images", "valid_images", "device", "versions",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    reference_config = None
    results = []

    for run_dir in args.run_dirs:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        best = json.loads((run_dir / "best_metrics.json").read_text(encoding="utf-8"))
        with (run_dir / "history.csv").open(encoding="utf-8", newline="") as file:
            history = list(csv.DictReader(file))
        if [int(row["epoch"]) for row in history] != list(range(1, config["epochs"] + 1)):
            raise ValueError(f"{run_dir}: 指定した全エポックの実行が完了していません。")
        best_row = max(history, key=lambda row: float(row["valid_auc"]))
        if int(best_row["epoch"]) != best["epoch"] or not math.isclose(
            float(best_row["valid_auc"]), best["valid_auc"], rel_tol=0, abs_tol=1e-12,
        ):
            raise ValueError(f"{run_dir}: 履歴と最良エポックの記録が一致しません。")
        split_hash = hashlib.sha256((run_dir / "split.csv").read_bytes()).hexdigest()
        if split_hash != config["split_sha256"]:
            raise ValueError(f"{run_dir}: 保存された分割が記録と異なります。")

        if reference_config is None:
            reference_config = config
        else:
            differences = [key for key in COMPARISON_KEYS if config[key] != reference_config[key]]
            if differences:
                raise ValueError(f"{run_dir}: 比較条件が異なります: {', '.join(differences)}")

        results.append(
            {
                "model": config["model"],
                "epoch": best["epoch"],
                "auc": best["valid_auc"],
                "loss": best["valid_loss"],
                "seconds": sum(float(row["seconds"]) for row in history),
                "run_dir": run_dir,
            }
        )

    print("分割・前処理・学習条件・GPUの一致確認OK")
    print(f"{'model':<12} {'best_epoch':>10} {'valid_auc':>10} {'valid_loss':>11} {'total_sec':>10}")
    for result in results:
        print(
            f"{result['model']:<12} {result['epoch']:>10} {result['auc']:>10.6f} "
            f"{result['loss']:>11.6f} {result['seconds']:>10.2f}  {result['run_dir']}"
        )


if __name__ == "__main__":
    main()
