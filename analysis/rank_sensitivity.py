"""固定した検証予測を小分けにして、評価する画像による順位の動きを調べる。"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from pathlib import Path


# PyTorchを読み込まず、CSVだけで集計できるようにする。
CLASS_NAMES = ("healthy", "multiple_diseases", "rust", "scab")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def binary_auc(labels, scores):
    """正例の順位和からAUCを計算する。同じ予測値には平均順位を使う。"""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUCの計算には正例と負例の両方が必要です。")
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    first_rank = 1
    for _, tied_pairs in itertools.groupby(ordered, key=lambda pair: pair[0]):
        group = list(tied_pairs)
        average_rank = first_rank + (len(group) - 1) / 2
        positive_rank_sum += average_rank * sum(label for _, label in group)
        first_rank += len(group)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate_subset(indices, targets, predictions):
    class_aucs = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        labels = [targets[index] == class_index for index in indices]
        scores = [predictions[index][class_index] for index in indices]
        class_aucs[class_name] = binary_auc(labels, scores)
    return {"mean_auc": sum(class_aucs.values()) / len(CLASS_NAMES), "class_aucs": class_aucs}


def winners(scores):
    best = max(scores.values())
    return [name for name, score in scores.items() if math.isclose(score, best, rel_tol=0, abs_tol=1e-12)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--run-dirs", nargs="+", type=Path,
        default=[Path("outputs/baseline_01"), Path("outputs/resnet34_01"), Path("outputs/resnet50_01")],
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("outputs/rank_sensitivity.json"))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trialsには1以上を指定してください。")

    truth = {}
    for row in read_csv(args.data_dir / "train.csv"):
        labels = [int(row[name]) for name in CLASS_NAMES]
        if sum(labels) != 1 or any(label not in (0, 1) for label in labels):
            raise ValueError("正解ラベルがone-hotではありません。")
        truth[row["image_id"]] = labels.index(1)

    model_predictions = {}
    image_ids = None
    full_results = {}
    input_hashes = {}
    for run_dir in args.run_dirs:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        verification = json.loads((run_dir / "verification.json").read_text(encoding="utf-8"))
        model_name = config["model"]
        if model_name in model_predictions:
            raise ValueError("モデル名が重複しています。")
        prediction_path = run_dir / "valid_predictions.csv"
        rows = read_csv(prediction_path)
        current_ids = [row["image_id"] for row in rows]
        if image_ids is None:
            image_ids = current_ids
        if current_ids != image_ids or len(set(current_ids)) != len(current_ids):
            raise ValueError("検証画像のID・行順がモデル間で一致しません。")
        predictions = [[float(row[name]) for name in CLASS_NAMES] for row in rows]
        if not all(math.isfinite(value) for prediction in predictions for value in prediction):
            raise ValueError("予測値にNaNまたは無限大があります。")
        model_predictions[model_name] = predictions
        targets = [truth[image_id] for image_id in image_ids]
        full = evaluate_subset(range(len(targets)), targets, predictions)
        for class_name in CLASS_NAMES:
            if not math.isclose(
                full["class_aucs"][class_name], verification["class_aucs"][class_name],
                rel_tol=0, abs_tol=1e-12,
            ):
                raise ValueError("順位和で計算したAUCが、記録されたscikit-learnの結果と一致しません。")
        full_results[model_name] = full
        input_hashes[model_name] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()

    groups = [[index for index, target in enumerate(targets) if target == class_index]
              for class_index in range(len(CLASS_NAMES))]
    if any(len(group) < 2 for group in groups):
        raise ValueError("各クラスに少なくとも2枚の検証画像が必要です。")
    rng = random.Random(args.seed)
    winner_counts = {name: 0 for name in model_predictions}
    different_winner_partitions = 0
    first_partition = None
    for trial in range(args.trials):
        half_a, half_b = [], []
        for group in groups:
            shuffled = group.copy()
            rng.shuffle(shuffled)
            middle = len(shuffled) // 2
            half_a.extend(shuffled[:middle])
            half_b.extend(shuffled[middle:])
        scores_a = {name: evaluate_subset(half_a, targets, predictions)["mean_auc"]
                    for name, predictions in model_predictions.items()}
        scores_b = {name: evaluate_subset(half_b, targets, predictions)["mean_auc"]
                    for name, predictions in model_predictions.items()}
        winners_a, winners_b = winners(scores_a), winners(scores_b)
        for name in winners_a + winners_b:
            winner_counts[name] += 1
        if set(winners_a) != set(winners_b):
            different_winner_partitions += 1
        if trial == 0:
            first_partition = {
                "a_images": len(half_a), "b_images": len(half_b),
                "a_scores": scores_a, "b_scores": scores_b,
                "a_winners": winners_a, "b_winners": winners_b,
            }

    report = {
        "method": "paired_stratified_halves_of_fixed_validation_predictions",
        "scope": "同じ検証画像を小分けにした感度分析。モデルの再学習・新しい画像での評価・交差検証は含まない。",
        "seed": args.seed, "partitions": args.trials, "evaluated_groups": args.trials * 2,
        "validation_images": len(image_ids),
        "class_counts": {name: len(group) for name, group in zip(CLASS_NAMES, groups)},
        "full_validation": full_results, "input_prediction_sha256": input_hashes,
        "top_counts_including_ties": winner_counts,
        "partitions_with_different_winners_between_halves": different_winner_partitions,
        "first_partition": first_partition,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("全検証画像のAUCと、記録されたscikit-learnの計算結果の一致確認OK")
    print(f"分割パターン数: {args.trials} / 評価した小集団: {args.trials * 2}")
    print(f"A・Bでトップが異なった分割: {different_winner_partitions}")
    print(f"各モデルがトップだった小集団数（同率を含む）: {winner_counts}")
    print("最初の分割:", json.dumps(first_partition, ensure_ascii=False))
    print("これは元の検証画像内での感度分析です。Kaggleでの順位逆転の原因や、新しい画像での性能を特定するものではありません。")
    print(f"保存先: {args.output}")


if __name__ == "__main__":
    main()
