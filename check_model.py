"""検証用画像4枚で、指定したResNetの入力・出力・損失計算を確認する。"""

import argparse
from pathlib import Path

import torch
from torch import nn

from data_loading import make_dataloaders
from dataset import CLASS_NAMES
from model import MODEL_NAMES, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="resnet18")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    torch.manual_seed(42)
    train_loader, valid_loader = make_dataloaders(args.data_dir, batch_size=4, model_name=args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用するデバイス: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)

    # 初回はImageNetの学習済み重みをダウンロードする。
    model = build_model(pretrained=True, model_name=args.model).to(device)
    criterion = nn.CrossEntropyLoss()

    images, labels = next(iter(valid_loader))
    images = images.to(device)
    labels = labels.to(device)

    # 検証用の動作に切り替え、勾配計算を行わずに予測する。
    model.eval()
    with torch.no_grad():
        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)

    if logits.shape != (len(labels), len(CLASS_NAMES)):
        raise ValueError("モデルの出力の形が想定と異なります。")
    if not torch.isfinite(loss) or not torch.isfinite(probabilities).all():
        raise ValueError("損失または予測確率にNaNや無限大が含まれています。")
    probability_sums = probabilities.sum(dim=1)
    if not torch.allclose(probability_sums, torch.ones_like(probability_sums)):
        raise ValueError("予測確率の合計が1になっていません。")

    print(f"学習用: {len(train_loader.dataset)}枚")
    print(f"検証用: {len(valid_loader.dataset)}枚")
    print(f"分類層: {model.fc}")
    print(f"入力画像の形: {tuple(images.shape)}")
    print(f"正解ラベルの形: {tuple(labels.shape)}")
    print(f"モデル出力の形: {tuple(logits.shape)}")
    print(f"予測確率の形: {tuple(probabilities.shape)}")
    print(f"画像ごとの確率の合計: {probability_sums.cpu().tolist()}")
    print(f"損失（学習前の検証用4枚）: {loss.item():.6f}")
    print(f"{args.model.replace('resnet', 'ResNet')}の実画像での動作確認OK")


if __name__ == "__main__":
    main()
