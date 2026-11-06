"""4クラス分類用のResNet18を作る。"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from dataset import CLASS_NAMES


def build_model(pretrained=True):
    """学習開始時はImageNetの重みを使い、最後の分類層を作り直す。"""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # ImageNetの1000クラス用の層を、今回の4クラス用に置き換える。
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def load_trained_model(checkpoint_path, device):
    """train.pyで保存した重みを、同じ構造のモデルに読み込む。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("保存したモデルとコードでクラスの並び順が異なります。")

    # 保存済みの重みで置き換えるので、ImageNetの重みはダウンロードしない。
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, checkpoint
