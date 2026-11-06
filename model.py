"""4クラス分類用のResNet18・34・50を作り、保存済みの重みを復元する。"""

import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

from dataset import CLASS_NAMES


# モデル名を「作成する関数」と「使う学習済み重み」に対応させる。
# 比較時は全モデルでV1を明示し、ResNet50のDEFAULT（V2）は使わない。
MODEL_CONFIGS = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
}
MODEL_NAMES = tuple(MODEL_CONFIGS)


def get_model_weights(model_name):
    """指定したモデルに対応するImageNetの重みの設定を返す。"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"対応するモデル名は{MODEL_NAMES}です: {model_name}")
    return MODEL_CONFIGS[model_name][1]


def get_preprocess(model_name="resnet18"):
    """モデルの重みに対応する固定の前処理を返す。重みはダウンロードしない。"""
    return get_model_weights(model_name).transforms()


def build_model(pretrained=True, model_name="resnet18"):
    """学習開始時はImageNetの重みを使い、最後の分類層を作り直す。"""
    weights = get_model_weights(model_name)
    builder = MODEL_CONFIGS[model_name][0]
    model = builder(weights=weights if pretrained else None)

    # ImageNetの1000クラス用の層を、今回の4クラス用に置き換える。
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def load_trained_model(checkpoint_path, device):
    """train.pyで保存した重みを、同じ構造のモデルに読み込む。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["class_names"] != CLASS_NAMES:
        raise ValueError("保存したモデルとコードでクラスの並び順が異なります。")

    # 拡張前のResNet18の保存ファイルにはモデル名がないため、18として扱う。
    model_name = checkpoint.get("model_name", "resnet18")
    weights_name = get_model_weights(model_name).name
    if checkpoint.get("pretrained_weights", weights_name) != weights_name:
        raise ValueError("保存した重みとコードで、学習済み重みの設定が異なります。")
    checkpoint.setdefault("model_name", model_name)
    checkpoint.setdefault("pretrained_weights", weights_name)

    # 保存済みの重みで置き換えるので、ImageNetの重みはダウンロードしない。
    model = build_model(pretrained=False, model_name=model_name)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, checkpoint
