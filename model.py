"""4クラス分類用のResNet18を作る。"""

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
