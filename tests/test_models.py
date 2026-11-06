"""モデルの切り替え、重みの復元、既存のResNet18との互換性を確認する。"""

import tempfile
import unittest
from pathlib import Path

import torch

from dataset import CLASS_NAMES
from model import MODEL_NAMES, build_model, get_preprocess, load_trained_model


class ModelTests(unittest.TestCase):
    def test_architectures_and_checkpoint_round_trip(self):
        # 学習済み重みはダウンロードせず、小さな仮画像で構造と復元を検証する。
        inputs = torch.randn(1, 3, 64, 64)
        expected_features = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}
        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name), tempfile.TemporaryDirectory() as folder:
                model = build_model(pretrained=False, model_name=model_name).eval()
                self.assertEqual(model.fc.in_features, expected_features[model_name])
                self.assertEqual(model.fc.out_features, 4)
                self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
                with torch.no_grad():
                    expected = model(inputs)
                self.assertEqual(tuple(expected.shape), (1, 4))

                checkpoint_path = Path(folder) / "model.pt"
                torch.save(
                    {
                        "model_name": model_name,
                        "pretrained_weights": "IMAGENET1K_V1",
                        "model_state_dict": model.state_dict(),
                        "class_names": CLASS_NAMES,
                        "epoch": 1,
                        "valid_auc": 0.5,
                    },
                    checkpoint_path,
                )
                restored, checkpoint = load_trained_model(checkpoint_path, torch.device("cpu"))
                with torch.no_grad():
                    actual = restored(inputs)
                torch.testing.assert_close(actual, expected)
                self.assertEqual(checkpoint["model_name"], model_name)
                self.assertFalse(restored.training)

    def test_original_resnet18_checkpoint(self):
        # 既存のbaseline_01と同じ、モデル名が保存されていない形式を確認する。
        with tempfile.TemporaryDirectory() as folder:
            model = build_model(pretrained=False).eval()
            checkpoint_path = Path(folder) / "legacy.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "epoch": 2,
                    "valid_auc": 0.989535,
                },
                checkpoint_path,
            )
            restored, checkpoint = load_trained_model(checkpoint_path, torch.device("cpu"))
            self.assertEqual(checkpoint["model_name"], "resnet18")
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)

    def test_shared_v1_preprocessing(self):
        # ResNet50のDEFAULT（V2、Resize 232）に切り替わらないことも確認する。
        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name):
                transform = get_preprocess(model_name)
                self.assertEqual(transform.resize_size, [256])
                self.assertEqual(transform.crop_size, [224])
                self.assertEqual(transform.mean, [0.485, 0.456, 0.406])
                self.assertEqual(transform.std, [0.229, 0.224, 0.225])

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            build_model(pretrained=False, model_name="resnet999")


if __name__ == "__main__":
    unittest.main()
