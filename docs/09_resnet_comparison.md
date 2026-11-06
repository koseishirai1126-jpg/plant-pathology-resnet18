# 09: ResNet18・34・50を同じ条件で比較する

## 今回変えるもの

モデルをResNet18からResNet34、ResNet50へ変更します。
まずは次の条件を揃えて比較します。

| 条件 | 指定 |
| --- | --- |
| 分割 | 保存済みのdata/split.csv。学習用1,456枚・検証用365枚 |
| 学習済み重み | 各モデルのImageNet IMAGENET1K_V1 |
| 画像の前処理 | 短辺256、中央224×224、Tensor化、ImageNet正規化 |
| データ拡張 | なし |
| エポック数 | 5 |
| バッチサイズ | 32 |
| 更新方法・学習率 | Adam、0.0001 |
| 更新する範囲 | モデル全体 |
| 乱数の設定 | 42 |
| GPU | NVIDIA GeForce RTX 5080 |
| モデル選択 | 検証用の平均ROC AUCが最良のエポック |

これは同じ学習条件での比較です。
各モデルに最適な学習率・エポック数を個別に探索した場合の最高性能を比較する実験ではありません。
深いモデルのほうが必ず高いスコアになるとは限りません。

## 重みと前処理を明示する理由

ResNet50のDEFAULTはIMAGENET1K_V2で、その前処理はリサイズの短辺が232です。
今回は3モデルともV1を明示し、短辺256・中央224×224という前処理を揃えます。
各アーキテクチャは、それぞれに対応する別の学習済み重みから開始します。

## コードの変更点

`model.py`に、モデル名と作成関数・重みを対応付ける表を追加しました。

```python
MODEL_CONFIGS = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1),
}
```

分類層の変更は、全モデルで同じ書き方を使います。

```python
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
```

ResNet18・34の入力特徴量は512、ResNet50は2048です。
`model.fc.in_features`から取得するため、この違いをコード側で決め打ちせず扱えます。
新しい保存ファイルにはモデル名も記録し、再評価・予測時に同じ構造を自動で復元します。
モデル名の記録がない既存のbaseline_01の重みは、ResNet18として復元します。

## 1. テストと既存モデルの読み込みを確認する

コンテナ内でコードを更新してから実行します。

```bash
python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 python evaluate.py --run-dir outputs/baseline_01
```

テストは、小さな仮画像で3種類の出力形、分類層、保存・復元後の予測の一致、
古い形式の重みの読み込み、前処理の設定を確認します。
テストでは学習済み重みをダウンロードしません。
続くevaluate.pyは、実際に保存したResNet18の検証予測が再現できるかを確認します。

## 2. ResNet34と50を学習する

順番に実行します。元のbaseline_01の結果と別のフォルダへ保存します。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet34 --epochs 5 --output-dir outputs/resnet34_01
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet50 --epochs 5 --output-dir outputs/resnet50_01
```

初回はそれぞれの学習済み重みをダウンロードします。
`--model`を省略した場合のモデルは、これまでどおりResNet18です。
保存先が既にある場合は、別の名前を指定します。

## 3. 結果を並べる

```bash
python compare_runs.py outputs/baseline_01 outputs/resnet34_01 outputs/resnet50_01
```

分割・前処理・学習条件・GPU・ライブラリの版が一致し、全エポックが完了していることを確認します。
その後、モデル名、最良エポック、検証AUC、検証損失、5エポック分の処理時間を表示します。
ここでの時間は、history.csvのseconds列の合計です。
データの読み込みと検証を含み、モデルのダウンロード時間は含みません。

| モデル | 保存先 | 最良検証AUC | Kaggle Public | Kaggle Private |
| --- | --- | ---: | ---: | ---: |
| ResNet18 | outputs/baseline_01 | 0.989535 | 0.94415 | 0.93658 |
| ResNet34 | outputs/resnet34_01 | 0.988262（epoch 4） | 0.94254 | 0.94203 |
| ResNet50 | outputs/resnet50_01 | 0.978693（epoch 3） | 0.95914 | 0.93802 |

利用者が実行して共有した結果に基づきます。
クラス別AUCと学習の推移の考察は[比較実験の記録](../experiments/resnet_comparison_01.md)にまとめています。
Kaggleの数値は利用者から共有された提出一覧の画像で確認しました。

同じ乱数の設定でも、GPUや実行環境をまたいで完全に同じ数値になるとは限りません。
まず今回の1回の比較結果を記録し、差が小さい場合は別の乱数で複数回確認することも検討できます。

## 4. 各モデルの再評価と提出CSV作成

学習結果を確認してから、各フォルダの最良モデルを再評価できます。

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --run-dir outputs/resnet34_01
CUDA_VISIBLE_DEVICES=0 python evaluate.py --run-dir outputs/resnet50_01
```

提出CSVを作る場合は、同じようにフォルダを指定します。

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py --run-dir outputs/resnet34_01
CUDA_VISIBLE_DEVICES=0 python predict.py --run-dir outputs/resnet50_01
```

作成するCSVはそれぞれの実験フォルダ内のsubmission.csvです。
再評価・予測時にモデル名を手動で指定する必要はありません。
Kaggleに提出した場合は、同じスコアの表示区分でResNet18の結果と比較します。

今回の実験では再評価・予測・手元への取得とファイルの一致確認まで完了しました。
手元に用意したモデル名付きの提出ファイルは、outputs/submissions/resnet34_01.csvとresnet50_01.csvです。

## 参考

- [torchvision 0.25: ResNet18](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.resnet18.html)
- [torchvision 0.25: ResNet34](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.resnet34.html)
- [torchvision 0.25: ResNet50](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.resnet50.html)
