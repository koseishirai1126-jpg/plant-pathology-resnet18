# 02: Datasetで画像1枚と正解ラベルを取り出す

## 今回作るもの

`dataset.py`の`PlantDataset`は、CSVから読み込んだ表と画像フォルダを受け取り、
指定した位置の「画像1枚と正解ラベル」を返します。
画像は必要になったときに1枚ずつ開きます。

```python
dataset = PlantDataset(train_df, "data/images", transform=transform)
image, label = dataset[0]
```

このクラスは、まず正解ラベルのある学習用・検証用データを対象にします。
正解ラベルのないテスト用データの読み込みは、予測コードを作る段階で扱います。

## 中心となる3つのメソッド

| メソッド | 呼ばれる場面 | 役割 |
| --- | --- | --- |
| `__init__` | `PlantDataset(...)` | CSVの表・画像フォルダ・前処理を保存し、正解ラベルをクラス番号に変換する |
| `__len__` | `len(dataset)` | 画像の枚数を返す |
| `__getitem__` | `dataset[0]`など | 指定された位置の画像を開き、前処理をして、画像と正解ラベルを返す |

`class PlantDataset(Dataset)`はPyTorchの`Dataset`を継承する書き方です。
`self`には、そのDataset自身が持つ表や設定を保存します。

`iloc[index]`は、DataFrameのインデックスの値ではなく「何番目の行か」で取り出します。
今後、学習用と検証用に表を分けた場合も、0番目から順番に指定できます。

## 正解ラベルを変換する

```python
CLASS_NAMES = ["healthy", "multiple_diseases", "rust", "scab"]
one_hot_labels = self.dataframe[CLASS_NAMES].to_numpy()
self.labels = one_hot_labels.argmax(axis=1)
```

`to_numpy()`で4列をNumPyの配列にし、`argmax(axis=1)`で行ごとに最大値の位置を求めます。
たとえば`[0, 0, 0, 1]`は`3`になります。
変換前に、値が0か1で、各行の合計が1になっているかも確認します。

`__getitem__`はラベルをPythonの整数で返します。
次の段階でDataLoaderを使うと、複数の整数ラベルがまとめてTensorになります。

## 画像の前処理

動作確認では`ResNet18_Weights.IMAGENET1K_V1.transforms()`を使います。

1. 画像の短辺を256ピクセルにリサイズする。
2. 中央の224×224ピクセルを切り出す。
3. RGBの画素値を0〜1の浮動小数点数に変換し、Tensorにする。
4. 学習済み重みが想定するRGBごとの平均・標準偏差で正規化する。

返される画像の形は`[3, 224, 224]`です。
先頭の`3`はRGBのチャンネル数、後ろの2つは高さと幅です。
正規化した後の数値には、0未満や1より大きい値も含まれます。

この前処理の設定を取得するだけでは、モデルの重みはダウンロードしません。
今回の前処理にはランダムな変換を含めていません。
学習時のランダムな変換は、学習用・検証用を分けてから追加します。

## 動作確認

プロジェクトのフォルダで実行します。

```bash
python check_dataset.py
```

データの場所を別途指定する場合は、次のように実行できます。

```bash
python check_dataset.py --data-dir /path/to/data
```

公式データの先頭3枚に対する想定出力は次のとおりです。

```text
画像の枚数: 1821
Train_0: shape=(3, 224, 224), dtype=torch.float32, label=3 (scab)
Train_1: shape=(3, 224, 224), dtype=torch.float32, label=1 (multiple_diseases)
Train_2: shape=(3, 224, 224), dtype=torch.float32, label=0 (healthy)
Datasetの動作確認OK
```

ここでは先頭3枚について、画像の読み込み・前処理・ラベルの取り出しを確認します。
学習・スコア計算は行いません。全画像の読み込み確認はまだ含みません。

## 参考

- [PyTorch公式: Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [torchvision 0.25: ResNet18の前処理](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.resnet18.html)
