# 05: 実画像を学習済みResNet18に入力する

## 今回使うファイル

| ファイル | 役割 |
| --- | --- |
| `data_loading.py` | 保存した分割から、学習用・検証用のDatasetとDataLoaderを作る |
| `model.py` | 4クラス分類用のResNet18を作る |
| `check_model.py` | 検証用画像4枚で、予測・確率への変換・損失計算を確認する |

## 保存した分割を使う

`train.csv`と`split.csv`を`image_id`で結合します。
行番号で対応させると、CSVの行順が違った場合に別の画像のラベルを使うことになるためです。

IDの重複・欠損、全画像を含むこと、割り当てが`train`か`valid`であることも確認します。
1つの画像IDにつき割り当ては1つなので、学習用と検証用で画像を共有しません。
その後、学習用と検証用に分け、それぞれにDatasetとDataLoaderを作ります。

最初のベースラインでは、画像の前処理は両方とも固定します。
学習用の取り出す順番だけをシャッフルし、検証用の順番は固定します。
ランダムな画像変換によるデータ拡張は、今後の改善課題として扱えます。

## モデルを作る

```python
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
```

最初の行で、ImageNetで学習済みの重みを持つResNet18を読み込みます。
次の行で、最後の分類層を1,000クラス用から4クラス用に置き換えます。
`in_features`は最後の分類層に入力される特徴量の数で、ResNet18では512です。
`nn.Linear(512, 4)`は、512個の特徴量から4個の出力値を計算する全結合層です。

置き換えた分類層は新しく初期化されます。
それ以外の部分にはImageNetの学習済み重みが入っています。
今後の学習では、最後の層を含むモデル全体を更新する方針です。

`build_model(pretrained=False)`は、学習済みの重みを使わずに同じ構造を作ります。
今後、自分で保存した重みを読み込んで予測する際に使えるようにしています。

## GPUへ渡す

```python
model = model.to(device)
images = images.to(device)
labels = labels.to(device)
```

モデル、画像、正解ラベルを同じデバイスに配置します。
今回の実行コマンドでは`CUDA_VISIBLE_DEVICES=0`で使用するGPUを選びます。
プログラム内では、利用可能なら`cuda`、それ以外は`cpu`を使います。

## 出力と損失の関係

```python
logits = model(images)
loss = criterion(logits, labels)
probabilities = torch.softmax(logits, dim=1)
```

`logits`は各クラスに対する確率化前の出力です。
`CrossEntropyLoss`には、この値と整数の正解ラベルを渡します。
損失関数の入力にする前にsoftmaxを適用する必要はありません。

予測結果を確率として扱うときに、クラス方向（`dim=1`）へsoftmaxを適用します。
画像1枚あたり4クラス分の確率が得られ、その合計は約1になります。

| データ | 今回の形 |
| --- | --- |
| 入力画像 | `[4, 3, 224, 224]` |
| 正解ラベル | `[4]` |
| logits | `[4, 4]` |
| 予測確率 | `[4, 4]` |

## 今回の実行では学習しない

`model.eval()`は、BatchNormなどを検証・予測時の動作に切り替えます。
`torch.no_grad()`は、勾配の計算を無効にします。それぞれ役割が異なります。
モデルのパラメータ更新は行いません。

今回表示する損失は、今回の葉のデータで学習する前の、検証用4枚に対する値です。
分類層を作り直した直後なので、ベースラインの性能を示すスコアではありません。

## 実行方法

コンテナのプロジェクトフォルダで実行します。

```bash
CUDA_VISIBLE_DEVICES=0 python check_model.py
```

初回は約45 MBの重みがPyTorchの配布元からダウンロードされます。
通常はコンテナ内の`~/.cache/torch/hub/checkpoints/`にキャッシュされ、以後は再利用されます。

学習用1,456枚・検証用365枚と、上の表にある形が表示され、
最後に`ResNet18の実画像での動作確認OK`が出る想定です。
確率の合計には浮動小数点の計算による小さな誤差が含まれることがあります。
ここで確認するのは検証用の先頭4枚です。全画像の評価は次の段階で行います。

## 参考

- [torchvision 0.25: ResNet18](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.resnet18.html)
- [PyTorch 2.10: CrossEntropyLoss](https://docs.pytorch.org/docs/2.10/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch 2.10: Module](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Module.html)
