# 07: 保存した重みを読み込み、テスト画像を予測する

## 保存したモデルを復元する

`model.py`に`load_trained_model`を追加しました。

```python
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model = build_model(pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model = model.to(device)
model.eval()
```

同じ構造のResNet18を作り、保存した全パラメータとBatchNormの統計値を読み込みます。
モデルの構造やクラスの順番が合わない場合は停止します。
ImageNetの重みを新たにダウンロードする必要はありません。

`map_location="cpu"`は、まずCPUへ重みを読み込む指定です。
その後、利用するデバイスにモデル全体を移します。

## 1. 保存したモデルを再評価する

コンテナのプロジェクトフォルダで実行します。

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --run-dir outputs/baseline_01
```

次のことを確認します。

- 現在のtrain.csvとsplit.csvが、学習時に記録したハッシュと一致する。
- 実行フォルダに保存した分割のコピーも一致する。
- 保存したモデルを読み込み、検証用365枚を再度予測できる。
- 再度予測した確率が、保存しておいた検証予測と小さな数値誤差の範囲で一致する。

学習時と同じバッチサイズをconfig.jsonから読み込みます。
結果を`outputs/baseline_01/verification.json`に保存します。
今回のベースラインでは保存されたエポックが2、保存時のAUCが約0.989535になる想定です。

この実行はモデルのパラメータを更新しません。

## 2. 正解ラベルのないテスト画像を読み込む

`dataset.py`に`PlantTestDataset`を追加しました。
画像の前処理は学習時の検証用と同じです。
正解ラベルがないため、画像と画像IDを返します。

```python
image, image_id = test_dataset[0]
```

DataLoaderでまとめて取り出した後も、IDを予測と一緒に保持します。
これにより、どの画像の予測かを追跡できます。

## 3. 提出CSVを作る

再評価の確認が通ったら実行します。

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py --run-dir outputs/baseline_01
```

保存したbest.ptを使い、テスト画像1,821枚を予測します。
`model.eval()`と`torch.no_grad()`を使い、モデル出力をsoftmaxで4クラスの確率へ変換します。
学習時に検証スコアで選んだモデルを、そのままテスト画像の予測に使います。

提出CSVは次の場所に保存します。

```text
outputs/baseline_01/submission.csv
```

列の順番は次のとおりです。

```csv
image_id,healthy,multiple_diseases,rust,scab
```

画像IDを使って提出見本の行順に揃え、以下を確認してから保存します。

- 画像IDに欠損や重複がなく、test.csvと提出見本に同じIDがある。
- 出力の列名・列順・画像ID・行数・行順が提出見本に対応する。
- 確率がすべて有限で、0以上1以下である。
- 画像ごとの4クラスの確率の合計が約1になる。

CSVにはDataFrameのインデックスを追加しません。
出力の形は1,821行・5列です。
実行したモデルとファイルのハッシュなどを、同じ場所の`submission.json`に記録します。

既存の提出CSVは上書きしません。
別名で再生成する場合は、次のように指定します。

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py --run-dir outputs/baseline_01 --output outputs/baseline_01/submission_02.csv
```

Kaggleへのアップロードは、CSV生成後に行います。
このスクリプトはファイルを作るところまでで、外部への提出は行いません。

## 保存と配布

学習済み重み・予測CSV・split.csvはGitの管理対象外です。
生成後、コンテナ内のoutputs/baseline_01をホストへコピーして保存し、提出CSVを手元のPCへ取得します。
後輩と検証条件を揃えるために、学習時に保存した分割CSVも共通のファイルとして配布します。
移動先を確認してから、ホスト側でコピーする手順を実行します。

## 参考

- [PyTorch 2.10: torch.load](https://docs.pytorch.org/docs/2.10/generated/torch.load.html)
- [PyTorch: モデルの保存と読み込み](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [コンペの提出形式](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7)
