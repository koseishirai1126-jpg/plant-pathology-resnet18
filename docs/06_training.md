# 06: 学習・検証・最良モデルの保存

## 学習の設定

`train.py`は、ここまでに作成したDataset・DataLoader・モデルを使います。

| 項目 | 最初のベースラインで使う値 |
| --- | --- |
| モデル | ImageNetの重みを初期値とするResNet18。最後の分類層は4クラス用に新規作成 |
| 更新する範囲 | 最後の分類層を含む、モデル全体 |
| 入力画像 | 224×224、RGB |
| 前処理 | 短辺256にリサイズ、中央224×224を切り抜き、Tensor化、ImageNetの平均・標準偏差で正規化 |
| ランダムな画像変換 | 今回は使わない |
| 損失関数 | CrossEntropyLoss |
| パラメータの更新方法 | Adam |
| 学習率 | 0.0001（`1e-4`） |
| バッチサイズ | 32 |
| エポック数 | 動作確認は1、本番は5 |
| 乱数の設定 | 42 |
| 保存するモデル | 検証用の平均ROC AUCが最も高かったエポックのモデル |

1エポックは、学習用の1,456枚を1周する単位です。
バッチサイズ32なら1エポックのパラメータ更新は46回で、最後のバッチは16枚です。
検証用365枚は、エポックが終わるごとにすべて予測します。

## 学習ループの中心

```python
model.train()
for images, labels in loader:
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
```

1. `model.train()`で、BatchNormなどを学習時の動作に切り替える。
2. `zero_grad()`で前のバッチの勾配を消す。
3. `model(images)`で予測値を計算する。
4. `criterion(...)`で正解とのずれを損失として求める。
5. `backward()`で各パラメータに対する勾配を計算する。
6. `step()`で、その勾配を使ってパラメータを更新する。

`loss.backward()`だけではパラメータは更新されません。
今回、新しく加わる更新操作が`optimizer.step()`です。
`model.parameters()`をAdamに渡しているため、最後の層だけでなくモデル全体が更新対象です。

損失の表示には、各バッチの平均損失に画像数を掛けた値を集計し、全画像数で割った値を使います。
バッチごとの平均を単純に平均すると、最後の少ない枚数のバッチを過大に重く扱うためです。

## 検証ループ

検証では`model.eval()`と`torch.no_grad()`を使い、パラメータ更新を行いません。
BatchNormの統計値も検証用画像では更新しません。
すべての検証画像について、softmaxで求めた4クラスの確率を集めます。

クラスごとに「そのクラスか、それ以外か」という0・1の正解を作り、ROC AUCを計算します。
たとえば`scab`のAUCなら、正解が`scab`の画像を1、それ以外を0として、`scab`の予測確率で評価します。
この4クラス分を単純平均した値を`valid_auc`として記録します。
これはコンペのmean column-wise ROC AUCに対応する計算です。

ROC AUCは、閾値を動かしたときに陽性と陰性をどの程度区別できるかを表す指標です。
正解率とは異なり、予測したクラス番号ではなく各クラスの予測確率を使います。
バッチごとのAUCの平均ではなく、検証用365枚をまとめて計算します。

学習では損失を減らすようにパラメータを更新し、モデルの選択には検証AUCを使います。
最後のエポックが最良とは限らないので、改善した時点の重みを`best.pt`に保存します。

## 最初は1エポック実行する

コンテナのプロジェクトフォルダで実行します。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 1 --output-dir outputs/smoke_01
```

ここでは実際に全学習画像でパラメータを更新し、全検証画像を評価します。
これまでの4枚の確認と異なり、全画像の読み込みを含む動作確認です。
10バッチごとと学習用の最後のバッチで進捗を表示します。

この実行結果を確認してから、本番のベースラインを5エポック学習します。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 5 --output-dir outputs/baseline_01
```

両方の実行は、それぞれImageNetの重みから新しく開始します。
本番の5エポック実行は、動作確認の1エポック実行の続きではありません。
使う画像の割り当ては、どちらも`data/split.csv`から読み込みます。

既存の結果を混ぜないため、保存先には未作成のフォルダを指定します。
再実行する場合は、`outputs/smoke_02`など新しい名前を使います。

## 保存するもの

```text
outputs/smoke_01/
├── config.json             # 学習条件・ライブラリの版・Gitコミット・データのハッシュ
├── split.csv               # この実行で使った分割のコピー
├── history.csv             # エポックごとの損失・平均AUC・各クラスのAUC・処理時間
├── best.pt                 # 最良モデルの重みと、クラス順・エポック・平均AUC
├── best_metrics.json       # 最良エポックの評価結果
└── valid_predictions.csv   # 最良モデルでの検証用365枚の予測確率
```

`valid_predictions.csv`には検証用画像のIDが入っています。Kaggleへの提出用ファイルは後の段階で作ります。
`best.pt`にはモデルの`state_dict`を保存します。
学習の再開に必要なoptimizerの状態は含めていません。

`outputs/`はGitの管理対象外です。
このコンテナにはホストと共有するフォルダがないため、学習後にこのフォルダをホスト側へコピーして保存します。

## 再現性と比較

Python・NumPy・PyTorchの乱数の設定を固定し、cuDNNの設定も揃えます。
ただし、異なるGPUやライブラリの版まで含めて完全に同じ数値になることは保証されません。
同じ分割の画像ID、実行条件、コードの版を残し、スコアを比較できるようにします。
保存したGitコミットと異なる変更がある場合は、`config.json`の`code.dirty`が`true`になります。

今回測る値は検証スコアです。Kaggleのスコアと区別して記録します。

## 参考

- [PyTorch 2.10: Adam](https://docs.pytorch.org/docs/2.10/generated/torch.optim.Adam.html)
- [scikit-learn: roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [PyTorch: Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [PyTorch 2.10: Reproducibility](https://docs.pytorch.org/docs/2.10/notes/randomness.html)
- [コンペの評価方法](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7)
