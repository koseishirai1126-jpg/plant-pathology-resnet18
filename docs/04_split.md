# 04: 学習用と検証用にデータを分ける

## 何を分けるか

正解ラベルのある`train.csv`の1,821枚を、学習用80%と検証用20%に分けます。
画像の枚数は整数にする必要があるため、学習用1,456枚、検証用365枚になる想定です。

学習用はモデルのパラメータの更新に使います。
検証用は更新に使わず、学習中の性能確認と、保存するモデルの選択に使います。
Kaggleの`test.csv`に書かれた画像は、この分割に使いません。

## 分割するコード

`prepare_split.py`の中心は次の部分です。

```python
train_df, valid_df = train_test_split(
    all_df,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)
```

| 指定 | 意味 |
| --- | --- |
| `all_df` | 配布された`train.csv`の全行 |
| `test_size=0.2` | この関数で分ける側を全体の20%にする。今回はその戻り値を検証用に使う |
| `random_state=42` | 分割に使う乱数の設定を固定する。42自体に特別な意味はない |
| `stratify=labels` | クラスの比率をなるべく保って分ける |

関数の引数名は`test_size`ですが、ここで作るのは検証用データです。
Kaggleのテスト用データとは別のものです。

分割前に画像IDで並び順を揃え、正解ラベルがone-hotであることと、IDに重複・欠損がないことを確認します。
分割後には、学習用と検証用のIDが重ならず、すべての画像が含まれているか確認します。

## 保存するファイル

`data/split.csv`に、すべての画像IDと割り当てを保存します。
列は`image_id`と`split`で、`split`の値は`train`または`valid`です。
画像ファイルの移動や複製は行いません。

```text
data/
├── train.csv
├── test.csv
├── sample_submission.csv
├── split.csv                 ← 今回作る割り当て表
└── images/
```

`data/`はGitの管理対象外なので、このCSVも自動的にはGitHubに送られません。
後輩と同じ条件で比較する際は、この分割CSVを共通のファイルとして配布し、同じ画像IDを使います。
今後の学習コードは、このファイルから割り当てを読み込む方針です。

同じ入力と同じライブラリ環境なら、固定した乱数の設定で同じ分割を生成できます。
比較する際の基準には、実際に保存した画像IDを使います。
スクリプトを再実行した際、既存のCSVと異なる分割になる場合は上書きせず停止します。

## 実行方法

分割にはscikit-learnを使います。後の段階でROC AUCの計算にも使うライブラリです。
コンテナ内で、必要なライブラリとスクリプトを実行します。

```bash
python -m pip install scikit-learn
python prepare_split.py
```

実行結果には、ライブラリのバージョン、画像数、クラスごとの枚数、重複の確認結果が表示されます。
想定されるクラスごとの枚数は次のとおりです。

| クラス | 全体 | 学習用 | 検証用 |
| --- | ---: | ---: | ---: |
| healthy | 516 | 413 | 103 |
| multiple_diseases | 91 | 73 | 18 |
| rust | 622 | 497 | 125 |
| scab | 592 | 473 | 119 |

検証用の画像を見てモデルを選ぶため、その検証スコアは完全に独立な最終評価ではありません。
Kaggleのスコアとは区別して記録します。

## 次の段階

この割り当てに従って学習用・検証用それぞれのDatasetとDataLoaderを作ります。
ランダムな画像変換は学習用だけに適用し、検証時の前処理は固定します。

## 参考

- [scikit-learn公式: train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [scikit-learn公式: データ漏洩を避ける](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
