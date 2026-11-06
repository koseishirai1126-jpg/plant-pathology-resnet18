# 03: DataLoaderで画像をまとめて取り出す

## Datasetとの関係

`Dataset`は、指定した位置の画像1枚と正解ラベルを返します。
`DataLoader`は、そのDatasetから複数枚を取り出してまとめます。
この1回にまとめる単位を「バッチ」と呼びます。

今回のコードは`check_dataloader.py`です。
データの読み込みには前の段階で作った`PlantDataset`をそのまま使います。

```python
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    drop_last=False,
)

images, labels = next(iter(loader))
```

`iter(loader)`で順番にバッチを取り出すためのイテレータを作り、
`next(...)`で先頭の1バッチを取り出します。
学習ループでは、今後`for images, labels in loader:`の形で順に取り出します。

## 今回指定した設定

| 設定 | 今回の値 | 意味 |
| --- | --- | --- |
| `batch_size` | `4` | 1バッチに画像を最大4枚まとめる |
| `shuffle` | `False` | 今回はCSVに書かれた順番で取り出す |
| `num_workers` | `0` | 今回は読み込み用の別プロセスを作らず、同じプロセスで読み込む |
| `drop_last` | `False` | 最後の4枚未満のバッチも残す |

ここでのバッチサイズ4は、形を確認するための設定です。
実際の学習に使うバッチサイズは、学習コードを作る段階で決めます。
学習用データは`shuffle=True`、検証用・テスト用データは`False`を使う方針です。

## まとめる前と後の形

| 項目 | Datasetから1枚取り出す | DataLoaderで4枚まとめる |
| --- | --- | --- |
| 画像 | Tensor、`[3, 224, 224]` | Tensor、`[4, 3, 224, 224]` |
| 正解ラベル | Pythonの整数。例：`3` | 整数のTensor、`[4]` |

画像は「バッチの枚数・チャンネル数・高さ・幅」の順になります。
ラベルも自動的にTensorにまとまり、型は`torch.int64`です。
これはクラス番号を渡して`CrossEntropyLoss`を使うときの型です。

全1,821枚を4枚ずつに分けるので、バッチ数は456です。
最後のバッチは1枚になります。モデルに渡すコードでは、バッチの枚数を4に決め打ちせず扱います。

## 実行方法と想定出力

プロジェクトのフォルダで実行します。

```bash
python check_dataloader.py
```

```text
画像の枚数: 1821
バッチの数: 456
画像の形: (4, 3, 224, 224)
画像の型: torch.float32
ラベルの形: (4,)
ラベルの型: torch.int64
正解ラベル: [3, 1, 0, 2]
対応するクラス: ['scab', 'multiple_diseases', 'healthy', 'rust']
DataLoaderの動作確認OK
```

ここでは先頭の1バッチだけを実際に読み込みます。
`len(loader)`はバッチの数を計算する操作で、全画像の読み込み確認ではありません。
このコードはCPU上でデータを準備します。モデルに入力する際のGPUへの転送は、後の段階で扱います。

## 次の段階

ここでのDatasetには、形を確認するために`train.csv`の全行を渡しています。
実際の学習前には学習用・検証用に分割し、それぞれにDatasetとDataLoaderを作ります。

## 参考

- [PyTorch 2.10公式: torch.utils.data](https://docs.pytorch.org/docs/2.10/data.html)
