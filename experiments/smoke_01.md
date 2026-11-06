# 1エポックの実行確認

## 結果

2026年11月6日、利用者が計算機のコンテナで実行し、共有した`history.csv`の記録です。
案内したコマンドは次のとおりです。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 1 --output-dir outputs/smoke_01
```

学習用1,456枚、検証用365枚の固定分割を使います。
実行時の`config.json`と学習済み重みは、手元のこのリポジトリには取得していません。

```csv
epoch,train_loss,valid_loss,valid_auc,auc_healthy,auc_multiple_diseases,auc_rust,auc_scab,seconds
1,0.49514403055002404,0.21808659891559654,0.9851981281889064,0.9958867560957533,0.9543707973102785,0.9945999999999999,0.9959349593495935,43.1341671498958
```

4クラスのAUCの単純平均は、記録された`valid_auc`と一致します。
クラス別では`multiple_diseases`のAUCが最も低い結果でした。

学習損失は、学習中の各バッチで計算した損失の平均です。
検証損失は、1エポックの更新を終えたモデルで測ります。
このため、今回のように検証損失が学習損失より小さくなることがあります。

この数値は固定した検証セット上のスコアです。Kaggleのテストスコアはまだ測っていません。

## 次の実行

同じ条件で5エポック学習し、最良の検証AUCを得たモデルを保存します。

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 5 --output-dir outputs/baseline_01
```

1エポックの確認とは別に、ImageNetの重みから新しく学習を開始します。
結果を確認して、後輩に示す検証スコアと、テストデータの予測に使うモデルを決めます。
