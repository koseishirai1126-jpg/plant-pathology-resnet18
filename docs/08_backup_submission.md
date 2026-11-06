# 08: 実行結果を保存し、Kaggleに提出する

## 保存する範囲

コンテナ内の`/root/plant-pathology-resnet18/outputs/baseline_01/`をフォルダごと保存します。
重み、実行条件、分割、検証予測、提出CSVとそれぞれの確認記録を同じ場所に残すためです。

現在のコンテナにはホストとの共有フォルダがありません。
まずホスト側にコピーし、続いて手元のPCにも取得します。

## ホスト側にコピーする

SSH接続先のホスト側で実行します。`CONTAINER_ID`は使用したコンテナのIDに置き換えます。

```bash
mkdir -p ~/kaggle/results
sudo docker cp CONTAINER_ID:/root/plant-pathology-resnet18/outputs/baseline_01 ~/kaggle/results/
sudo chown -R "$(id -u):$(id -g)" ~/kaggle/results/baseline_01
```

`docker cp`はコンテナからホストへのコピーにも使えます。
`sudo`でコピーしたファイルをログイン中のユーザーから扱えるように、コピー先の所有者を揃えます。
`chown`の対象は、この実行結果フォルダだけです。

## 手元のPCに取得する

手元のプロジェクトに`outputs/`を作成し、scpでホストの実行結果フォルダを取得します。
`scp -r`はフォルダを中身ごとコピーする指定です。
接続先と保存先は各自の環境に合わせます。

取得後のプロジェクト内の構成は、次のようになります。

```text
outputs/baseline_01/
├── best.pt
├── best_metrics.json
├── config.json
├── history.csv
├── split.csv
├── valid_predictions.csv
├── verification.json
├── submission.csv
└── submission.json
```

`submission.json`のハッシュと取得したCSVのハッシュを比較すると、転送前後でファイルが同じか確認できます。
`verification.json`と`submission.json`に記録した重みのハッシュも一致する想定です。
`outputs/`はGitの管理対象外です。

## Kaggleへの提出

[コンペのページ](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7)を開き、
ログインした状態でLate Submissionなどの提出入口を確認します。
手元に取得した`submission.csv`を指定し、評価結果を確認します。
実際に操作可能かどうかは、ログイン後の画面で確認します。

提出したモデルを区別できる説明の例：

```text
ResNet18 ImageNet, 224px, Adam lr=1e-4, batch=32, best epoch=2/5, seed=42, local val AUC=0.989535
```

採点されたら、Kaggleに表示されたスコアと、Public / Privateなどの表示区分、提出日時、説明文を記録します。
ローカル検証AUCとは別の欄に残し、後輩との比較に使う指標を揃えます。

## 後輩と共有するもの

- コードと解説のGitHubリポジトリ。
- 学習時に保存した共通の分割CSV。各自の`data/split.csv`として使う。
- 実行条件、検証スコア、Kaggleの評価結果。

各自がKaggleから取得した元データを使い、検証画像の割り当てを共通にします。
新たに分割し直したスコアと、今回の検証スコアは分けて扱います。

## 参考

- [Docker: docker cp](https://docs.docker.com/reference/cli/docker/container/cp/)
- [OpenSSH: scp](https://man.openbsd.org/scp)
