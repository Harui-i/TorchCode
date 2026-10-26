# PyTorch の学習環境

`.py` をエディタで編集し、ターミナルから実行・採点します。Python 3.12・CPU 版 PyTorch・NumPy は uv で管理します。

## 使い方

このディレクトリで実行してください。

```bash
uv sync --locked
./torchcode list
```

`exercises/01_relu.py` を開いて実装します。ファイルの先頭に問題文とルールがコメントで入っています。

```bash
# 自分で書いたデバッグ用コードを実行
uv run python exercises/01_relu.py

# テストで採点（全問合格なら終了コード 0、不合格なら 1）
./torchcode check relu
# ファイル名の番号付きでも指定可能
./torchcode check 01_relu

./torchcode hint relu
./torchcode status
```

`./torchcode` は `uv run python scripts/judge.py` のラッパーです。直接実行しても同じです。

```bash
uv run python scripts/judge.py check relu
./torchcode check relu --file solutions/01_relu_solution.py --no-progress
./torchcode check relu --timeout 120
```

- `exercises/`：41問の解答用 `.py`。問題文・実装欄・直接実行時だけ動くデバッグコード。
- `solutions/`：参考解答の `.py`。
- `judge/tasks.json`：問題情報と174件のテスト。元の Python モジュールを import せず、データとして読み込みます。
- `scripts/judge.py`：ローカル採点 CLI。解答の読み込み時には `if __name__ == "__main__":` 以下を実行しません。
- `data/progress.json`：合格状況と試行回数。過去の合格は、その後不合格になっても保持します。
- `pyproject.toml` / `uv.lock`：依存関係と解決済みバージョン。Jupyter は含みません。

採点は子プロセスで実行し、既定で60秒を超えると中断します（終了コード124）。これは隔離環境ではありません。解答とテストはホストのユーザー権限で実行されます。グローバルの Python にライブラリはインストールしません。

## 旧環境からの移行

Docker 内の保存済みノートから解答を `.py` に変換し、進捗も引き継いでいます。

- `backups/docker-migration/`：旧 Docker ボリュームの退避。元ノート・進捗・Jupyter 設定を保存。
- `backups/upstream-source.tar.gz`：変更前の取得元コード一式。

これらのバックアップだけは復元用に元の `.ipynb` を含みます。通常の学習では使いません。旧コンテナ・プロジェクト専用イメージ・ボリューム・ネットワークは削除済みです。

問題文・参考解答・テストの取得元は [duoan/TorchCode](https://github.com/duoan/TorchCode)、取得コミットは `e3c8782a4be4f7cdb4de871ec739f66370968a77`（取得元のライセンス表記: MIT）です。上流のインストールスクリプトや Web/Jupyter アプリは使用しません。
