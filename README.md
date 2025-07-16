# sol2disorder

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![testing: pytest](https://img.shields.io/badge/testing-pytest-blueviolet.svg)](https://pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

第一原理計算から得られるタイトバインディングモデルを基に、結晶中の乱れを考慮したハミルトニアンを構築・解析するPythonツールキット。

---

## 機能

* **Wannier90モデルの解析**: `wannier90_hr.dat`と`wannier90.win`を読み込み、ホッピング、サイト、軌道の情報を抽出します。
* **ホッピングの距離依存性の解析**: 原子ペア（原子種・軌道）ごとにホッピングパラメータを距離の関数として評価し、フィッティングします。
* **乱れた系のハミルトニアン構築**: フィッティングした関数に基づき、原子変位や置換を含む系のハミルトニアンを再構築します。
* **YAMLによる設定管理**: 解析条件はすべて設定ファイルで管理し、コードの再利用性と可読性を高めます。

---

## インストール

1.  **リポジトリのクローン**
    ```bash
    git clone [https://github.com/nad-tat1025/sol2disorder.git](https://github.com/nad-tat1025/sol2disorder.git)
    cd sol2disorder
    ```

2.  **依存ライブラリのインストール**
    ```bash
    pip install -r requirements.txt
    ```

---

## 使用方法

### 1. 設定ファイルの準備

解析条件をYAMLファイルに記述します。

**`configs/config.yaml` の例:**

```yaml
model_path: ./data/Bi/wannier90_hr.dat
wannier_win_path: ./data/Bi/wannier90.win
position_units: fractional
spin: False
cutoff_radius: 6.0
fitting:
  function: exp_decay
  params: [1.0, 1.0]
```

### 2. 解析の実行

作成した設定ファイルを指定してメインスクリプトを実行します。

```bash
python main.py --config_path ./configs/config.yaml
```

結果は`outputs/`ディレクトリに保存されます。

---

## プロジェクト構成

```
.
├── configs/          # 解析設定ファイル
├── data/             # 入力データ (wannier90_hr.datなど)
├── outputs/          # 解析結果の出力先
├── src/              # ソースコード
│   ├── core/         # 中核モジュール (analyzer, fitting, mapper)
│   └── utils/        # 補助モジュール (I/O, プロット)
├── tests/            # テストコード
├── main.py           # メインスクリプト
└── requirements.txt  # 依存ライブラリ
```

---

## テスト

`pytest`による単体・結合テストを整備しています。以下のコマンドで実行できます。

```bash
pytest
```

---

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。