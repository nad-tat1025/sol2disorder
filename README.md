# sol2disorder

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![testing: pytest](https://img.shields.io/badge/testing-pytest-blueviolet.svg)](https://pytest.org)

第一原理計算から得られるタイトバインディングモデルを基に、結晶中の乱れを考慮したハミルトニアンを構築・解析するPythonツールキット。

---

## 機能

* **Wannier90モデルの解析**: `wannier90_hr.dat`と`wannier90.win`を読み込み、ホッピング、サイト、軌道の情報を抽出します。
* **ホッピングの距離依存性の解析**: 原子ペア（原子種・軌道）ごとにホッピングパラメータを距離の関数として評価し、フィッティングします。
* **乱れた系のハミルトニアン構築**: フィッティングした関数に基づき、アモルファスや液相のハミルトニアンを再構築します。
* **YAMLによる設定管理**: 解析条件はすべて設定ファイルで管理し、コードの再利用性と可読性を高めます。

---

## インストール

1.  **リポジトリのクローン**
    ```bash
    git clone https://github.com/nad-tat1025/sol2disorder.git
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
# --- ソースモデル情報 ---
source_structure_file: "./data/input/Bi/PBE/spinless/POSCAR"
source_hr_file: "./data/input/Bi/PBE/spinless/wannier90_hr.dat"
source_win_file: "./data/input/Bi/PBE/spinless/wannier90.win" 

# --- ターゲット構造情報 ---
target_structure_file: null
target_projection: null
output_hr_file: "./data/output/Bi/PBE/spinless/mapped_hr.dat"

# --- 一般設定 ---
spin: false
fermi_energy: 2.9478

# --- 解析オプション ---
distance_threshold: 1.0e-9
zero_hop_threshold: 1.0e-9
hopping_cutoff_distance: 6.5
use_hermitian_symmetrization: true
use_symmetry_averaging: false
fit_type: "linear" # "exp", "gaussian", "linear"

# --- 方向依存性オプション ---
# use_auto_direction_clusters: true
# max_direction_clusters: 8

# --- 検証オプション ---
run_plotting: true
run_validation: true
compare_parameters: true

# --- デバッグ用オプション
use_ungrouped_interpolation: true
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
├── configs/
├── data/
│   ├── input/
│   └── output/
├── src/              
│   ├── core/         
│   └── utils/        
├── tests/            
├── main.py           
└── requirements.txt  
```

---

## テスト

`pytest`による単体・結合テストを整備しています。以下のコマンドで実行できます。

```bash
pytest
```

