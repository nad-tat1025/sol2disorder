# sol2disorder

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![testing: pytest](https://img.shields.io/badge/testing-pytest-blueviolet.svg)](https://pytest.org)

第一原理計算から得られるタイトバインディングモデルを基に、アモルファスや液相のハミルトニアンを構築・解析するPythonツールキット。

## 機能

* **Wannier90モデルの解析**: `POSCAR`, `wannier90_hr.dat`, `wannier90.win`を読み込み、ホッピング、サイト、軌道の情報を抽出します。
* **ホッピングの距離依存性の解析**: 原子ペア（原子種・軌道・スピン）ごとにホッピングパラメータを距離の関数として評価し、フィッティングします。
* **乱れた系のハミルトニアン構築**: フィッティングした関数に基づき、アモルファスや液相のハミルトニアンを再構築します。

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
    
## 使用方法

### 1. 設定ファイルの準備

解析条件をYAMLファイルに記述します。

**`configs/config.yaml` の例:**

```yaml
# --- ソースモデル情報 ---
source_structure_file: "./data/input/POSCAR_source"
source_hr_file: "./data/input/wannier90_hr.dat"
source_win_file: "./data/input/wannier90.win" 

# --- ターゲットモデル情報 ---
target_structure_file: "./data/output/POSCAR_target"
# target_structure_file: null   (target_structure = source_structure の場合)
output_hr_file: "./data/output/mapped_hr.dat"

# --- 一般設定 ---
spin: false
fermi_energy: 2.9478

# --- 解析オプション ---
distance_threshold: 1.0e-9
zero_hop_threshold: 1.0e-9
hopping_cutoff_distance: 7.0
use_hermitian_symmetrization: true
use_symmetry_averaging: true
fit_type: "linear"    # Options"exp", "gaussian", "linear"

# --- 検証オプション ---
run_plotting: true
run_validation: true
compare_parameters: true

```

### 2. 解析の実行

作成した設定ファイルを指定してメインスクリプトを実行します。

```bash
python main.py --config_path ./configs/config.yaml
```

結果は`outputs/`ディレクトリに保存されます。

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
