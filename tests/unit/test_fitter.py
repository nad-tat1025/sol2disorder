import pytest
import pandas as pd
import numpy as np

# テスト対象のクラスをインポート
from ..src.core.fitting import Fitter 

def test_fitter_with_perfect_exponential_data():
    """
    Fitterが理想的な指数関数的減衰データを正しくフィッティングできるかテストする。
    """
    # 1. Arrange (準備): テストの準備をします
    # ----------------------------------------------------
    # テスト用のFitterインスタンスを作成
    fitter = Fitter(fitting_config={'function': 'exp_decay'})

    # テスト用の人工データを作成します。
    # このデータは、解析的に答えが分かっている y = 2.5 * exp(-1.5 * x) という
    # 完全な指数関数に従います。
    true_a, true_b = 2.5, 1.5
    distances = np.linspace(0, 5, 50)
    values = true_a * np.exp(-true_b * distances)
    
    # Fitterが受け取る形式 (DataFrame) に変換
    test_df = pd.DataFrame({'distance': distances, 'value': values})

    # 2. Act (実行): テストしたい機能を呼び出します
    # ----------------------------------------------------
    fitted_params = fitter.fit(test_df)

    # 3. Assert (検証): 結果が期待通りか確認します
    # ----------------------------------------------------
    # 期待されるパラメータは、人工データ作成時に使った [2.5, 1.5]
    expected_params = np.array([true_a, true_b])

    # 浮動小数点数の比較なので、非常に近い値であればOKとする
    np.testing.assert_allclose(fitted_params, expected_params, rtol=1e-6)