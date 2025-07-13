# # src/workflow.py
# import logging
# import tbmodels as tb
# import numpy as np

# # 各モジュールをインポート
# from src.utils.io import _read_geometry, parse_wannier_win
# from src.utils.plotting import (
#     plot_hopping_with_fit,
#     validate_band_structure,
#     print_hopping_summary,
#     compare_hopping_parameters,
#     plot_hopping_comparison
# )
# from src.core.analyzer import HoppingAnalyzer
# from src.core.fitting import FittingManager
# from src.core.mapper import HamiltonianMapper

# def run_workflow(config: dict):
#     """
#     タイトバインディングモデルの解析とマッピングを行うメインワークフロー。
#     この関数は設定(config)辞書を受け取って処理を実行する。
#     """
#     # --- Step 0a: wannier90.winから設定を読み込み ---
#     win_file = config.get("source_win_file")
#     if win_file:
#         parsed_proj, parsed_kpath = parse_wannier_win(win_file)
#         if parsed_proj:
#             config["source_projection"] = parsed_proj
#             logging.info("Source projection was loaded from .win file.")
#         if parsed_kpath:
#             config["k_path"] = parsed_kpath
#             logging.info("K-path was loaded from .win file.")

#     # --- Step 0b: 構造情報の読み込み ---
#     logging.info("--- Reading Source Geometry ---")
#     source_geometry = _read_geometry(config["source_structure_file"])
#     if source_geometry is None: return

#     target_geometry = None
#     target_file_path = config.get("target_structure_file")
#     if target_file_path:
#         logging.info("--- Reading Target Geometry ---")
#         target_geometry = _read_geometry(target_file_path)
#         if target_geometry is None: return
#     else:
#         logging.info("--- Target structure file not specified. Using source structure for mapping. ---")
#         target_geometry = source_geometry

#     # --- Step 1: ソースモデルのロード ---
#     logging.info("--- Loading source model from Wannier90 files... ---")
#     try:
#         source_model = tb.Model.from_wannier_files(
#             hr_file=config["source_hr_file"],
#             uc=source_geometry['uc'],
#             contains_cc=True,
#         )
#         logging.info("Source TB-model loaded successfully.")
#     except Exception as e:
#         logging.error(f"Failed to load model from Wannier90 files. Details: {e}")
#         return

#     fermi_energy = config.get("fermi_energy")
#     if fermi_energy is not None:
#         logging.info(f"Shifting on-site energies by Fermi energy: {-fermi_energy:.4f} eV")
#         onsite_key = (0, 0, 0)
#         if onsite_key in source_model.hop:
#             h_onsite = source_model.hop[onsite_key]
#             h_onsite -= fermi_energy / 2 * np.eye(h_onsite.shape[0], dtype=complex) # NOTE: TBmodelsの実装上, 半分にする (原因不明)
#             logging.info("On-site energies have been shifted successfully.")
#         else:
#             logging.warning("On-site Hamiltonian H(R=0) not found. Cannot shift by Fermi energy.")

#     logging.info("Step 1: Analyzing hoppings from the source model...")
#     analyzer_source = HoppingAnalyzer(
#         composition=source_geometry["composition"],
#         projection=config["source_projection"],
#         positions=source_geometry["positions"],
#         uc=source_geometry["uc"],
#         config=config,
#         model=source_model
#     )
#     hopping_dict = analyzer_source.extract_and_rotate_hoppings()
#     grouped_data = analyzer_source.group_hoppings_by_type(hopping_dict)
#     if config.get("use_symmetry_averaging", False):
#         grouped_data = analyzer_source.average_symmetric_hoppings(grouped_data)
#     print_hopping_summary(grouped_data)

#     # --- Step 2: 補間関数の作成 ---
#     logging.info(f"Step 2: Fitting data with '{config['fit_type']}' model...")
#     interpolated_functions = FittingManager.create_interpolated_functions(grouped_data, config)
#     if config.get("run_plotting", True):
#         plot_hopping_with_fit(grouped_data, interpolated_functions)

#     # --- Step 3: ターゲットハミルトニアンの再構築 ---
#     logging.info("Step 3: Reconstructing Hamiltonian for the target structure...")
#     target_projection = config.get("target_projection") or config["source_projection"]
    
#     analyzer_target = HoppingAnalyzer(
#         composition=target_geometry["composition"],
#         projection=target_projection,
#         positions=target_geometry["positions"],
#         uc=target_geometry["uc"],
#         config=config
#     )
#     mapper = HamiltonianMapper(interpolated_functions, analyzer_target, config)
#     bravais_vectors_list = list(source_model.hop.keys())
#     H_mapped = mapper.construct_hermitian_hops(bravais_vectors_list)
    
#     # --- Step 4: 新しいモデルの作成と出力、検証 ---
#     try:
#         mapped_model = tb.Model(
#             uc=target_geometry["uc"],
#             hop=H_mapped,
#             contains_cc=False,
#         )
#         logging.info("Mapped TB-model constructed successfully.")
        
#         output_file = config["output_hr_file"]
#         mapped_model.to_hr_file(output_file)
#         logging.info(f"Reconstructed Hamiltonian saved to '{output_file}'")

#         # マッピング前後のデータの比較
#         if config.get("compare_parameters", True):
#             logging.info("Graphically comparing original and mapped hopping parameters...")
#             # マッピング後のモデルからパラメータを抽出
#             analyzer_mapped = HoppingAnalyzer(
#                 model=mapped_model,
#                 composition=source_geometry["composition"],
#                 projection=config["source_projection"],
#                 positions=source_geometry["positions"],
#                 config=config
#             )
#             mapped_hopping_dict = analyzer_mapped.extract_and_rotate_hoppings()
#             mapped_grouped_data = analyzer_mapped.group_hoppings_by_type(mapped_hopping_dict)
#             if config.get("use_symmetry_averaging", False):
#                 mapped_grouped_data = analyzer_mapped.average_symmetric_hoppings(mapped_grouped_data)

#             plot_hopping_comparison(grouped_data, mapped_grouped_data, interpolated_functions, config)

#         if config.get("run_validation", False):
#             validate_band_structure(config, source_model, mapped_model)
        
#     except ValueError as e:
#         logging.error(f"Failed to construct the mapped model: {e}")

import logging
import tbmodels as tb
import numpy as np
from collections import defaultdict # 追加

# 各モジュールをインポート
from src.utils.io import _read_geometry, parse_wannier_win
from src.utils.plotting import (
    plot_hopping_with_fit,
    validate_band_structure,
    print_hopping_summary,
    compare_hopping_parameters,
    plot_hopping_comparison,
    plot_ungrouped_hopping_with_fit,
)
from src.core.analyzer import HoppingAnalyzer
from src.core.fitting import FittingManager, LinearInterpolationFunction, InterpolationFunction # 追加
from src.core.mapper import HamiltonianMapper

def run_workflow(config: dict):
    """
    タイトバインディングモデルの解析とマッピングを行うメインワークフロー。
    この関数は設定(config)辞書を受け取って処理を実行する。
    """
    # (Step 0a, 0b, 1前半のロード処理は変更なしです)
    # --- Step 0a: wannier90.winから設定を読み込み ---
    win_file = config.get("source_win_file")
    if win_file:
        parsed_proj, parsed_kpath = parse_wannier_win(win_file)
        if parsed_proj:
            config["source_projection"] = parsed_proj
            logging.info("Source projection was loaded from .win file.")
        if parsed_kpath:
            config["k_path"] = parsed_kpath
            logging.info("K-path was loaded from .win file.")

    # --- Step 0b: 構造情報の読み込み ---
    logging.info("--- Reading Source Geometry ---")
    source_geometry = _read_geometry(config["source_structure_file"])
    if source_geometry is None: return

    target_geometry = None
    target_file_path = config.get("target_structure_file")
    if target_file_path:
        logging.info("--- Reading Target Geometry ---")
        target_geometry = _read_geometry(target_file_path)
        if target_geometry is None: return
    else:
        logging.info("--- Target structure file not specified. Using source structure for mapping. ---")
        target_geometry = source_geometry

    # --- Step 1: ソースモデルのロード ---
    logging.info("--- Loading source model from Wannier90 files... ---")
    try:
        source_model = tb.Model.from_wannier_files(
            hr_file=config["source_hr_file"],
            uc=source_geometry['uc'],
            contains_cc=True,
        )
        logging.info("Source TB-model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model from Wannier90 files. Details: {e}")
        return

    fermi_energy = config.get("fermi_energy")
    if fermi_energy is not None:
        logging.info(f"Shifting on-site energies by Fermi energy: {-fermi_energy:.4f} eV")
        onsite_key = (0, 0, 0)
        if onsite_key in source_model.hop:
            h_onsite = source_model.hop[onsite_key]
            h_onsite -= fermi_energy / 2 * np.eye(h_onsite.shape[0], dtype=complex)
            logging.info("On-site energies have been shifted successfully.")
        else:
            logging.warning("On-site Hamiltonian H(R=0) not found. Cannot shift by Fermi energy.")

    # --- Step 1後半: ホッピングデータの抽出 ---
    logging.info("Step 1: Analyzing hoppings from the source model...")
    analyzer_source = HoppingAnalyzer(
        composition=source_geometry["composition"],
        projection=config["source_projection"],
        positions=source_geometry["positions"],
        uc=source_geometry["uc"],
        config=config,
        model=source_model
    )
    hopping_dict = analyzer_source.extract_and_rotate_hoppings()


    # ▼▼▼ ここからが修正箇所です ▼▼▼
    logging.info(f"Step 2: Fitting data with '{config['fit_type']}' model...")

    use_ungrouped_interpolation = config.get("use_ungrouped_interpolation", False)
    if use_ungrouped_interpolation:
        # [テスト用] グループ化・平均化を行わず、生のデータで補間する
        logging.warning("TEST MODE: Bypassing grouping/averaging. Interpolating from raw hopping data.")
        
        # ungroupedデータ用の補間関数生成ロジック
        interpolated_functions = FittingManager.create_interpolated_functions_from_raw(hopping_dict, config)

        if config.get("run_plotting", True):
            plot_ungrouped_hopping_with_fit(hopping_dict, interpolated_functions, config)

    else:
        # [通常処理] グループ化、平均化を行ってから補間する
        grouped_data = analyzer_source.group_hoppings_by_type(hopping_dict)
        if config.get("use_symmetry_averaging", False):
            grouped_data = analyzer_source.average_symmetric_hoppings(grouped_data)
        print_hopping_summary(grouped_data)
        
        interpolated_functions = FittingManager.create_interpolated_functions(grouped_data, config)
        
        if config.get("run_plotting", True):
            plot_hopping_with_fit(grouped_data, interpolated_functions)

    # --- Step 3: ターゲットハミルトニアンの再構築 ---
    logging.info("Step 3: Reconstructing Hamiltonian for the target structure...")
    target_projection = config.get("target_projection") or config["source_projection"]
    
    analyzer_target = HoppingAnalyzer(
        composition=target_geometry["composition"],
        projection=target_projection,
        positions=target_geometry["positions"],
        uc=target_geometry["uc"],
        config=config,
        model=source_model # mapperがソースのucや座標を参照できるよう、source_modelを渡す
    )
    mapper = HamiltonianMapper(interpolated_functions, analyzer_target, config)
    bravais_vectors_list = list(source_model.hop.keys())
    H_mapped = mapper.construct_hermitian_hops(bravais_vectors_list)
    
    # --- Step 4: 新しいモデルの作成と出力、検証 ---
    try:
        mapped_model = tb.Model(
            # pos=target_geometry["positions"], # ターゲットの座標を使用
            uc=target_geometry["uc"],
            hop=H_mapped,
            # size=analyzer_source.size, # サイズはソースモデルと一致させる
            contains_cc=False,
        )
        logging.info("Mapped TB-model constructed successfully.")
        
        output_file = config["output_hr_file"]
        mapped_model.to_hr_file(output_file)
        logging.info(f"Reconstructed Hamiltonian saved to '{output_file}'")

        if config.get("run_validation", False):
            validate_band_structure(config, source_model, mapped_model)
        
    except ValueError as e:
        logging.error(f"Failed to construct the mapped model: {e}")

    # ▲▲▲ 修正はここまでです ▲▲▲