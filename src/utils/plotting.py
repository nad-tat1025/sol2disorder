# src/utils/plotting.py
import logging
from collections import defaultdict
from math import ceil, sqrt
from typing import Dict, Callable

import matplotlib.pyplot as plt
import numpy as np
import tbmodels as tb

# 別のモジュールで定義した型エイリアスをインポート
from ..core.fitting import HoppingDict, HoppingData

def validate_band_structure(config: dict, source_model: tb.Model, mapped_model: tb.Model):
    """元のモデルとマッピングされたモデルのバンド構造を計算し、比較プロットする。"""
    logging.info("Validating mapping by comparing band structures...")

    if config.get("target_structure_file") and config.get("target_structure_file") != config.get("source_structure_file"):
        logging.warning("Source and target structures are different.")
        logging.warning("Band structure comparison is only meaningful if the lattice vectors are identical.")

    k_path_dict = config.get("k_path")
    if not k_path_dict:
        logging.warning("k_path is not defined. Skipping band structure validation.")
        return

    K_POINTS_PER_SEGMENT = 100

    def _calc_bands_manual(model: tb.Model, path_dict: dict) -> np.ndarray:
        all_eigenvalues = []
        for segment in path_dict.values():
            k_start, k_end = segment[0], segment[1]
            k_points_on_segment = np.linspace(k_start, k_end, num=K_POINTS_PER_SEGMENT, endpoint=True)
            for k in k_points_on_segment:
                eigenvalues = model.eigenval(k)
                all_eigenvalues.append(eigenvalues)
        return np.array(all_eigenvalues)

    evals_source = _calc_bands_manual(source_model, k_path_dict)
    evals_mapped = _calc_bands_manual(mapped_model, k_path_dict)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(evals_source, color="b", label="Source")
    ax.plot(evals_mapped, color="r", linestyle='--', label="Mapped")

    path_keys = list(k_path_dict.keys())
    tick_positions = [0]
    tick_labels = [path_keys[0].split('-')[0]] 

    current_pos = 0
    for key in path_keys:
        current_pos += K_POINTS_PER_SEGMENT
        position = current_pos - 1
        tick_positions.append(position)
        tick_labels.append(key.split('-')[1])
        if position < len(evals_source) - 1:
            ax.axvline(x=position, color='gray', linestyle='--', linewidth=0.8)
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy [eV]")
    ax.set_xlim(0, len(evals_source) - 1)
    # ax.legend()
    plt.tight_layout()
    plt.show()


def print_hopping_summary(hopping_data: HoppingDict):
    """オンサイト、最近接(NN)、次近接(NNN)ホッピングパラメータのサマリーを表示する。"""
    logging.info("--- Hopping Parameter Summary (NN and NNN) ---")
    header = f"{'Hopping Type':<25} | {'On-site [eV]':<25} | {'NN Hopping [eV] @ Dist [Å]':<45} | {'NNN Hopping [eV] @ Dist [Å]':<45}"
    logging.info(header)
    logging.info("-" * len(header))

    for key in sorted(hopping_data.keys(), key=str):
        values = hopping_data[key]
        if not values: continue

        type_i_str = "".join(map(str, key[0]))
        type_j_str = "".join(map(str, key[1]))
        type_str = f"{type_i_str} -> {type_j_str}"

        onsite_vals = [v for d, v in values if np.isclose(d, 0)]
        offsite_vals = [(d, v) for d, v in values if not np.isclose(d, 0)]
        
        onsite_energy_str = f"{(np.mean(onsite_vals)).real:+.5f}" if onsite_vals else "N/A"

        nn_info_str, nnn_info_str = "N/A", "N/A"
        if offsite_vals:
            dist_map = defaultdict(list)
            for d, v in offsite_vals:
                dist_map[round(d, 5)].append(v)
            
            unique_distances = sorted(dist_map.keys())
            
            if len(unique_distances) > 0:
                nn_dist = unique_distances[0]
                nn_hopping_avg = np.mean(dist_map[nn_dist])
                nn_info_str = f"{nn_hopping_avg.real:+.5f} @ {nn_dist:.5f}"
            if len(unique_distances) > 1:
                nnn_dist = unique_distances[1]
                nnn_hopping_avg = np.mean(dist_map[nnn_dist])
                nnn_info_str = f"{nnn_hopping_avg.real:+.5f} @ {nnn_dist:.5f}"

        logging.info(f"{type_str:<25} | {onsite_energy_str:<25} | {nn_info_str:<45} | {nnn_info_str:<45}")
    logging.info("-" * len(header))


def _get_key_hoppings(hopping_values: HoppingData) -> dict:
    """オンサイト、最近接(NN)、次近接(NNN)のホッピング値を抽出するヘルパー関数。"""
    summary = {'onsite': 'N/A', 'nn': 'N/A', 'nnn': 'N/A'}
    if not hopping_values:
        return summary

    onsite_vals = [v for d, v in hopping_values if np.isclose(d, 0)]
    if onsite_vals:
        summary['onsite'] = f"{(np.mean(onsite_vals)).real:+.5f}"

    offsite_vals = sorted([(d, v) for d, v in hopping_values if not np.isclose(d, 0)])
    if offsite_vals:
        dist_map = defaultdict(list)
        for d, v in offsite_vals:
            dist_map[round(d, 5)].append(v)
        unique_distances = sorted(dist_map.keys())
        
        if len(unique_distances) > 0:
            nn_dist = unique_distances[0]
            nn_hopping_avg = np.mean(dist_map[nn_dist])
            summary['nn'] = f"{nn_hopping_avg.real:+.5f} @ {nn_dist:.5f}"
        if len(unique_distances) > 1:
            nnn_dist = unique_distances[1]
            nnn_hopping_avg = np.mean(dist_map[nnn_dist])
            summary['nnn'] = f"{nnn_hopping_avg.real:+.5f} @ {nnn_dist:.5f}"
    
    return summary

def compare_hopping_parameters(original_hoppings: HoppingDict, mapped_hoppings: HoppingDict):
    """マッピング前後のホッピングパラメータを比較する表を出力する。"""
    logging.info("--- Hopping Parameter Comparison (Original vs. Mapped) ---")
    header = f"{'Hopping Type':<25} | {'Parameter':<15} | {'Original Value':<35} | {'Mapped Value':<35}"
    logging.info(header)
    logging.info("-" * len(header))

    all_keys = sorted(list(set(original_hoppings.keys()) | set(mapped_hoppings.keys())), key=str)

    for key in all_keys:
        type_i_str = "".join(map(str, key[0]))
        type_j_str = "".join(map(str, key[1]))
        type_str = f"{type_i_str} -> {type_j_str}"

        orig_vals = _get_key_hoppings(original_hoppings.get(key, []))
        mapped_vals = _get_key_hoppings(mapped_hoppings.get(key, []))
        
        logging.info(f"{type_str:<25} | {'On-site [eV]':<15} | {orig_vals['onsite']:<35} | {mapped_vals['onsite']:<35}")
        logging.info(f"{'':<25} | {'NN [eV @ Å]':<15} | {orig_vals['nn']:<35} | {mapped_vals['nn']:<35}")
        logging.info(f"{'':<25} | {'NNN [eV @ Å]':<15} | {orig_vals['nnn']:<35} | {mapped_vals['nnn']:<35}")
        logging.info("-" * (len(header) - 15))


def plot_hopping_with_fit(hopping_by_type: HoppingDict, interpolated_functions: Dict[Callable, any]):
    """抽出されたホッピングデータとフィット/補間関数をプロットする。"""
    keys_to_plot = [k for k, v in hopping_by_type.items() if v]
    n_plots = len(keys_to_plot)

    if n_plots == 0:
        logging.info("No data to plot.")
        return

    ncols = int(ceil(sqrt(n_plots)))
    nrows = int(ceil(n_plots / ncols))
    
    figsize = (3.0 * ncols, 2.0 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, key in zip(axes, keys_to_plot):
        distances = np.array([d for d, _ in hopping_by_type[key]])
        values = np.array([v for _, v in hopping_by_type[key]])

        ax.scatter(distances, values.real, color='blue', s=10, label='data (real)')
        if not np.allclose(values.imag, 0):
            ax.scatter(distances, values.imag, color='orange', s=10, marker='x', label='data (imag)')

        fit_func = interpolated_functions.get(key)
        if fit_func:
            d_fit = np.linspace(0, 10, 300)
            t_fit = fit_func(d_fit)
            ax.plot(d_fit, t_fit.real, color='red', label='fit (real)')
            if not np.allclose(t_fit.imag, 0):
                ax.plot(d_fit, t_fit.imag, color='green', linestyle='--', label='fit (imag)')

        title_str = f"{''.join(key[0])} -> {''.join(key[1])}"
        ax.set_title(title_str, fontsize=10)
        ax.set_xlabel("Distance [Å]")
        ax.set_ylabel("t [eV]")
        # ax.legend(fontsize=8)
        ax.set_xlim(-0.5, 10)
        ax.set_ylim(-3, 3)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)

    for ax_idx in range(n_plots, len(axes)):
        axes[ax_idx].axis("off")

    plt.tight_layout()
    plt.show()