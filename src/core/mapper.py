# src/core/mapper.py
from typing import Callable, Dict, List
import numpy as np

from .analyzer import HoppingAnalyzer
from ..core.fitting import HoppingKey, BravaisVector

class HamiltonianMapper:
    """
    補間関数を用いて、ターゲット構造のハミルトニアンを再構築するクラス。
    """
    def __init__(self, interpolated_functions: Dict[HoppingKey, Callable], analyzer: HoppingAnalyzer, config: dict):
        self.interpolated_functions = interpolated_functions
        self.analyzer = analyzer
        self.config = config

    def construct_hermitian_hops(self, bravais_vectors: List[BravaisVector]) -> Dict[BravaisVector, np.ndarray]:
        """
        ホッピング行列の辞書を構築する。
        """
        symm_hop = {}
        processed_R = set()
        
        for R_tuple in bravais_vectors:
            if R_tuple in processed_R: continue
            
            H_R = self._construct_global_block(np.array(R_tuple))
            if self.config.get("transpose_hamiltonian", True):
                symm_hop[R_tuple] = H_R.T
            else:
                symm_hop[R_tuple] = H_R
            
            minus_R_tuple = tuple(-r for r in R_tuple)
            processed_R.add(R_tuple)
            processed_R.add(minus_R_tuple)
        return symm_hop

    def _construct_global_block(self, R: np.ndarray) -> np.ndarray:
        """指定されたブラベーベクトルRに対する完全なハミルトニアンブロックを構築する。"""
        dim = self.analyzer.size
        H_global = np.zeros((dim, dim), dtype=complex)
                
        for i in self.analyzer.site_info["site_orbitals"]:
            for j in self.analyzer.site_info["site_orbitals"]:
                H_block_global = self._calculate_block(R, i, j)
                orbs_i = self.analyzer.site_info["site_orbitals"][i]
                orbs_j = self.analyzer.site_info["site_orbitals"][j]
                H_global[np.ix_(orbs_i, orbs_j)] = H_block_global
        return H_global

    # def _calculate_block(self, R_bravais: np.ndarray, site_i: int, site_j: int) -> np.ndarray:
    #     """H_ijブロックを補間関数から計算し、逆回転を適用してグローバル座標系に戻す。"""
    #     r_vec = self.analyzer.get_displacement_vector(R_bravais, site_i, site_j)
    #     dist = np.linalg.norm(r_vec)
        
    #     labels_i = self.analyzer.site_info["site_basis_labels"][site_i]
    #     labels_j = self.analyzer.site_info["site_basis_labels"][site_j]
    #     H_ij_local = np.zeros((len(labels_i), len(labels_j)), dtype=complex)
        
    #     for m, label_m in enumerate(labels_i):
    #         for n, label_n in enumerate(labels_j):
    #             simp_i = self.analyzer._simplify_label(label_m, self.config['spin'])
    #             simp_j = self.analyzer._simplify_label(label_n, self.config['spin'])
                
    #             func = self.interpolated_functions.get((simp_i, simp_j))
    #             if func:
    #                 H_ij_local[m, n] = func(dist)
        
    #     if dist > self.config["distance_threshold"]:
    #         direction = r_vec / dist
    #         U_i = self.analyzer._get_rotation_for_site(site_i, direction)
    #         U_j = self.analyzer._get_rotation_for_site(site_j, direction)
    #         H_ij_global = U_i.T.conj() @ H_ij_local @ U_j
    #         return H_ij_global
    #     else:
    #         return H_ij_local
    
    def _calculate_block(self, R_bravais: np.ndarray, site_i: int, site_j: int) -> np.ndarray:
        """
        H_ijブロックを補間関数から計算し、逆回転を適用してグローバル座標系に戻す。
        use_ungrouped_interpolation フラグに応じてキーの生成方法を切り替える。
        """
        r_vec = self.analyzer.get_displacement_vector(R_bravais, site_i, site_j)
        dist = np.linalg.norm(r_vec)
        
        labels_i = self.analyzer.site_info["site_basis_labels"][site_i]
        labels_j = self.analyzer.site_info["site_basis_labels"][site_j]
        H_ij_local = np.zeros((len(labels_i), len(labels_j)), dtype=complex)
        
        use_ungrouped = self.config.get("use_ungrouped_interpolation", False)

        for m, label_m in enumerate(labels_i):
            for n, label_n in enumerate(labels_j):
                
                if use_ungrouped:
                    # trueの場合: 'Bi1,s -> Bi2,pz' のような生の文字列キー
                    func_key = f"{label_m} -> {label_n}"
                else:
                    # falseの場合: (('Bi','s'), ('Bi','pz')) のようなグループ化されたタプルキー
                    simp_i = self.analyzer._simplify_label(label_m, self.config['spin'])
                    simp_j = self.analyzer._simplify_label(label_n, self.config['spin'])
                    func_key = (simp_i, simp_j)
                    
                func = self.interpolated_functions.get(func_key)
                if func:
                    H_ij_local[m, n] = func(dist)
        
        if dist > self.config.get("distance_threshold", 1e-9):
            direction = r_vec / dist
            U_i = self.analyzer._get_rotation_for_site(site_i, direction)
            U_j = self.analyzer._get_rotation_for_site(site_j, direction)
            H_ij_global = U_i.T.conj() @ H_ij_local @ U_j
            return H_ij_global
        else:
            return H_ij_local