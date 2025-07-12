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

    # def construct_hermitian_hops(self, bravais_vectors: List[BravaisVector]) -> Dict[BravaisVector, np.ndarray]:
    #     """ホッピング行列の辞書を構築し、エルミート対称性を保証する。"""
    #     final_hop = {}
        
    #     if not self.config.get("use_hermitian_symmetrization", True):
    #         for R_tuple in bravais_vectors:
    #             final_hop[R_tuple] = self._construct_global_block(np.array(R_tuple))
    #         return final_hop

    #     processed_R = set()
    #     for R_tuple in bravais_vectors:
    #         if R_tuple in processed_R: continue
            
    #         H_R = self._construct_global_block(np.array(R_tuple))
    #         # final_hop[R_tuple] = H_R
    #         final_hop[R_tuple] = H_R.T.conj() # TEST
            
            
    #         minus_R_tuple = tuple(-r for r in R_tuple)
    #         if R_tuple != minus_R_tuple:
    #             # final_hop[minus_R_tuple] = H_R.T.conj()
    #             final_hop[minus_R_tuple] = H_R # TEST
            
    #         processed_R.add(R_tuple)
    #         processed_R.add(minus_R_tuple)
        
    #     if (0, 0, 0) in final_hop:
    #         H_0 = final_hop[(0, 0, 0)]
    #         final_hop[(0, 0, 0)] = (H_0 + H_0.T.conj()) / 2.0
            
    #     return final_hop

    def construct_hermitian_hops(self, bravais_vectors: List[BravaisVector]) -> Dict[BravaisVector, np.ndarray]:
        """
        ホッピング行列の辞書を構築する。
        ループ構造を統一し、use_hermitian_symmetrizationフラグは計算方法の切り替えにのみ使用する。
        """
        final_hop = {}
        processed_R = set()
        symmetrize = self.config.get("use_hermitian_symmetrization", True)

        for R_tuple in bravais_vectors:
            if R_tuple in processed_R:
                continue

            # H(R)を計算
            H_R = self._construct_global_block(np.array(R_tuple))
            final_hop[R_tuple] = H_R

            minus_R_tuple = tuple(-r for r in R_tuple)
            if R_tuple != minus_R_tuple:
                if symmetrize:
                    # trueの場合: H(-R) = H(R)† を適用して対称性を保証
                    final_hop[minus_R_tuple] = H_R.T.conj()
                else:
                    # falseの場合: H(-R)も独立して計算する
                    H_minus_R = self._construct_global_block(np.array(minus_R_tuple))
                    final_hop[minus_R_tuple] = H_minus_R
            
            processed_R.add(R_tuple)
            processed_R.add(minus_R_tuple)
        
        # オンサイト項がエルミートになるように保証 (どちらのケースでも実行)
        if (0, 0, 0) in final_hop:
            H_0 = final_hop[(0, 0, 0)]
            final_hop[(0, 0, 0)] = (H_0 + H_0.T.conj()) / 2.0
            
        return final_hop

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

    def _calculate_block(self, R_bravais: np.ndarray, site_i: int, site_j: int) -> np.ndarray:
        """H_ijブロックを補間関数から計算し、逆回転を適用してグローバル座標系に戻す。"""
        r_vec = self.analyzer.get_displacement_vector(R_bravais, site_i, site_j)
        dist = np.linalg.norm(r_vec)
        
        labels_i = self.analyzer.site_info["site_basis_labels"][site_i]
        labels_j = self.analyzer.site_info["site_basis_labels"][site_j]
        H_ij_local = np.zeros((len(labels_i), len(labels_j)), dtype=complex)
        
        for m, label_m in enumerate(labels_i):
            for n, label_n in enumerate(labels_j):
                simp_i = self.analyzer._simplify_label(label_m, self.config['spin'])
                simp_j = self.analyzer._simplify_label(label_n, self.config['spin'])
                
                func = self.interpolated_functions.get((simp_i, simp_j))
                if func:
                    H_ij_local[m, n] = func(dist)
        
        if dist > self.config["distance_threshold"]:
            direction = r_vec / dist
            U_i = self.analyzer._get_rotation_for_site(site_i, direction)
            U_j = self.analyzer._get_rotation_for_site(site_j, direction)
            H_ij_global = U_i.T.conj() @ H_ij_local @ U_j
            return H_ij_global
        else:
            return H_ij_local