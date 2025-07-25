# src/core/analyzer.py
import logging
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tbmodels as tb
from scipy.linalg import block_diag

# このプロジェクトで定義した他のモジュールをインポート
from .rotation import RotationManager
from ..utils.constants import L_FROM_ORBITAL, REAL_ORBITAL_ORDER_W90, M_MAP
from ..core.fitting import HoppingDict, HoppingData, BravaisVector

np.set_printoptions(linewidth=120, precision=4, suppress=True)


class HoppingAnalyzer:
    """
    tbmodels.Modelオブジェクトまたは構造情報からホッピングパラメータを解析するクラス。
    """
    def __init__(self, composition: Dict[str, int], projection: Dict[str, List[str]],
                 positions: np.ndarray, config: dict,
                 model: Optional[tb.Model] = None, uc: Optional[np.ndarray] = None):
        self.model = model
        self.composition = composition
        self.projection = projection
        self.positions = positions
        self.config = config

        if uc is None and model is None:
            raise ValueError("引数として 'uc' または 'model' のどちらかは必須です。")
        self.uc = model.uc if model is not None else uc

        self.rotation_manager = RotationManager()
        self.site_info = self._get_site_info()

        if model:
            self.size = model.size
        else:
            num_orbitals = 0
            for elem, count in self.composition.items():
                if elem in self.projection:
                    num_orbitals += count * len(self.projection[elem])
            
            spin_multiplier = 2 if self.config.get('spin', False) else 1
            self.size = num_orbitals * spin_multiplier

    def _get_site_info(self) -> Dict[str, Dict[int, Any]]:
        """サイトごとの基底指数、軌道ラベル、軌道タイプをマッピングした辞書を生成する。"""
        info: Dict[str, Dict[int, Any]] = {
            "site_orbitals": {}, "site_basis_labels": {}, "site_orbital_types": {}
        }
        site_idx, basis_idx = 0, 0
        
        for elem, count in self.composition.items():
            for n in range(count):
                orbs_indices, labels, types = [], [], []
                for orb in self.projection[elem]:
                    for s in (['↑','↓'] if self.config['spin'] else ['']):
                        orbs_indices.append(basis_idx)
                        labels.append(f"{elem}{n+1},{orb}" + (f", {s}" if self.config['spin'] else ""))
                        types.append(orb)
                        basis_idx += 1
                info["site_orbitals"][site_idx] = orbs_indices
                info["site_basis_labels"][site_idx] = labels
                info["site_orbital_types"][site_idx] = types
                site_idx += 1
        return info

    def _get_rotation_for_site(self, site_idx: int, n: np.ndarray) -> np.ndarray:
        """指定されたサイト上の全軌道に対する回転行列を生成する。"""
        site_orbs = self.site_info["site_orbital_types"][site_idx]
        unique_orbs = sorted(list(set(site_orbs)), key=site_orbs.index)
        
        orbs_by_l: Dict[int, List[str]] = defaultdict(list)
        for orb in unique_orbs:
            orbs_by_l[L_FROM_ORBITAL[orb]].append(orb)
        
        R_blocks: List[np.ndarray] = []
        for l in sorted(orbs_by_l.keys()):
            R_l = self.rotation_manager.get_real_orbital_rotation_matrix(l, n)
            indices = [REAL_ORBITAL_ORDER_W90[l].index(orb) for orb in orbs_by_l[l]]
            R_blocks.append(R_l[np.ix_(indices, indices)])
        
        site_R = block_diag(*R_blocks)
        return np.kron(site_R, np.eye(2)) if self.config['spin'] else site_R

    def get_displacement_vector(self, R_bravais: np.ndarray, site_i: int, site_j: int) -> np.ndarray:
        """サイトiからサイトjへの変位ベクトルを計算する。"""
        R_cart = R_bravais.dot(self.uc)
        relative_pos_vector = self.positions[site_j] - self.positions[site_i]
        
        position_units = self.config.get("position_units", "fractional")
        if position_units == 'fractional':
            delta_pos_cart = relative_pos_vector.dot(self.uc)
        elif position_units == 'cartesian':
            delta_pos_cart = relative_pos_vector
        else:
            raise ValueError(f"Unknown position_units: '{position_units}'")

        return R_cart + delta_pos_cart

    def _symmetrize_hoppings(self) -> Dict[BravaisVector, np.ndarray]:
        """ハミルトニアンのエルミート対称性 H(R) = H(-R)† を保証するようにホッピングを対称化する。"""
        symm_hop = {}
        processed_R = set()
        
        for R_tuple in list(self.model.hop.keys()):
            if R_tuple in processed_R: continue
            
            H_R = self.model.hop[R_tuple].T.conj()
            symm_hop[R_tuple] = H_R

            minus_R_tuple = tuple(-r for r in R_tuple)
            # H_minus_R_dagger = self.model.hop.get(minus_R_tuple, H_R).T.conj()
            # symm_hop[R_tuple] = (H_R + H_minus_R_dagger) / 2

            processed_R.add(R_tuple)
            processed_R.add(minus_R_tuple)
        return symm_hop

    def extract_and_rotate_hoppings(self) -> Dict[str, HoppingData]:
        """ホッピングを抽出し、サイト間ベクトルに沿って回転させ、局所座標系に変換する。"""
        hopping_dict = defaultdict(list)
        
        if self.config.get("use_hermitian_symmetrization", True):
            hopping_matrices = self._symmetrize_hoppings()
        else:
            hopping_matrices = self.model.hop
        
        for R_tuple, H_matrix in hopping_matrices.items():
            for i in self.site_info["site_orbitals"]:
                for j in self.site_info["site_orbitals"]:
                    r_vec = self.get_displacement_vector(np.array(R_tuple), i, j)
                    dist = np.linalg.norm(r_vec)
                    
                    orbs_i = self.site_info["site_orbitals"][i]
                    orbs_j = self.site_info["site_orbitals"][j]
                    H_ij = H_matrix[np.ix_(orbs_i, orbs_j)]
                    
                    if dist > self.config["distance_threshold"]:
                        direction = r_vec / dist
                        U_i = self._get_rotation_for_site(i, direction)
                        U_j = self._get_rotation_for_site(j, direction)
                        H_local = U_i @ H_ij @ U_j.T.conj()
                    else:
                        H_local = H_ij
                    
                    labels_i = self.site_info["site_basis_labels"][i]
                    labels_j = self.site_info["site_basis_labels"][j]
                    for m, label_m in enumerate(labels_i):
                        for n, label_n in enumerate(labels_j):
                            hopping_dict[f"{label_m} -> {label_n}"].append((dist, H_local[m, n]))
        return hopping_dict

    @staticmethod
    def _simplify_label(label: str, spin: bool) -> Tuple[str, ...]:
        """基底ラベルを単純化する (例: 'Hg1,s,↑' -> ('Hg','s','↑'))。"""
        parts = label.split(',')
        element = ''.join(filter(str.isalpha, parts[0]))
        orb = parts[1]
        
        if spin and len(parts) > 2:
            return (element, orb, parts[2])
        return (element, orb)

    def group_hoppings_by_type(self, hopping_dict: Dict[str, HoppingData]) -> HoppingDict:
        """ホッピングを軌道の種類でグループ化する。"""
        hopping_by_type = defaultdict(list)
        for key, values in hopping_dict.items():
            label_i, label_j = key.split(" -> ")
            simp_i = self._simplify_label(label_i, self.config['spin'])
            simp_j = self._simplify_label(label_j, self.config['spin'])
            hopping_by_type[(simp_i, simp_j)].extend(values)
        return hopping_by_type

    def average_symmetric_hoppings(self, hopping_by_type: HoppingDict) -> HoppingDict:
        """対称性により等価であるべきホッピングを平均化する。"""
        averaged_data = hopping_by_type.copy()
        processed_keys = set()

        def swap_p_xy(t: Tuple[str, ...]) -> Tuple[str, ...]:
            elem, orb, *spin_info = t
            if orb == 'px': return (elem, 'py', *spin_info)
            if orb == 'py': return (elem, 'px', *spin_info)
            return t

        for key in list(averaged_data.keys()):
            if key in processed_keys: continue
            
            type_i, type_j = key
            sym_key = (swap_p_xy(type_i), swap_p_xy(type_j))

            if sym_key == key or sym_key in processed_keys or sym_key not in averaged_data:
                continue

            values1 = averaged_data[key]
            values2 = averaged_data[sym_key]

            dist_map = defaultdict(list)
            for d, v in values1 + values2:
                dist_map[round(d, 5)].append(v)
            
            averaged_values = sorted([(d_rounded, np.mean(v_list)) for d_rounded, v_list in dist_map.items()])

            averaged_data[key] = averaged_values
            averaged_data[sym_key] = averaged_values
            processed_keys.add(key)
            processed_keys.add(sym_key)
            
        return averaged_data