# src/core/rotation.py
from typing import Dict, Tuple
import numpy as np
import sympy as sp
from scipy.spatial.transform import Rotation

# ユーティリティモジュールから定数をインポート
from ..utils.constants import REAL_ORBITAL_ORDER_W90, COMPLEX_TO_REAL_M_VALUES, M_MAP

class RotationManager:
    """
    球面調和関数に基づき、原子軌道の回転を管理するクラス。
    計算した回転行列をキャッシュし、再計算を避ける。
    """
    _rotation_cache: Dict[Tuple[int, Tuple[float, ...]], np.ndarray] = {}

    def get_real_orbital_rotation_matrix(self, l: int, n: np.ndarray) -> np.ndarray:
        """
        角運動量 'l' の実数軌道に対する回転行列を計算する。
        この回転は、z軸をベクトル 'n' の方向に向ける。

        Args:
            l (int): 軌道の角運動量量子数。
            n (np.ndarray): 回転先の方向を示す3次元ベクトル (正規化されている必要はない)。

        Returns:
            np.ndarray: (2l+1) x (2l+1) の回転行列。基底はWannier90の規約に従う。
        """
        # 計算済みの回転行列はキャッシュから返す
        cache_key = (l, tuple(np.round(n, 8)))
        if cache_key in self._rotation_cache:
            return self._rotation_cache[cache_key]

        # Wigner-D行列と基底変換を用いて回転行列を計算
        alpha, beta, gamma = self._get_euler_angles(n)
        D = self._get_wigner_d_matrix(l, alpha, beta, gamma)
        C = self._get_complex_to_real_matrix(l)
        R = C @ D @ C.T.conj()
        R_reordered = self._reorder_basis_to_w90(R, l)

        self._rotation_cache[cache_key] = R_reordered
        return R_reordered

    @staticmethod
    def _get_euler_angles(n: np.ndarray) -> Tuple[float, float, float]:
        """z軸をベクトル 'n' の方向に向けるためのZYZオイラー角を計算する。"""
        norm = np.linalg.norm(n)
        if np.isclose(norm, 0):
            return 0.0, 0.0, 0.0
        
        n_normalized = n / norm
        
        # align_vectorsが不安定になるエッジケースを処理
        if np.isclose(np.dot(n_normalized, [0, 0, 1]), 1.0):
            return 0.0, 0.0, 0.0
        elif np.isclose(np.dot(n_normalized, [0, 0, 1]), -1.0):
            return 0.0, np.pi, 0.0
        
        rot, _ = Rotation.align_vectors([[0, 0, 1]], [n_normalized])
        return rot.as_euler('ZYZ', degrees=False)

    @staticmethod
    def _get_wigner_d_matrix(l: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
        """Wigner D行列 D^l_{m'm}を計算する。"""
        from sympy.physics.wigner import wigner_d
        D_sympy = wigner_d(l, alpha, beta, gamma)
        return sp.matrix2numpy(D_sympy, dtype=np.complex128)

    @staticmethod
    def _get_complex_to_real_matrix(l: int) -> np.ndarray:
        """
        物理で一般的な複素球面調和関数の基底から、
        Wannier90などで用いられる実球面調和関数の基底への変換行列を生成する。
        """
        dim = 2 * l + 1
        C = np.zeros((dim, dim), dtype=np.complex128)
        m_phys_ordered = list(range(l, -l - 1, -1))
        m_phys_map = {m: idx for idx, m in enumerate(m_phys_ordered)}
        m_conv_ordered = COMPLEX_TO_REAL_M_VALUES[l]

        for i, mc in enumerate(m_conv_ordered):
            if mc == 0:
                C[i, m_phys_map[0]] = 1.0
            elif mc > 0:
                C[i, m_phys_map[mc]] = (-1)**mc / np.sqrt(2)
                C[i, m_phys_map[-mc]] = 1.0 / np.sqrt(2)
            else:
                m_abs = abs(mc)
                C[i, m_phys_map[m_abs]] = 1j * (-1)**(m_abs + 1) / np.sqrt(2)
                C[i, m_phys_map[-m_abs]] = 1j / np.sqrt(2)
        return C

    @staticmethod
    def _reorder_basis_to_w90(matrix: np.ndarray, l: int) -> np.ndarray:
        """
        回転行列の基底を、内部計算で用いた規約からWannier90の規約へと並べ替える。
        """
        w90_order = REAL_ORBITAL_ORDER_W90[l]
        conv_m_values = COMPLEX_TO_REAL_M_VALUES[l]
        indices = [conv_m_values.index(M_MAP[name]) for name in w90_order]
        return matrix[np.ix_(indices, indices)]