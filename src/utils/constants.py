# src/utils/constants.py
from typing import Dict, List

OrbitalName = str

# 軌道の角運動量量子数 'l' を軌道名からマッピング
L_FROM_ORBITAL: Dict[OrbitalName, int] = {
    's': 0,
    'px': 1, 'py': 1, 'pz': 1,
    'dxy': 2, 'dyz': 2, 'dz2': 2, 'dxz': 2, 'dx2-y2': 2,
    'fz3': 3, 'fxz2': 3, 'fyz2': 3, 'fz(x2-y2)': 3, 'fxyz': 3, 'fx(x2-3y2)': 3, 'fy(3x2-y2)': 3
}

# Wannier90で用いられる実数球面調和関数の軌道の順序
REAL_ORBITAL_ORDER_W90: Dict[int, List[OrbitalName]] = {
    0: ['s'],
    1: ['pz', 'px', 'py'],
    2: ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy'],
    3: ['fz3', 'fxz2', 'fyz2', 'fz(x2-y2)', 'fxyz', 'fx(x2-3y2)', 'fy(3x2-y2)']
}

# 複素球面調和関数から実球面調和関数へ変換する際の磁気量子数mの規約
COMPLEX_TO_REAL_M_VALUES: Dict[int, List[int]] = {
    0: [0],
    1: [0, 1, -1],
    2: [0, 1, -1, 2, -2],
    3: [0, 1, -1, 2, -2, 3, -3]
}

# 各実数軌道名に対応する磁気量子数mのマッピング
M_MAP: Dict[OrbitalName, int] = {
    's': 0,
    'pz': 0, 'px': 1, 'py': -1,
    'dz2': 0, 'dxz': 1, 'dyz': -1, 'dx2-y2': 2, 'dxy': -2,
    'fz3': 0, 'fxz2': 1, 'fyz2': -1, 'fz(x2-y2)': 2, 'fxyz': -2, 'fx(x2-3y2)': 3, 'fy(3x2-y2)': -3
}