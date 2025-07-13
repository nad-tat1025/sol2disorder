# src/core/fitting.py
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import curve_fit
import logging

# ### フィット関数の定義 ###
def exp_decay(d: np.ndarray, c1: float, c: float) -> np.ndarray:
    """指数関数的減衰を記述するフィット関数。"""
    return c1 * np.exp(-c * d)

def gaussian(d: np.ndarray, a: float, c: float) -> np.ndarray:
    """ガウス関数。d=0を中心とする。"""
    return a * np.exp(- c * d ** 2)

FIT_FUNCTIONS = {
    'exp': {
        'func': exp_decay,
        'p0_func': lambda x, y: [y[np.argmin(x)], 1.5],
        'bounds': ([-np.inf, 1e-9], [np.inf, np.inf])
    },
    'gaussian': {
        'func': gaussian,
        'p0_func': lambda x, y: [y.max(), 1.0],
        'bounds': ([-np.inf, 1e-9], [np.inf, np.inf])
    }
}


# ### 型エイリアスの定義 ###
HoppingKey = Tuple[Tuple[str, ...], Tuple[str, ...]]
HoppingData = List[Tuple[float, complex]]
HoppingDict = Dict[HoppingKey, HoppingData]
BravaisVector = Tuple[int, int, int]

# ### データ補間用の関数クラス群 ###
class InterpolationFunction:
    """カーブフィットに基づく補間関数。ホッピングのカットオフを考慮する。"""
    def __init__(self, onsite_val: complex, popt_real: Optional[np.ndarray],
                 popt_imag: Optional[np.ndarray], fit_type: str, config: dict):
        self.onsite_val = onsite_val
        self.popt_real = popt_real
        self.popt_imag = popt_imag
        self.config = config

        if fit_type not in FIT_FUNCTIONS:
            raise ValueError(f"Unsupported fit type: '{fit_type}'")
        self.func = FIT_FUNCTIONS[fit_type]['func']

    def __call__(self, d: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
        is_scalar = np.isscalar(d)
        d_arr = np.atleast_1d(d)
        result = np.zeros_like(d_arr, dtype=np.complex128)

        mask_hopping = (d_arr > self.config["distance_threshold"])
        if self.config["hopping_cutoff_distance"] is not None:
            mask_hopping &= (d_arr <= self.config["hopping_cutoff_distance"])

        if self.popt_real is not None:
            result[mask_hopping] += self.func(d_arr[mask_hopping], *self.popt_real)
        if self.popt_imag is not None:
            result[mask_hopping] += 1j * self.func(d_arr[mask_hopping], *self.popt_imag)

        mask_onsite = np.isclose(d_arr, 0)
        if np.any(mask_onsite):
            result[mask_onsite] = self.onsite_val

        return result.item() if is_scalar else result

class LinearInterpolationFunction:
    """データ点を線形補間し、オンサイトと最近接の間を外挿する関数クラス。"""
    def __init__(self, hopping_data_points: List[Tuple[float, complex]], config: dict):
        self.config = config
        
        dist_map = defaultdict(list)
        for d, v in hopping_data_points:
            dist_map[round(d, 5)].append(v)
        avg_points = {d: np.mean(v_list) for d, v_list in dist_map.items()}
        
        self.onsite_val = avg_points.get(0.0, 0.0)
        offsite_points = sorted([(d, v) for d, v in avg_points.items() if not np.isclose(d, 0)])
        
        d_coords_raw = np.array([p[0] for p in offsite_points])
        v_coords_raw = np.array([p[1] for p in offsite_points])

        cutoff = self.config["hopping_cutoff_distance"]
        if cutoff is not None:
            in_range_mask = d_coords_raw <= cutoff
            self.d_coords = d_coords_raw[in_range_mask]
            self.v_coords = v_coords_raw[in_range_mask]
            
            if self.d_coords.size == 0 or not np.isclose(self.d_coords[-1], cutoff):
                self.d_coords = np.append(self.d_coords, cutoff)
                self.v_coords = np.append(self.v_coords, 0.0 + 0.0j)
        else:
            self.d_coords = d_coords_raw
            self.v_coords = v_coords_raw

        self.nn_point = (self.d_coords[0], self.v_coords[0]) if self.d_coords.size > 0 else None
        self.nnn_point = (self.d_coords[1], self.v_coords[1]) if self.d_coords.size > 1 else None

    def __call__(self, d: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
        is_scalar = np.isscalar(d)
        d_arr = np.atleast_1d(d)
        result = np.zeros_like(d_arr, dtype=np.complex128)

        mask_onsite = np.isclose(d_arr, 0)
        mask_extrapolate = (d_arr > 1e-9) & (d_arr < self.nn_point[0]) if self.nn_point else np.zeros_like(d_arr, dtype=bool)
        mask_interpolate = (d_arr >= self.nn_point[0]) if self.nn_point else np.zeros_like(d_arr, dtype=bool)

        result[mask_onsite] = self.onsite_val

        if np.any(mask_extrapolate) and self.nn_point and self.nnn_point:
            d_nn, v_nn = self.nn_point
            d_nnn, v_nnn = self.nnn_point
            slope = (v_nnn - v_nn) / (d_nnn - d_nn)
            d_target = d_arr[mask_extrapolate]
            result[mask_extrapolate] = v_nn + slope * (d_target - d_nn)
        elif np.any(mask_extrapolate) and self.nn_point:
            result[mask_extrapolate] = self.nn_point[1]

        if np.any(mask_interpolate) and self.d_coords.size >= 1:
            d_target = d_arr[mask_interpolate]
            real_part = np.interp(d_target, self.d_coords, self.v_coords.real)
            imag_part = np.interp(d_target, self.d_coords, self.v_coords.imag)
            result[mask_interpolate] = real_part + 1j * imag_part
            
        return result.item() if is_scalar else result


# class DiscreteDataFunction:
#     """補間を行わず、オリジナルの離散データをそのまま返す関数クラス。"""
#     def __init__(self, hopping_data_points: List[Tuple[float, complex]], config: dict):
#         self.config = config
#         dist_map = defaultdict(list)
#         for d, v in hopping_data_points:
#             dist_map[round(d, 5)].append(v)
#         self.data_map = dict(dist_map)

#     def __call__(self, d: Union[float, np.ndarray]) -> Union[complex, np.ndarray]:
#         is_scalar = np.isscalar(d)
#         d_arr = np.atleast_1d(d)
        
#         def lookup_func(dist):
#             values = self.data_map.get(round(dist, 5))
#             if self.config["hopping_cutoff_distance"] is not None and dist > self.config["hopping_cutoff_distance"]:
#                 return 0.0 + 0.0j
#             return values[0] if values else 0.0 + 0.0j

#         vectorized_lookup = np.vectorize(lookup_func)
#         result = vectorized_lookup(d_arr)
        
#         return result.item() if is_scalar else result


# ### フィッティング管理クラス ###
class FittingManager:
    """ホッピングデータのフィッティングと補間関数の生成を管理するクラス。"""
    @staticmethod
    def fit_data(x: np.ndarray, y: np.ndarray, fit_type: str) -> Optional[np.ndarray]:
        """指定された関数形でデータをフィッティングする。"""
        if fit_type not in FIT_FUNCTIONS:
            raise ValueError(f"Unsupported fit type: '{fit_type}'")

        fit_info = FIT_FUNCTIONS[fit_type]
        func_to_fit = fit_info['func']
        
        try:
            initial_guess = fit_info['p0_func'](x, y)
            if len(x) < len(initial_guess):
                return None
            bounds = fit_info['bounds']
            popt, _ = curve_fit(func_to_fit, x, y, p0=initial_guess, bounds=bounds, maxfev=50000)
            return popt
        except (RuntimeError, ValueError):
            return None

    @classmethod
    def create_interpolated_functions(cls, hopping_by_type: HoppingDict, config: dict) -> Dict[HoppingKey, Callable]:
        """指定された方法に基づき、補間関数オブジェクトを生成する。"""
        fit_type = config["fit_type"]
        
        if fit_type == 'discrete':
            return {key: DiscreteDataFunction(data, config) for key, data in hopping_by_type.items() if data}
        
        if fit_type == 'linear':
            return {key: LinearInterpolationFunction(data, config) for key, data in hopping_by_type.items() if data}
        
        # カーブフィットの場合
        interpolated_functions = {}
        for key, data in hopping_by_type.items():
            if not data: continue
            
            dist_map = defaultdict(list)
            for d, v in data:
                dist_map[round(d, 5)].append(v)
            avg_data = {d: np.mean(v_list) for d, v_list in dist_map.items()}
            
            onsite_val = avg_data.get(0.0, 0.0)
            d_fit_avg = np.array([d for d in sorted(avg_data.keys()) if not np.isclose(d, 0)])
            v_fit_avg = np.array([avg_data[d] for d in d_fit_avg])

            if v_fit_avg.size > 0 and np.max(np.abs(v_fit_avg)) < config["zero_hop_threshold"]:
                num_params = len(FIT_FUNCTIONS[fit_type]['p0_func'](np.array([1]), np.array([1])))
                popt_zero = np.array([0.0] * num_params)
                func = InterpolationFunction(0.0, popt_zero, None, fit_type, config)
                interpolated_functions[key] = func
                continue
            
            popt_real, popt_imag = None, None
            if d_fit_avg.size >= 2:
                popt_real = cls.fit_data(d_fit_avg, v_fit_avg.real, fit_type)
                if not np.allclose(v_fit_avg.imag, 0):
                    popt_imag = cls.fit_data(d_fit_avg, v_fit_avg.imag, fit_type)
            
            func = InterpolationFunction(onsite_val, popt_real, popt_imag, fit_type, config)
            interpolated_functions[key] = func
            
        return interpolated_functions

    @classmethod
    def create_interpolated_functions_from_raw(cls, hopping_dict: Dict[str, HoppingData], config: dict) -> Dict[str, Callable]:
        """
        [テスト用] グループ化されていない生のホッピングデータ(hopping_dict)から補間関数を生成する。
        キーは ('Bi','s') -> ('Bi','pz') のようなタプルではなく、'Bi1,s -> Bi2,pz' のような文字列。
        """
        logging.info("Creating interpolation functions from raw (ungrouped) hopping data...")
        fit_type = config["fit_type"]

        # このメソッドは、fit_typeが 'linear', 'exp', 'gaussian' の場合を想定
        if fit_type not in ['linear', 'exp', 'gaussian']:
            raise NotImplementedError(f"Raw interpolation for fit_type '{fit_type}' is not implemented.")

        interpolated_functions = {}
        # キーが文字列である hopping_dict を直接ループ処理
        for key, data in hopping_dict.items():
            if not data:
                continue
            
            # fit_typeに応じて適切な補間関数オブジェクトを生成
            if fit_type == 'linear':
                interpolated_functions[key] = LinearInterpolationFunction(data, config)
            else: # exp, gaussian
                # データ平滑化（同じ距離の値を平均）
                dist_map = defaultdict(list)
                for d, v in data:
                    dist_map[round(d, 5)].append(v)
                avg_data = {d: np.mean(v_list) for d, v_list in dist_map.items()}
                
                onsite_val = avg_data.get(0.0, 0.0)
                d_fit_avg = np.array([d for d in sorted(avg_data.keys()) if not np.isclose(d, 0)])
                v_fit_avg = np.array([avg_data[d] for d in d_fit_avg])

                popt_real, popt_imag = None, None
                if d_fit_avg.size >= 2:
                    popt_real = cls.fit_data(d_fit_avg, v_fit_avg.real, fit_type)
                    if not np.allclose(v_fit_avg.imag, 0):
                        popt_imag = cls.fit_data(d_fit_avg, v_fit_avg.imag, fit_type)
                
                func = InterpolationFunction(onsite_val, popt_real, popt_imag, fit_type, config)
                interpolated_functions[key] = func
        
        return interpolated_functions