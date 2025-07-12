# src/utils/io.py
import logging
from collections import Counter
from typing import Any, Dict,  Tuple, List, Optional
from ase.io import read

def _read_geometry(file_path: str) -> Optional[Dict[str, Any]]:
    """ASEを使用して構造ファイルを読み込み、ジオメトリ情報を辞書として返す。"""
    try:
        logging.info(f"--- Reading geometry from: {file_path} ---")
        atoms = read(file_path)
        geometry = {
            "positions": atoms.get_scaled_positions(wrap=False),
            "uc": atoms.get_cell(),
            "composition": dict(Counter(atoms.get_chemical_symbols()))
        }
        logging.info("  > Geometry read successfully.")
        return geometry
    except Exception as e:
        logging.error(f"Failed to read structure file '{file_path}'. Details: {e}")
        return None


def parse_wannier_win(win_file_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    wannier90.winファイルを解析し、projectionsとkpoint_pathを抽出する。

    Args:
        win_file_path (str): wannier90.winファイルのパス。

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: (projections, kpoint_path) のタプル。
    """
    projections = {}
    kpoint_path = {}
    in_block = None

    logging.info(f"--- Parsing projections and k-path from: {win_file_path} ---")
    try:
        with open(win_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # コメント行や空行はスキップ
                if not line or line.startswith(('!', '#')):
                    continue

                # ブロックの開始/終了を判定
                if 'begin projections' in line.lower():
                    in_block = 'projections'
                    continue
                elif 'end projections' in line.lower():
                    in_block = None
                    continue
                elif 'begin kpoint_path' in line.lower():
                    in_block = 'kpoint_path'
                    continue
                elif 'end kpoint_path' in line.lower():
                    in_block = None
                    continue

                # 各ブロック内の処理
                if in_block == 'projections':
                    parts = line.split(':')
                    if len(parts) == 2:
                        element = parts[0].strip()
                        orbs = [orb.strip() for orb in parts[1].split(';')]
                        projections[element] = orbs
                        logging.info(f"  > Found projection for {element}: {orbs}")

                elif in_block == 'kpoint_path':
                    parts = line.split()
                    if len(parts) == 8:
                        l1, x1, y1, z1, l2, x2, y2, z2 = parts
                        path_label = f"{l1}-{l2}"
                        coords1 = (float(x1), float(y1), float(z1))
                        coords2 = (float(x2), float(y2), float(z2))
                        kpoint_path[path_label] = [coords1, coords2]
                        logging.info(f"  > Found k-path segment: {path_label}")

    except FileNotFoundError:
        logging.error(f"wannier90.win file not found at: {win_file_path}")

    return projections, kpoint_path