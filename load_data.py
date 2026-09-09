"""
load_data.py - Bruker data reader & Excel loader
=================================================
Reads Bruker 2dseq files, splits Fat/Water, loads Excel mouse list.
"""

import os
import numpy as np
import pandas as pd


def find_mouse_folder(data_root, dam_id):
    """Find the folder containing this Dam ID."""
    dam_str = str(int(dam_id))
    for folder_name in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder_name)
        if os.path.isdir(folder_path):
            if dam_str in folder_name:
                return folder_path
    raise FileNotFoundError(f"No folder found for Dam ID {dam_id} in {data_root}")


def parse_visu_pars(visu_pars_path):
    """Parse Bruker visu_pars file and extract key parameters."""
    params = {}

    with open(visu_pars_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('##$'):
            key_val = line[3:]
            if '=' in key_val:
                key, val = key_val.split('=', 1)
                val = val.strip()

                # Check if value continues on next lines (arrays)
                if val.startswith('(') and ')' in val:
                    array_str = ''
                    i += 1
                    while i < len(lines) and not lines[i].startswith('##') and not lines[i].startswith('$$'):
                        array_str += lines[i].strip() + ' '
                        i += 1
                    params[key] = _parse_array(val, array_str)
                    continue
                else:
                    params[key] = _parse_value(val)
        i += 1

    return params


def _parse_value(val):
    """Parse a single value."""
    if val.startswith('@'):
        parts = val.split('*')
        if len(parts) == 2:
            n = int(parts[0][1:])
            v = float(parts[1].strip('()'))
            return [v] * n
    try:
        return int(val)
    except ValueError:
        try:
            return float(val)
        except ValueError:
            return val


def _parse_array(declaration, array_str):
    """Parse array values."""
    if '@' in array_str:
        parts = array_str.strip().split('*')
        if len(parts) == 2:
            n = int(parts[0].strip().lstrip('@'))
            v = float(parts[1].strip('() '))
            return [v] * n

    values = array_str.strip().split()
    try:
        return [int(v) for v in values if v]
    except ValueError:
        try:
            return [float(v) for v in values if v]
        except ValueError:
            return array_str.strip()


def read_bruker_2dseq(scan_path, pdata_id=1):
    """
    Read Bruker 2dseq binary data with parameters from visu_pars.

    Parameters
    ----------
    scan_path : str - Path to the scan folder (e.g., .../mouse_folder/2/)
    pdata_id : int - Reconstruction ID (1, 2, or 3)

    Returns
    -------
    data : numpy array (frames, height, width)
    params : dict with visu_pars parameters
    """
    pdata_path = os.path.join(scan_path, 'pdata', str(pdata_id))
    visu_pars_path = os.path.join(pdata_path, 'visu_pars')
    data_path = os.path.join(pdata_path, '2dseq')

    if not os.path.exists(visu_pars_path):
        raise FileNotFoundError(f"visu_pars not found: {visu_pars_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"2dseq not found: {data_path}")

    params = parse_visu_pars(visu_pars_path)

    frame_count = params.get('VisuCoreFrameCount', 1)
    core_size = params.get('VisuCoreSize', [256, 256])
    if isinstance(core_size, list):
        nx, ny = core_size[0], core_size[1]
    else:
        nx = ny = 256

    word_type = params.get('VisuCoreWordType', '_16BIT_SGN_INT')

    dtype_map = {
        '_8BIT_UNSGN_INT': np.uint8,
        '_16BIT_SGN_INT': np.int16,
        '_32BIT_SGN_INT': np.int32,
        '_32BIT_FLOAT': np.float32,
    }
    dtype = dtype_map.get(word_type, np.int16)

    data = np.fromfile(data_path, dtype=dtype)

    expected_size = frame_count * ny * nx
    if data.size >= expected_size:
        data = data[:expected_size]
        data = data.reshape(frame_count, ny, nx)
    else:
        raise ValueError(
            f"Data size mismatch: expected {expected_size}, got {data.size}. "
            f"FrameCount={frame_count}, Size={nx}x{ny}"
        )

    return data.astype(np.float64), params


def split_fat_water(data):
    """
    Split a Fat+Water 2dseq array into separate Fat and Water volumes.
    Assumes first half = Fat slices, second half = Water slices.
    """
    n_total = data.shape[0]
    n_slices = n_total // 2

    fat = data[:n_slices]
    water = data[n_slices:]

    return fat, water


def get_voxel_dimensions(params):
    """
    Extract voxel dimensions from visu_pars parameters.
    Returns (dx, dy, dz) in mm.
    """
    extent = params.get('VisuCoreExtent', [40, 40])
    size = params.get('VisuCoreSize', [150, 150])

    if isinstance(extent, list) and isinstance(size, list):
        dx = extent[0] / size[0]
        dy = extent[1] / size[1]
    else:
        dx = dy = 0.267

    dz = params.get('VisuCoreSlicePacksSliceDist', [2.4])
    if isinstance(dz, list):
        dz = dz[0]

    return (dx, dy, dz)


def load_excel_list(excel_path):
    """
    Load Excel file with Dam ID and scan folder number.
    Expected columns: 'Dam ID' (numeric), 'scan' (numeric)
    """
    df = pd.read_excel(excel_path)

    # Drop rows where Dam ID or scan is empty/NaN
    df = df.dropna(subset=['Dam ID', 'scan'])

    col_map = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if 'dam' in col_lower or 'id' in col_lower:
            col_map['Dam ID'] = col
        elif 'scan' in col_lower:
            col_map['scan'] = col

    if 'Dam ID' not in col_map or 'scan' not in col_map:
        print(f"WARNING: Could not auto-detect columns. Found: {list(df.columns)}")
        print("Using first two columns as 'Dam ID' and 'scan'")
        col_map['Dam ID'] = df.columns[0]
        col_map['scan'] = df.columns[1]

    result = df[[col_map['Dam ID'], col_map['scan']]].copy()
    result.columns = ['Dam ID', 'scan']
    result['Dam ID'] = result['Dam ID'].astype(int)
    result['scan'] = result['scan'].astype(int)

    return result
