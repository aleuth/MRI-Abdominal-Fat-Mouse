"""
segmentation.py - Fat threshold, SAT/VAT separation, volume calculation, QC
=============================================================================
Includes thorax/abdomen regional volumes and total analyzed body volume.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import binary_fill_holes, binary_erosion, binary_dilation
from skimage.filters import threshold_otsu
from skimage.draw import polygon as ski_polygon
import matplotlib.pyplot as plt


def get_contrast_limits(image):
    """Get good contrast limits using percentiles of non-zero pixels."""
    nonzero = image[image > 0]
    if len(nonzero) > 0:
        vmin, vmax = np.percentile(nonzero, [2, 98])
    else:
        vmin, vmax = 0, 1
    return vmin, vmax


def create_body_mask(combined_volume):
    """
    Create a binary mask of the mouse body from the combined image.
    Uses Otsu thresholding + morphological cleanup.
    """
    body_mask = np.zeros_like(combined_volume, dtype=bool)

    for i in range(combined_volume.shape[0]):
        sl = combined_volume[i]
        if sl.max() == 0:
            continue

        # Otsu threshold at lower level to include all tissue
        try:
            thresh = threshold_otsu(sl[sl > 0]) * 0.3
        except ValueError:
            thresh = sl.mean() * 0.3

        mask = sl > thresh
        mask = binary_fill_holes(mask)
        mask = binary_erosion(mask, iterations=1)
        mask = binary_dilation(mask, iterations=1)

        body_mask[i] = mask

    return body_mask


def create_fat_mask(fat_volume, body_mask, k=4.0):
    """
    Create fat mask using threshold: mean_background + k * std_background.

    Parameters
    ----------
    fat_volume : 3D array
    body_mask : 3D boolean array
    k : float, threshold sensitivity (higher = more conservative)

    Returns
    -------
    fat_mask : 3D boolean array
    threshold : float
    """
    # Background = outside body
    background = fat_volume[~body_mask]
    background = background[background >= 0]

    if len(background) == 0:
        all_vals = fat_volume.flatten()
        background = all_vals[all_vals < np.percentile(all_vals, 20)]

    bg_mean = np.mean(background)
    bg_std = np.std(background)
    threshold = bg_mean + k * bg_std

    # Apply threshold within body only
    fat_mask = (fat_volume > threshold) & body_mask

    # Cleanup
    fat_mask = binary_erosion(fat_mask, iterations=1)
    fat_mask = binary_dilation(fat_mask, iterations=1)

    print(f"     Fat threshold: {threshold:.1f} "
          f"(bg_mean={bg_mean:.1f}, bg_std={bg_std:.1f}, k={k})")

    return fat_mask, threshold


def interpolate_contours(polygons, n_slices, image_shape):
    """
    Interpolate abdominal wall contours from annotated slices to all slices.
    Uses radial interpolation (center + radius at fixed angles).

    Parameters
    ----------
    polygons : dict {slice_idx: [(x,y), ...]}
    n_slices : total number of slices
    image_shape : (height, width)

    Returns
    -------
    contour_masks : 3D boolean array (True = inside abdominal cavity)
    """
    h, w = image_shape
    contour_masks = np.zeros((n_slices, h, w), dtype=bool)

    if not polygons:
        contour_masks[:] = True
        return contour_masks

    annotated_slices = sorted(polygons.keys())

    # Create masks for annotated slices
    for sl_idx in annotated_slices:
        poly = polygons[sl_idx]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        rr, cc = ski_polygon(ys, xs, shape=(h, w))
        contour_masks[sl_idx][rr, cc] = True

    if len(annotated_slices) >= 2:
        n_angles = 72  # Every 5 degrees
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

        # Convert each polygon to radial representation
        radial_data = {}
        centers = {}

        for sl_idx in annotated_slices:
            poly = polygons[sl_idx]
            xs = np.array([p[0] for p in poly[:-1]])  # Exclude closing point
            ys = np.array([p[1] for p in poly[:-1]])

            cx, cy = np.mean(xs), np.mean(ys)
            centers[sl_idx] = (cx, cy)

            poly_angles = np.arctan2(ys - cy, xs - cx)
            poly_radii = np.sqrt((xs - cx)**2 + (ys - cy)**2)

            sort_idx = np.argsort(poly_angles)
            poly_angles = poly_angles[sort_idx]
            poly_radii = poly_radii[sort_idx]

            # Handle wraparound for interpolation
            poly_angles_ext = np.concatenate([
                poly_angles - 2*np.pi, poly_angles, poly_angles + 2*np.pi
            ])
            poly_radii_ext = np.concatenate([poly_radii, poly_radii, poly_radii])

            f_interp = interp1d(poly_angles_ext, poly_radii_ext, kind='linear')
            radial_data[sl_idx] = f_interp(angles)

        # Interpolate between annotated slices
        sl_array = np.array(annotated_slices)
        center_x = np.array([centers[s][0] for s in annotated_slices])
        center_y = np.array([centers[s][1] for s in annotated_slices])

        f_cx = interp1d(sl_array, center_x, kind='linear', fill_value='extrapolate')
        f_cy = interp1d(sl_array, center_y, kind='linear', fill_value='extrapolate')

        radii_matrix = np.array([radial_data[s] for s in annotated_slices])

        for sl_idx in range(n_slices):
            if sl_idx in annotated_slices:
                continue

            cx = float(f_cx(sl_idx))
            cy = float(f_cy(sl_idx))

            radii_interp = np.zeros(n_angles)
            for a_idx in range(n_angles):
                f_r = interp1d(sl_array, radii_matrix[:, a_idx],
                              kind='linear', fill_value='extrapolate')
                radii_interp[a_idx] = max(0, float(f_r(sl_idx)))

            xs = cx + radii_interp * np.cos(angles)
            ys = cy + radii_interp * np.sin(angles)

            rr, cc = ski_polygon(ys, xs, shape=(h, w))
            if len(rr) > 0:
                contour_masks[sl_idx][rr, cc] = True

    elif len(annotated_slices) == 1:
        contour_masks[:] = contour_masks[annotated_slices[0]]

    return contour_masks


def separate_sat_vat(fat_mask, contour_masks, body_mask):
    """
    Separate fat into SAT and VAT using abdominal wall contours.
    VAT = fat INSIDE abdominal wall
    SAT = fat OUTSIDE abdominal wall (but inside body)
    """
    vat_mask = fat_mask & contour_masks
    sat_mask = fat_mask & ~contour_masks & body_mask
    return sat_mask, vat_mask


def calculate_volumes(sat_mask, vat_mask, body_mask, voxel_size, 
                      thorax_abdomen_idx=None):
    """
    Calculate tissue volumes from masks.

    Includes:
    - Total volumes (all selected slices)
    - Abdomen volumes (slices 0 to thorax_abdomen_idx - 1)
    - Thorax volumes (slices thorax_abdomen_idx to end)
    - Total analyzed body volume (for normalization)

    Parameters
    ----------
    sat_mask : 3D boolean array
    vat_mask : 3D boolean array
    body_mask : 3D boolean array (outer boundary of mouse)
    voxel_size : tuple (dx, dy, dz) in mm
    thorax_abdomen_idx : int, index within selected slices where thorax starts
                         (relative to cropped volume, not absolute!)

    Returns
    -------
    dict with all volume measurements
    """
    voxel_vol = voxel_size[0] * voxel_size[1] * voxel_size[2]
    n_slices = sat_mask.shape[0]

    # --- TOTAL volumes (all selected slices) ---
    sat_total = float(np.sum(sat_mask)) * voxel_vol
    vat_total = float(np.sum(vat_mask)) * voxel_vol
    fat_total = sat_total + vat_total
    body_total = float(np.sum(body_mask)) * voxel_vol

    ratio_total = vat_total / sat_total if sat_total > 0 else float('inf')

    volumes = {
        # Total
        'SAT_total_mm3': sat_total,
        'VAT_total_mm3': vat_total,
        'Total_fat_mm3': fat_total,
        'SAT_total_cm3': sat_total / 1000,
        'VAT_total_cm3': vat_total / 1000,
        'Total_fat_cm3': fat_total / 1000,
        'VAT_SAT_ratio_total': ratio_total,
        # Total body volume
        'Total_analyzed_volume_mm3': body_total,
        'Total_analyzed_volume_cm3': body_total / 1000,
        # Fat fractions (total)
        'SAT_fraction_total': sat_total / body_total if body_total > 0 else 0,
        'VAT_fraction_total': vat_total / body_total if body_total > 0 else 0,
        'Total_fat_fraction': fat_total / body_total if body_total > 0 else 0,
    }

    # --- REGIONAL volumes (abdomen vs thorax) ---
    if thorax_abdomen_idx is not None and 0 < thorax_abdomen_idx < n_slices:
        # Abdomen: slices 0 to thorax_abdomen_idx - 1 (caudal)
        abd_sat = sat_mask[:thorax_abdomen_idx]
        abd_vat = vat_mask[:thorax_abdomen_idx]
        abd_body = body_mask[:thorax_abdomen_idx]

        sat_abd = float(np.sum(abd_sat)) * voxel_vol
        vat_abd = float(np.sum(abd_vat)) * voxel_vol
        fat_abd = sat_abd + vat_abd
        body_abd = float(np.sum(abd_body)) * voxel_vol
        ratio_abd = vat_abd / sat_abd if sat_abd > 0 else float('inf')

        # Thorax: slices thorax_abdomen_idx to end (cranial)
        tho_sat = sat_mask[thorax_abdomen_idx:]
        tho_vat = vat_mask[thorax_abdomen_idx:]
        tho_body = body_mask[thorax_abdomen_idx:]

        sat_tho = float(np.sum(tho_sat)) * voxel_vol
        vat_tho = float(np.sum(tho_vat)) * voxel_vol
        fat_tho = sat_tho + vat_tho
        body_tho = float(np.sum(tho_body)) * voxel_vol
        ratio_tho = vat_tho / sat_tho if sat_tho > 0 else float('inf')

        volumes.update({
            # Abdomen
            'SAT_abdomen_mm3': sat_abd,
            'VAT_abdomen_mm3': vat_abd,
            'Total_fat_abdomen_mm3': fat_abd,
            'SAT_abdomen_cm3': sat_abd / 1000,
            'VAT_abdomen_cm3': vat_abd / 1000,
            'Total_fat_abdomen_cm3': fat_abd / 1000,
            'VAT_SAT_ratio_abdomen': ratio_abd,
            'Body_volume_abdomen_mm3': body_abd,
            'Body_volume_abdomen_cm3': body_abd / 1000,
            'Fat_fraction_abdomen': fat_abd / body_abd if body_abd > 0 else 0,
            'n_slices_abdomen': thorax_abdomen_idx,
            # Thorax
            'SAT_thorax_mm3': sat_tho,
            'VAT_thorax_mm3': vat_tho,
            'Total_fat_thorax_mm3': fat_tho,
            'SAT_thorax_cm3': sat_tho / 1000,
            'VAT_thorax_cm3': vat_tho / 1000,
            'Total_fat_thorax_cm3': fat_tho / 1000,
            'VAT_SAT_ratio_thorax': ratio_tho,
            'Body_volume_thorax_mm3': body_tho,
            'Body_volume_thorax_cm3': body_tho / 1000,
            'Fat_fraction_thorax': fat_tho / body_tho if body_tho > 0 else 0,
            'n_slices_thorax': n_slices - thorax_abdomen_idx,
        })
    else:
        # No thorax/abdomen separation available
        volumes.update({
            'SAT_abdomen_mm3': np.nan,
            'VAT_abdomen_mm3': np.nan,
            'Total_fat_abdomen_mm3': np.nan,
            'SAT_abdomen_cm3': np.nan,
            'VAT_abdomen_cm3': np.nan,
            'Total_fat_abdomen_cm3': np.nan,
            'VAT_SAT_ratio_abdomen': np.nan,
            'Body_volume_abdomen_mm3': np.nan,
            'Body_volume_abdomen_cm3': np.nan,
            'Fat_fraction_abdomen': np.nan,
            'n_slices_abdomen': np.nan,
            'SAT_thorax_mm3': np.nan,
            'VAT_thorax_mm3': np.nan,
            'Total_fat_thorax_mm3': np.nan,
            'SAT_thorax_cm3': np.nan,
            'VAT_thorax_cm3': np.nan,
            'Total_fat_thorax_cm3': np.nan,
            'VAT_SAT_ratio_thorax': np.nan,
            'Body_volume_thorax_mm3': np.nan,
            'Body_volume_thorax_cm3': np.nan,
            'Fat_fraction_thorax': np.nan,
            'n_slices_thorax': np.nan,
        })

    return volumes


def save_qc_image(fat_volume, sat_mask, vat_mask, body_mask, dam_id, 
                  output_path, start_slice=0, thorax_abdomen_idx=None):
    """Save a QC image showing SAT/VAT overlay on representative slices."""
    n_slices = fat_volume.shape[0]

    if n_slices >= 9:
        indices = np.linspace(0, n_slices-1, 9, dtype=int)
    else:
        indices = np.arange(n_slices)

    n_show = len(indices)
    cols = min(3, n_show)
    rows = int(np.ceil(n_show / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f'QC - Dam {dam_id} | Yellow=SAT, Red=VAT, Blue=Body boundary', 
                 fontsize=14)

    for idx, sl in enumerate(indices):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        fat_sl = fat_volume[sl]
        vmin, vmax = get_contrast_limits(fat_sl)
        ax.imshow(fat_sl, cmap='gray', vmin=vmin, vmax=vmax)

        # SAT/VAT overlay
        overlay = np.zeros((*fat_sl.shape, 4))
        overlay[sat_mask[sl]] = [1, 1, 0, 0.4]   # Yellow = SAT
        overlay[vat_mask[sl]] = [1, 0, 0, 0.4]    # Red = VAT
        ax.imshow(overlay)

        # Region label
        region = ""
        if thorax_abdomen_idx is not None:
            if sl < thorax_abdomen_idx:
                region = " [ABD]"
            else:
                region = " [THO]"

        ax.set_title(f'Slice {start_slice + sl}{region}', fontsize=9)
        ax.axis('off')

    for idx in range(n_show, rows*cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"     QC image saved: {output_path}")
