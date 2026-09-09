"""
main.py - Mouse Abdominal Fat Segmentation Pipeline (SPYDER VERSION)
=====================================================================
Open this file in Spyder and press F5 to run.

Features:
- Thorax/Abdomen separation (scan direction: caudal -> cranial)
- Total analyzed body volume (for normalization)
- SAT/VAT per region
- Review & Correction step

CRITICAL: Set matplotlib backend to Qt5 first!
    Tools -> Preferences -> IPython console -> Graphics -> Backend: Qt5
    Then restart the kernel.
    OR type in console: %matplotlib qt
"""

import sys
import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION - EDIT THESE PATHS!
# ============================================================

DATA_ROOT = r'C:\'           # Folder with all MRI scans
EXCEL_PATH = r'C:\'     # Your Excel file containing the file names
OUTPUT_DIR = r'C:\'              # Results folder

# Parameters
FAT_THRESHOLD_K = 3.0          # Fat threshold sensitivity (3-5, higher=stricter)
N_ANNOTATION_SLICES = 5        # Number of slices to draw abdominal wall on

# Testing: Set to a single Mouse ID (e.g. 1284) to test one mouse,
# or None to process ALL mice from Excel
TEST_SINGLE_DAM_ID = None     # Change to None for batch processing

# ============================================================
# SETUP
# ============================================================

# Add script directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Set interactive backend
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
except Exception:
    try:
        matplotlib.use('TkAgg')
    except Exception:
        print("WARNING: Could not set interactive backend.")
        print("Please run: %matplotlib qt")

import matplotlib.pyplot as plt

# Import pipeline modules
from load_data import (find_mouse_folder, read_bruker_2dseq, split_fat_water,
                       get_voxel_dimensions, load_excel_list)
from interactive_gui import (SliceSelector, AbdominalWallDrawer,
                            SegmentationReviewer)
from segmentation import (create_body_mask, create_fat_mask,
                          interpolate_contours, separate_sat_vat,
                          calculate_volumes, save_qc_image)

# ============================================================
# MAIN PIPELINE
# ============================================================

def process_single_mouse(dam_id, scan_nr, data_root, output_dir,
                         fat_threshold_k=4.0, n_annotation_slices=5):
    """
    Complete pipeline for a single mouse.
    Returns volumes dict or None if failed.
    """
    print(f"\n{'='*60}")
    print(f"  Processing Dam {dam_id} (Scan {scan_nr})")
    print(f"{'='*60}")

    # --- Step 1: Load data ---
    print("\n  Step 1: Loading Bruker data...")
    try:
        mouse_folder = find_mouse_folder(data_root, dam_id)
        scan_path = os.path.join(mouse_folder, str(scan_nr))

        if not os.path.exists(scan_path):
            print(f"  ERROR: Scan folder not found: {scan_path}")
            return None

        # Load Fat+Water (pdata/1)
        print(f"     Loading Fat+Water from pdata/1...")
        fw_data, fw_params = read_bruker_2dseq(scan_path, pdata_id=1)
        fat_volume, water_volume = split_fat_water(fw_data)
        print(f"     Fat: {fat_volume.shape}, Water: {water_volume.shape}")

        # Load Combined (pdata/2)
        print(f"     Loading Combined from pdata/2...")
        combined_volume, comb_params = read_bruker_2dseq(scan_path, pdata_id=2)
        print(f"     Combined: {combined_volume.shape}")

        # Voxel dimensions
        voxel_size = get_voxel_dimensions(fw_params)
        print(f"     Voxel size: {voxel_size[0]:.3f} x {voxel_size[1]:.3f} x "
              f"{voxel_size[2]:.3f} mm")

    except Exception as e:
        print(f"  ERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- Step 2: Select START, THORAX/ABDOMEN, END slices ---
    print("\n  Step 2: Select START (caudal), THORAX/ABDOMEN boundary, END (cranial)...")
    print("     Scroll with mouse wheel, click buttons, then DONE.")
    print("     Scan direction: caudal (abdomen) -> cranial (thorax)")

    selector = SliceSelector(fat_volume, water_volume)
    start_sl, thorax_abdomen_sl, end_sl = selector.run()

    if start_sl is None or end_sl is None:
        print("  ERROR: No slice range selected!")
        return None
    if thorax_abdomen_sl is None:
        print("  WARNING: No thorax/abdomen boundary set!")
        print("  All slices will be treated as one region.")

    n_total_sel = end_sl - start_sl + 1
    print(f"     Selected: slice {start_sl} to {end_sl} ({n_total_sel} slices)")
    if thorax_abdomen_sl is not None:
        n_abd = thorax_abdomen_sl - start_sl
        n_tho = end_sl - thorax_abdomen_sl + 1
        print(f"     Abdomen: slices {start_sl}-{thorax_abdomen_sl-1} ({n_abd} slices)")
        print(f"     Thorax:  slices {thorax_abdomen_sl}-{end_sl} ({n_tho} slices)")

    # Crop to selected range
    fat_sel = fat_volume[start_sl:end_sl+1]
    water_sel = water_volume[start_sl:end_sl+1]
    combined_sel = combined_volume[start_sl:end_sl+1]
    n_sel = fat_sel.shape[0]

    # Calculate relative thorax/abdomen index within cropped volume
    if thorax_abdomen_sl is not None:
        thorax_abdomen_idx = thorax_abdomen_sl - start_sl  # relative index
    else:
        thorax_abdomen_idx = None

    # --- Step 3: Body mask ---
    print("\n  Step 3: Creating body mask...")
    body_mask = create_body_mask(combined_sel)
    body_voxels = np.sum(body_mask)
    print(f"     Body mask created ({body_voxels} voxels)")

    # --- Step 4: Fat segmentation ---
    print("\n  Step 4: Fat thresholding...")
    fat_mask, threshold = create_fat_mask(fat_sel, body_mask, k=fat_threshold_k)
    n_fat_voxels = np.sum(fat_mask)
    print(f"     {n_fat_voxels} fat voxels detected")

    # --- Step 5: Draw abdominal wall ---
    print(f"\n  Step 5: Draw abdominal wall on {n_annotation_slices} slices...")

    annotation_indices = np.linspace(0, n_sel-1, n_annotation_slices, dtype=int)
    annotation_indices = list(np.unique(annotation_indices))

    drawer = AbdominalWallDrawer(fat_sel, water_sel, annotation_indices)
    polygons = drawer.run()

    if not polygons:
        print("  WARNING: No contours drawn! Treating all fat as VAT.")

    # --- Step 6: Interpolate contours ---
    print("\n  Step 6: Interpolating contours to all slices...")
    contour_masks = interpolate_contours(polygons, n_sel, fat_sel.shape[1:])
    print(f"     Contours interpolated")

    # --- Step 7: Separate SAT/VAT ---
    print("\n  Step 7: Separating SAT and VAT...")
    sat_mask, vat_mask = separate_sat_vat(fat_mask, contour_masks, body_mask)
    print(f"     SAT voxels: {np.sum(sat_mask)}")
    print(f"     VAT voxels: {np.sum(vat_mask)}")

    # --- Step 8: REVIEW & CORRECTION ---
    print("\n  Step 8: Review segmentation (scroll all slices, C=mark, Enter=done)...")

    reviewer = SegmentationReviewer(
        fat_sel, water_sel, sat_mask, vat_mask,
        body_mask, contour_masks, fat_mask
    )
    corrected, contour_masks, sat_mask, vat_mask = reviewer.run()

    if corrected:
        print(f"     Corrections applied!")
        print(f"     Updated SAT: {np.sum(sat_mask)} voxels")
        print(f"     Updated VAT: {np.sum(vat_mask)} voxels")

    # --- Step 9: Calculate volumes ---
    print("\n  Step 9: Calculating volumes...")
    volumes = calculate_volumes(
        sat_mask, vat_mask, body_mask, voxel_size,
        thorax_abdomen_idx=thorax_abdomen_idx
    )

    # Add metadata
    volumes['Dam_ID'] = dam_id
    volumes['start_slice'] = start_sl
    volumes['thorax_abdomen_slice'] = thorax_abdomen_sl if thorax_abdomen_sl is not None else np.nan
    volumes['end_slice'] = end_sl
    volumes['n_slices_total'] = n_sel
    volumes['threshold'] = threshold
    volumes['corrected'] = corrected

    # Print results
    print(f"\n  {'='*50}")
    print(f"  RESULTS for Dam {dam_id}:")
    print(f"  {'='*50}")
    print(f"  --- TOTAL ---")
    print(f"     {'SAT:':<25} {volumes['SAT_total_mm3']:>10.1f} mm3 "
          f"({volumes['SAT_total_cm3']:.4f} cm3)")
    print(f"     {'VAT:':<25} {volumes['VAT_total_mm3']:>10.1f} mm3 "
          f"({volumes['VAT_total_cm3']:.4f} cm3)")
    print(f"     {'Total fat:':<25} {volumes['Total_fat_mm3']:>10.1f} mm3 "
          f"({volumes['Total_fat_cm3']:.4f} cm3)")
    print(f"     {'VAT/SAT ratio:':<25} {volumes['VAT_SAT_ratio_total']:>10.3f}")
    print(f"     {'Total body volume:':<25} {volumes['Total_analyzed_volume_mm3']:>10.1f} mm3 "
          f"({volumes['Total_analyzed_volume_cm3']:.4f} cm3)")
    print(f"     {'Total fat fraction:':<25} {volumes['Total_fat_fraction']*100:>10.1f} %")

    if thorax_abdomen_idx is not None:
        print(f"  --- ABDOMEN (caudal, {volumes.get('n_slices_abdomen', '?')} slices) ---")
        print(f"     {'SAT abdomen:':<25} {volumes['SAT_abdomen_mm3']:>10.1f} mm3")
        print(f"     {'VAT abdomen:':<25} {volumes['VAT_abdomen_mm3']:>10.1f} mm3")
        print(f"     {'Total fat abdomen:':<25} {volumes['Total_fat_abdomen_mm3']:>10.1f} mm3")
        print(f"     {'Body vol abdomen:':<25} {volumes['Body_volume_abdomen_mm3']:>10.1f} mm3")
        print(f"     {'Fat fraction abdomen:':<25} {volumes['Fat_fraction_abdomen']*100:>10.1f} %")
        print(f"  --- THORAX (cranial, {volumes.get('n_slices_thorax', '?')} slices) ---")
        print(f"     {'SAT thorax:':<25} {volumes['SAT_thorax_mm3']:>10.1f} mm3")
        print(f"     {'VAT thorax:':<25} {volumes['VAT_thorax_mm3']:>10.1f} mm3")
        print(f"     {'Total fat thorax:':<25} {volumes['Total_fat_thorax_mm3']:>10.1f} mm3")
        print(f"     {'Body vol thorax:':<25} {volumes['Body_volume_thorax_mm3']:>10.1f} mm3")
        print(f"     {'Fat fraction thorax:':<25} {volumes['Fat_fraction_thorax']*100:>10.1f} %")

    # --- Step 10: Save QC image ---
    print("\n  Step 10: Saving QC image...")
    qc_path = os.path.join(output_dir, f"QC_Dam_{dam_id}.png")
    save_qc_image(fat_sel, sat_mask, vat_mask, body_mask, dam_id, qc_path,
                  start_slice=start_sl, thorax_abdomen_idx=thorax_abdomen_idx)

    return volumes


# ============================================================
# RUN
# ============================================================

if __name__ == '__main__':

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Excel
    print("Loading mouse list from Excel...")
    mouse_list = load_excel_list(EXCEL_PATH)
    print(f"   Found {len(mouse_list)} mice in list")

    # Filter for testing
    if TEST_SINGLE_DAM_ID is not None:
        mouse_list = mouse_list[mouse_list['Dam ID'] == TEST_SINGLE_DAM_ID]
        if len(mouse_list) == 0:
            print(f"ERROR: Dam ID {TEST_SINGLE_DAM_ID} not found in Excel!")
            sys.exit(1)
        print(f"   TEST MODE: Processing only Dam {TEST_SINGLE_DAM_ID}")

    # Process mice
    all_results = []

    for idx, row in mouse_list.iterrows():
        dam_id = int(row['Dam ID'])
        scan_nr = int(row['scan'])

        volumes = process_single_mouse(
            dam_id=dam_id,
            scan_nr=scan_nr,
            data_root=DATA_ROOT,
            output_dir=OUTPUT_DIR,
            fat_threshold_k=FAT_THRESHOLD_K,
            n_annotation_slices=N_ANNOTATION_SLICES
        )

        if volumes is not None:
            all_results.append(volumes)

            # Interim save (crash protection)
            interim_df = pd.DataFrame(all_results)
            interim_path = os.path.join(OUTPUT_DIR, 'fat_volumes_INTERIM.csv')
            interim_df.to_csv(interim_path, index=False)

    # Final save
    if all_results:
        final_df = pd.DataFrame(all_results)
        
        # --- BACKUP: timestamped copy (never overwritten!) ---
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(OUTPUT_DIR, f'fat_volumes_{timestamp}.csv')
        final_df.to_csv(backup_path, index=False)
        print(f"  Backup saved: {backup_path}")

        # Define column order
        col_order = [
            'Dam_ID',
            # Total
            'SAT_total_mm3', 'VAT_total_mm3', 'Total_fat_mm3',
            'SAT_total_cm3', 'VAT_total_cm3', 'Total_fat_cm3',
            'VAT_SAT_ratio_total',
            'Total_analyzed_volume_mm3', 'Total_analyzed_volume_cm3',
            'SAT_fraction_total', 'VAT_fraction_total', 'Total_fat_fraction',
            # Abdomen
            'SAT_abdomen_mm3', 'VAT_abdomen_mm3', 'Total_fat_abdomen_mm3',
            'SAT_abdomen_cm3', 'VAT_abdomen_cm3', 'Total_fat_abdomen_cm3',
            'VAT_SAT_ratio_abdomen',
            'Body_volume_abdomen_mm3', 'Body_volume_abdomen_cm3',
            'Fat_fraction_abdomen', 'n_slices_abdomen',
            # Thorax
            'SAT_thorax_mm3', 'VAT_thorax_mm3', 'Total_fat_thorax_mm3',
            'SAT_thorax_cm3', 'VAT_thorax_cm3', 'Total_fat_thorax_cm3',
            'VAT_SAT_ratio_thorax',
            'Body_volume_thorax_mm3', 'Body_volume_thorax_cm3',
            'Fat_fraction_thorax', 'n_slices_thorax',
            # Metadata
            'start_slice', 'thorax_abdomen_slice', 'end_slice',
            'n_slices_total', 'threshold', 'corrected',
        ]
        final_df = final_df[[c for c in col_order if c in final_df.columns]]

        final_path = os.path.join(OUTPUT_DIR, 'fat_volumes_all_mice.csv')
        final_df.to_csv(final_path, index=False)

        print(f"\n\n{'='*60}")
        print(f"  ALL DONE! {len(all_results)} mice processed.")
        print(f"  Results saved to: {final_path}")
        print(f"{'='*60}")
        print(f"\n{final_df.to_string()}")
    else:
        print("\n  No results generated!")
