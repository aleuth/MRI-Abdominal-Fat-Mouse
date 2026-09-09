# MRI-Abdominal-Fat-Mouse
Mouse Abdominal Fat Segmentation Pipeline from Bruker BioSpin 7T MRI Scans to use in Spyder

Scans used: Multi-echo gradient-echo FLASH sequence with Dixon-based fat/water separation was used to generate in-phase and opposed-phase images, from which separate fat and water images were reconstructed by the scanner software.
Bruker-format binary image data (2dseq) were read directly using the metadata from the associated visu_pars parameter files. 

-------------------------------------------------
Features: Thorax/Abdomen separation + Total Body Volume + Review/Correction

Scan direction: caudal -> cranial
  - START = first analyzed slice (caudal, abdomen begins)
  - THORAX/ABDOMEN = boundary (first thorax slice)
  - END = last analyzed slice (cranial, thorax ends)

SETUP
-----
1. Anaconda Prompt:
     conda create -n fat_seg python=3.10
     conda activate fat_seg
     pip install numpy matplotlib scipy scikit-image pandas openpyxl spyder pyqt5
     spyder

2. In Spyder:
     Tools -> Preferences -> IPython console -> Graphics -> Backend: "Qt5"
     Consoles -> Restart kernel

3. Put all 4 .py files in one folder

4. Open main.py, edit CONFIGURATION section (3 paths to data and for Output folder!)

5. Press F5

WORKFLOW PER MOUSE (~5-8 min)
-----------------------------
Step 1:  Load Bruker data, split Fat/Water                    [AUTO]
Step 2:  Select START, THORAX/ABDOMEN boundary, END           [INTERACTIVE]
Step 3:  Create body mask                                     [AUTO]
Step 4:  Calculate adaptive fat threshold                     [AUTO]
Step 5:  Draw abdominal wall on 5 slices                      [INTERACTIVE]
Step 6:  Interpolate wall to all slices                       [AUTO]
Step 7:  Separate SAT/VAT                                     [AUTO]
Step 8:  Review overlay, mark corrections                     [INTERACTIVE]
Step 9:  Calculate volumes (total, thorax, abdomen, body)     [AUTO]
Step 10: Save QC image                                        [AUTO]

CONTROLS
--------
Slice Selection:
  Mouse wheel / Slider / Arrows : scroll
  Buttons: Set START (caudal) | Set THORAX/ABDOMEN | Set END (cranial) | DONE

Draw Abdominal Wall:
  Left-click: add point | Middle-click: undo | Right-click: close polygon

Review:
  Scroll: browse | 'C': mark for correction | Enter: done reviewing

OUTPUT COLUMNS (CSV)
--------------------
Total:     SAT_total, VAT_total, Total_fat, VAT_SAT_ratio_total
           (all in mm3 AND cm3)
Body:      Total_analyzed_volume (mm3, cm3)
Fractions: SAT_fraction_total, VAT_fraction_total, Total_fat_fraction
Abdomen:   SAT_abdomen, VAT_abdomen, Total_fat_abdomen, Body_volume_abdomen
           VAT_SAT_ratio_abdomen, Fat_fraction_abdomen, n_slices_abdomen
Thorax:    SAT_thorax, VAT_thorax, Total_fat_thorax, Body_volume_thorax
           VAT_SAT_ratio_thorax, Fat_fraction_thorax, n_slices_thorax
Metadata:  Dam_ID, start_slice, thorax_abdomen_slice, end_slice,
           n_slices_total, threshold, corrected

TUNING
------
FAT_THRESHOLD_K = 3.0  (higher=stricter, lower=more sensitive)
N_ANNOTATION_SLICES = 5 (more=better boundary, slower)

TIPS
----
- Test ONE mouse first (TEST_SINGLE_DAM_ID = 1284)
- If Fat/Water look swapped: in load_data.py, swap lines in split_fat_water():
    fat = data[n_slices:]
    water = data[:n_slices]
- Total_analyzed_volume = outer body boundary (for normalization)
- INTERIM csv saves after each mouse (crash protection)
- QC images show 9 slices with SAT(yellow)/VAT(red) overlay + region label
- Abdomen = caudal slices (START to THORAX/ABDOMEN-1)
- Thorax = cranial slices (THORAX/ABDOMEN to END)
