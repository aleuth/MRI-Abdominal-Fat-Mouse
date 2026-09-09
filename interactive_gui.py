"""
interactive_gui.py - Slice viewer, Start/End selection, Polygon drawing, Review/Correction
============================================================================================
Interactive matplotlib GUIs for the fat segmentation pipeline.
Includes Thorax/Abdomen boundary selection and REVIEW step.

Scan direction: caudal -> cranial
  START = caudal (abdomen begins)
  THORAX/ABDOMEN = boundary (abdomen ends, thorax begins)
  END = cranial (thorax ends)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.interpolate import interp1d
from scipy.ndimage import binary_erosion
from skimage.draw import polygon as ski_polygon


def get_contrast_limits(image):
    """Get good contrast limits using percentiles of non-zero pixels."""
    nonzero = image[image > 0]
    if len(nonzero) > 0:
        vmin, vmax = np.percentile(nonzero, [2, 98])
    else:
        vmin, vmax = 0, 1
    return vmin, vmax


# ---------------------
# SLICE SELECTION GUI
# ---------------------

class SliceSelector:
    """
    Interactive slice viewer to select START, THORAX/ABDOMEN boundary, and END slices.

    Scan direction: caudal -> cranial
      START = first slice (caudal, abdomen)
      THORAX/ABDOMEN = boundary slice (first thorax slice)
      END = last slice (cranial, thorax)

    Abdomen slices: START to THORAX/ABDOMEN - 1
    Thorax slices:  THORAX/ABDOMEN to END
    """

    def __init__(self, fat_volume, water_volume=None):
        self.fat = fat_volume
        self.water = water_volume
        self.n_slices = fat_volume.shape[0]
        self.current_slice = 0
        self.start_slice = None
        self.thorax_abdomen_slice = None
        self.end_slice = None
        self.done = False

    def run(self):
        """Open the interactive viewer. Returns (start, thorax_abdomen, end) slice indices."""
        if self.water is not None:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        else:
            self.fig, self.ax1 = plt.subplots(1, 1, figsize=(7, 6))
            self.ax2 = None

        plt.subplots_adjust(bottom=0.28)
        self._update_display()

        # Slider
        ax_slider = plt.axes([0.15, 0.14, 0.55, 0.03])
        self.slider = Slider(ax_slider, 'Slice', 0, self.n_slices - 1,
                           valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)

        # Buttons
        ax_start = plt.axes([0.05, 0.04, 0.15, 0.05])
        ax_ta = plt.axes([0.22, 0.04, 0.22, 0.05])
        ax_end = plt.axes([0.46, 0.04, 0.15, 0.05])
        ax_done = plt.axes([0.65, 0.04, 0.15, 0.05])

        self.btn_start = Button(ax_start, 'Set START\n(caudal)')
        self.btn_ta = Button(ax_ta, 'Set THORAX/ABDOMEN\nboundary')
        self.btn_end = Button(ax_end, 'Set END\n(cranial)')
        self.btn_done = Button(ax_done, 'DONE')

        self.btn_start.on_clicked(self._set_start)
        self.btn_ta.on_clicked(self._set_thorax_abdomen)
        self.btn_end.on_clicked(self._set_end)
        self.btn_done.on_clicked(self._on_done)

        # Event handlers
        self._cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self._cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.show(block=True)
        return self.start_slice, self.thorax_abdomen_slice, self.end_slice

    def _update_display(self):
        self.ax1.clear()
        fat_sl = self.fat[self.current_slice]
        vmin, vmax = get_contrast_limits(fat_sl)
        self.ax1.imshow(fat_sl, cmap='gray', vmin=vmin, vmax=vmax)

        title = f'FAT - Slice {self.current_slice}/{self.n_slices-1}'
        if self.start_slice is not None:
            title += f'  |  START={self.start_slice}'
        if self.thorax_abdomen_slice is not None:
            title += f'  T/A={self.thorax_abdomen_slice}'
        if self.end_slice is not None:
            title += f'  END={self.end_slice}'

        # Show region info
        if self.start_slice is not None and self.thorax_abdomen_slice is not None and self.end_slice is not None:
            if self.start_slice <= self.current_slice < self.thorax_abdomen_slice:
                title += '  [ABDOMEN]'
            elif self.thorax_abdomen_slice <= self.current_slice <= self.end_slice:
                title += '  [THORAX]'

        self.ax1.set_title(title)
        self.ax1.axis('off')

        if self.ax2 is not None and self.water is not None:
            self.ax2.clear()
            water_sl = self.water[self.current_slice]
            vmin_w, vmax_w = get_contrast_limits(water_sl)
            self.ax2.imshow(water_sl, cmap='gray', vmin=vmin_w, vmax=vmax_w)
            self.ax2.set_title(f'WATER - Slice {self.current_slice}')
            self.ax2.axis('off')

        self.fig.canvas.draw_idle()

    def _on_slider(self, val):
        self.current_slice = int(val)
        self._update_display()

    def _on_scroll(self, event):
        if event.button == 'up':
            self.current_slice = min(self.current_slice + 1, self.n_slices - 1)
        else:
            self.current_slice = max(self.current_slice - 1, 0)
        self.slider.set_val(self.current_slice)

    def _on_key(self, event):
        if event.key in ('right', 'up'):
            self.current_slice = min(self.current_slice + 1, self.n_slices - 1)
            self.slider.set_val(self.current_slice)
        elif event.key in ('left', 'down'):
            self.current_slice = max(self.current_slice - 1, 0)
            self.slider.set_val(self.current_slice)
        elif event.key == 'enter':
            self._on_done(None)

    def _set_start(self, event):
        self.start_slice = self.current_slice
        print(f"  START slice (caudal) set to: {self.start_slice}")
        self._update_display()

    def _set_thorax_abdomen(self, event):
        self.thorax_abdomen_slice = self.current_slice
        print(f"  THORAX/ABDOMEN boundary set to: {self.thorax_abdomen_slice}")
        self._update_display()

    def _set_end(self, event):
        self.end_slice = self.current_slice
        print(f"  END slice (cranial) set to: {self.end_slice}")
        self._update_display()

    def _on_done(self, event):
        if self.start_slice is None or self.end_slice is None:
            print("  Please set at least START and END slices!")
            return
        if self.thorax_abdomen_slice is None:
            print("  Please set THORAX/ABDOMEN boundary!")
            return
        if not (self.start_slice < self.thorax_abdomen_slice <= self.end_slice):
            print("  ERROR: Must be START < THORAX/ABDOMEN <= END!")
            print(f"  Got: START={self.start_slice}, T/A={self.thorax_abdomen_slice}, END={self.end_slice}")
            return
        self.done = True
        plt.close(self.fig)


# ---------------------
# POLYGON DRAWING GUI
# ---------------------

class AbdominalWallDrawer:
    """Interactive tool to draw abdominal wall contour on selected slices."""

    def __init__(self, fat_volume, water_volume, slice_indices):
        self.fat = fat_volume
        self.water = water_volume
        self.slice_indices = slice_indices
        self.polygons = {}
        self.current_points = []

    def run(self):
        """Open interactive drawing window for each slice. Returns dict."""
        for i, sl_idx in enumerate(self.slice_indices):
            print(f"  Draw abdominal wall on slice {sl_idx} "
                  f"({i+1}/{len(self.slice_indices)})")
            print(f"     Left-click: add point | Right-click: close | "
                  f"Middle-click: undo")

            self.current_points = []
            closed = [False]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
            fig.suptitle(
                f'Draw abdominal wall - Slice {sl_idx} '
                f'({i+1}/{len(self.slice_indices)})\n'
                f'LEFT = add point | RIGHT = close polygon | MIDDLE = undo',
                fontsize=10
            )

            fat_sl = self.fat[sl_idx]
            water_sl = self.water[sl_idx]
            vmin_f, vmax_f = get_contrast_limits(fat_sl)
            vmin_w, vmax_w = get_contrast_limits(water_sl)

            ax1.imshow(fat_sl, cmap='gray', vmin=vmin_f, vmax=vmax_f)
            ax1.set_title('FAT image - DRAW HERE')
            ax1.axis('off')

            ax2.imshow(water_sl, cmap='gray', vmin=vmin_w, vmax=vmax_w)
            ax2.set_title('WATER image (muscle = bright = wall!)')
            ax2.axis('off')

            line1, = ax1.plot([], [], 'r.-', linewidth=1.5, markersize=8)
            line2, = ax2.plot([], [], 'r.-', linewidth=1.5, markersize=8)

            points = self.current_points

            def update_lines():
                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    line1.set_data(xs, ys)
                    line2.set_data(xs, ys)
                else:
                    line1.set_data([], [])
                    line2.set_data([], [])

            def on_click(event, _sl=sl_idx, _fig=fig):
                if event.inaxes not in [ax1, ax2]:
                    return
                if closed[0]:
                    return

                if event.button == 1:
                    points.append((event.xdata, event.ydata))
                    update_lines()
                    _fig.canvas.draw_idle()
                elif event.button == 2:
                    if points:
                        points.pop()
                        update_lines()
                        _fig.canvas.draw_idle()
                elif event.button == 3:
                    if len(points) >= 3:
                        points.append(points[0])
                        update_lines()
                        self.polygons[_sl] = list(points)
                        closed[0] = True
                        print(f"     Polygon closed with {len(points)-1} points")
                        plt.close(_fig)
                    else:
                        print("     Need at least 3 points!")

            _cid = fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show(block=True)

            if sl_idx not in self.polygons:
                print(f"     No polygon drawn for slice {sl_idx}, skipping")

        return self.polygons


# ---------------------
# REVIEW & CORRECTION GUI
# ---------------------

class SegmentationReviewer:
    """
    Review SAT/VAT segmentation on ALL slices after interpolation.
    Allows manual correction of the abdominal wall on specific slices.

    Controls:
        Mouse wheel / Slider / Arrow keys : scroll slices
        'C' key or button                 : mark/unmark slice for correction
        'Enter' or DONE button            : finish review
    """

    def __init__(self, fat_volume, water_volume, sat_mask, vat_mask,
                 body_mask, contour_masks, fat_mask):
        self.fat = fat_volume
        self.water = water_volume
        self.sat_mask = sat_mask.copy()
        self.vat_mask = vat_mask.copy()
        self.body_mask = body_mask
        self.contour_masks = contour_masks.copy()
        self.fat_mask = fat_mask
        self.n_slices = fat_volume.shape[0]
        self.current_slice = 0
        self.slices_to_correct = []
        self.corrections = {}

    def run(self):
        """
        Open review window.

        Returns
        -------
        corrected : bool
        contour_masks : updated 3D array
        sat_mask : updated 3D array
        vat_mask : updated 3D array
        """
        print("\n  REVIEW: Scroll through all slices.")
        print("     Press 'C' to mark current slice for CORRECTION.")
        print("     Press 'ENTER' or click DONE when finished reviewing.")
        print("     Yellow = SAT, Red = VAT")

        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 5.5))
        plt.subplots_adjust(bottom=0.22)

        ax_slider = plt.axes([0.15, 0.10, 0.55, 0.03])
        self.slider = Slider(ax_slider, 'Slice', 0, self.n_slices - 1,
                           valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)

        ax_correct = plt.axes([0.15, 0.02, 0.18, 0.05])
        ax_done = plt.axes([0.55, 0.02, 0.18, 0.05])
        ax_info = plt.axes([0.35, 0.02, 0.18, 0.05])

        self.btn_correct = Button(ax_correct, 'Mark for Correction [C]')
        self.btn_done = Button(ax_done, 'DONE Reviewing')
        self.btn_info = Button(ax_info, 'Marked: 0 slices')

        self.btn_correct.on_clicked(self._mark_correction)
        self.btn_done.on_clicked(self._on_done)

        self._cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self._cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._update_display()
        plt.show(block=True)

        # If slices marked for correction, open correction tool
        if self.slices_to_correct:
            print(f"\n  Correcting {len(self.slices_to_correct)} slices...")
            self._run_corrections()
            return True, self.contour_masks, self.sat_mask, self.vat_mask
        else:
            print("  No corrections needed!")
            return False, self.contour_masks, self.sat_mask, self.vat_mask

    def _update_display(self):
        # Left: Fat image with SAT/VAT overlay
        self.axes[0].clear()
        fat_slice = self.fat[self.current_slice]
        vmin, vmax = get_contrast_limits(fat_slice)
        self.axes[0].imshow(fat_slice, cmap='gray', vmin=vmin, vmax=vmax)

        overlay = np.zeros((*fat_slice.shape, 4))
        sat_sl = self.sat_mask[self.current_slice]
        vat_sl = self.vat_mask[self.current_slice]
        overlay[sat_sl] = [1, 1, 0, 0.35]   # Yellow
        overlay[vat_sl] = [1, 0, 0, 0.35]   # Red
        self.axes[0].imshow(overlay)

        marker = " ** MARKED **" if self.current_slice in self.slices_to_correct else ""
        self.axes[0].set_title(f'FAT + Overlay - Slice {self.current_slice}{marker}')
        self.axes[0].axis('off')

        # Middle: Water image
        self.axes[1].clear()
        water_sl = self.water[self.current_slice]
        vmin_w, vmax_w = get_contrast_limits(water_sl)
        self.axes[1].imshow(water_sl, cmap='gray', vmin=vmin_w, vmax=vmax_w)
        self.axes[1].set_title(f'WATER - Slice {self.current_slice}')
        self.axes[1].axis('off')

        # Right: Contour boundary visualization
        self.axes[2].clear()
        self.axes[2].imshow(fat_slice, cmap='gray', vmin=vmin, vmax=vmax)
        contour_sl = self.contour_masks[self.current_slice]
        boundary = contour_sl ^ binary_erosion(contour_sl, iterations=2)
        boundary_overlay = np.zeros((*fat_slice.shape, 4))
        boundary_overlay[boundary] = [0, 1, 0, 0.8]  # Green boundary
        self.axes[2].imshow(boundary_overlay)
        self.axes[2].set_title(f'Abdominal Wall (green)')
        self.axes[2].axis('off')

        self.fig.canvas.draw_idle()

    def _on_slider(self, val):
        self.current_slice = int(val)
        self._update_display()

    def _on_scroll(self, event):
        if event.button == 'up':
            self.current_slice = min(self.current_slice + 1, self.n_slices - 1)
        else:
            self.current_slice = max(self.current_slice - 1, 0)
        self.slider.set_val(self.current_slice)

    def _on_key(self, event):
        if event.key in ('right', 'up'):
            self.current_slice = min(self.current_slice + 1, self.n_slices - 1)
            self.slider.set_val(self.current_slice)
        elif event.key in ('left', 'down'):
            self.current_slice = max(self.current_slice - 1, 0)
            self.slider.set_val(self.current_slice)
        elif event.key in ('c', 'C'):
            self._mark_correction(None)
        elif event.key == 'enter':
            self._on_done(None)

    def _mark_correction(self, event):
        sl = self.current_slice
        if sl in self.slices_to_correct:
            self.slices_to_correct.remove(sl)
            print(f"     Slice {sl} UNMARKED")
        else:
            self.slices_to_correct.append(sl)
            print(f"     Slice {sl} marked for correction")
        self.slices_to_correct.sort()
        self.btn_info.label.set_text(f'Marked: {len(self.slices_to_correct)} slices')
        self._update_display()

    def _on_done(self, event):
        plt.close(self.fig)

    def _run_corrections(self):
        """Open polygon drawing tool for each marked slice."""
        for i, sl_idx in enumerate(self.slices_to_correct):
            print(f"\n  Correcting slice {sl_idx} "
                  f"({i+1}/{len(self.slices_to_correct)})")
            print(f"     Redraw the abdominal wall contour.")

            points = []
            closed = [False]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
            fig.suptitle(
                f'CORRECT abdominal wall - Slice {sl_idx} '
                f'({i+1}/{len(self.slices_to_correct)})\n'
                f'LEFT = add point | RIGHT = close | MIDDLE = undo',
                fontsize=10
            )

            # Show fat with current overlay
            fat_sl = self.fat[sl_idx]
            vmin_f, vmax_f = get_contrast_limits(fat_sl)
            ax1.imshow(fat_sl, cmap='gray', vmin=vmin_f, vmax=vmax_f)
            overlay = np.zeros((*fat_sl.shape, 4))
            overlay[self.sat_mask[sl_idx]] = [1, 1, 0, 0.2]
            overlay[self.vat_mask[sl_idx]] = [1, 0, 0, 0.2]
            ax1.imshow(overlay)
            ax1.set_title('FAT + current overlay (to correct)')
            ax1.axis('off')

            water_sl = self.water[sl_idx]
            vmin_w, vmax_w = get_contrast_limits(water_sl)
            ax2.imshow(water_sl, cmap='gray', vmin=vmin_w, vmax=vmax_w)
            ax2.set_title('WATER (muscle = bright = wall)')
            ax2.axis('off')

            line1, = ax1.plot([], [], 'g.-', linewidth=2, markersize=10)
            line2, = ax2.plot([], [], 'g.-', linewidth=2, markersize=10)

            def update_lines():
                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    line1.set_data(xs, ys)
                    line2.set_data(xs, ys)
                else:
                    line1.set_data([], [])
                    line2.set_data([], [])

            def on_click(event, _fig=fig):
                if event.inaxes not in [ax1, ax2]:
                    return
                if closed[0]:
                    return

                if event.button == 1:
                    points.append((event.xdata, event.ydata))
                    update_lines()
                    _fig.canvas.draw_idle()
                elif event.button == 2:
                    if points:
                        points.pop()
                        update_lines()
                        _fig.canvas.draw_idle()
                elif event.button == 3:
                    if len(points) >= 3:
                        points.append(points[0])
                        update_lines()
                        closed[0] = True
                        print(f"     New polygon with {len(points)-1} points")
                        _fig.canvas.draw_idle()
                        plt.close(_fig)
                    else:
                        print("     Need at least 3 points!")

            _cid = fig.canvas.mpl_connect('button_press_event', on_click)
            plt.show(block=True)

            # Apply correction
            if closed[0] and len(points) >= 4:
                self.corrections[sl_idx] = points
                self._apply_single_correction(sl_idx, points)

    def _apply_single_correction(self, sl_idx, polygon_points):
        """Apply a new polygon contour to a single slice."""
        h, w = self.fat[sl_idx].shape

        xs = [p[0] for p in polygon_points]
        ys = [p[1] for p in polygon_points]
        rr, cc = ski_polygon(ys, xs, shape=(h, w))

        # Update contour mask
        new_contour = np.zeros((h, w), dtype=bool)
        new_contour[rr, cc] = True
        self.contour_masks[sl_idx] = new_contour

        # Recalculate SAT/VAT for this slice
        fat_sl = self.fat_mask[sl_idx]
        self.vat_mask[sl_idx] = fat_sl & new_contour
        self.sat_mask[sl_idx] = fat_sl & ~new_contour & self.body_mask[sl_idx]

        print(f"     Slice {sl_idx} corrected")
