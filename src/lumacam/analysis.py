import os
import subprocess
from pathlib import Path
from tqdm.notebook import tqdm
from enum import IntEnum
from dataclasses import dataclass
import json
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import tifffile
from roifile import ImagejRoi, ROI_TYPE
from lmfit.models import Model
from scipy.special import erfc
from matplotlib import pyplot as plt
import glob

class VerbosityLevel(IntEnum):
    """Verbosity levels for simulation output."""
    QUIET = 0    # Show nothing except progress bar
    BASIC = 1    # Show progress bar and basic info
    DETAILED = 2 # Show everything


class Analysis:
    def __init__(self, archive: str = "test", empir_dirpath: str = None):
        """
        Analysis class for processing TPX3 data through the EMPIR pipeline.

        Args:
            archive: The directory containing TPX3 data or a groupby subfolder.
            empir_dirpath: Path to the EMPIR directory. If None, defaults to "./empir".
        """
        self.archive = Path(archive)
        
        # Check if this is a groupby folder
        self._is_groupby = False
        self._groupby_metadata = None
        self._groupby_subfolders = []
        
        # Look for groupby metadata in the archive or parent
        metadata_file = self.archive / ".groupby_metadata.json"
        if metadata_file.exists():
            self._is_groupby = True
            with open(metadata_file, 'r') as f:
                self._groupby_metadata = json.load(f)
            
            # Find all subfolders with SimPhotons or tpx3Files
            for subfolder in sorted(self.archive.iterdir()):
                if subfolder.is_dir() and not subfolder.name.startswith('.'):
                    if (subfolder / "SimPhotons").exists() or (subfolder / "tpx3Files").exists():
                        self._groupby_subfolders.append(subfolder)
            
            print(f"Detected groupby structure: {len(self._groupby_subfolders)} groups")
            print(f"Groupby column: {self._groupby_metadata.get('column', 'unknown')}")
        
        if empir_dirpath is not None:
            self.empir_dirpath = Path(empir_dirpath)
        else:
            try:
                from G4LumaCam.config.paths import EMPIR_PATH
                self.empir_dirpath = Path(EMPIR_PATH)
            except ImportError:
                self.empir_dirpath = Path("./empir")
        
        # Set photon_files_dir only for non-groupby structures
        if not self._is_groupby:
            self.photon_files_dir = self.archive / "photonFiles"
            self.photon_files_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.photon_files_dir = None

        if not self.empir_dirpath.exists():
            raise FileNotFoundError(f"{self.empir_dirpath} does not exist.")
        
        required_files = {
            "empir_export_events": "empir_export_events",
            "empir_export_photons": "empir_export_photons",
            "empir_pixel2photon_tpx3spidr": "bin/empir_pixel2photon_tpx3spidr",
            "empir_photon2event": "bin/empir_photon2event",
            "empir_event2image": "bin/empir_event2image"
        }
        
        self.executables = {}
        for attr_name, filename in required_files.items():
            file_path = self.empir_dirpath / filename
            if not file_path.exists():
                raise FileNotFoundError(f"{filename} not found in {self.empir_dirpath}")
            self.executables[attr_name] = file_path
            setattr(self, attr_name, file_path)

        # Load parameters from config/empir_params.py
        try:
            from .config.empir_params import DEFAULT_PARAMS
            self.default_params = DEFAULT_PARAMS
        except ImportError:
            raise ImportError(
                "Could not import DEFAULT_PARAMS from G4LumaCam.config.empir_params. "
                "Please ensure empir_params.py exists in the config directory."
            )


    def _run_roi_analysis(self, 
                    tiff_path: str, 
                    roi_zip_path: str, 
                    output_dir: str,
                    verbosity: VerbosityLevel = VerbosityLevel.BASIC,
                    pixel_size_um: float = 120.0,
                    detector_pixels: int = 512) -> None:
        """
        Reads a TIFF stack and applies rectangular ROIs from a .zip file, calculates counts
        (sum of pixel values) per slice per ROI, and saves CSV files with columns: stack
        (1-based), counts, err (sqrt(counts)) in the specified output directory.
        
        Also generates a summary statistics CSV with total counts and statistics for the 
        entire TIFF and each ROI.
        
        If ROI name contains "mtf", performs MTF analysis on the summed image within that ROI.

        Args:
            tiff_path: Absolute path to the TIFF stack file.
            roi_zip_path: Absolute path to the roi.zip file.
            output_dir: Absolute path to the output directory for CSV files.
            verbosity: Controls the level of output during processing.
            pixel_size_um: Pixel size in micrometers (default: 120.0).
            detector_pixels: Number of detector pixels for frequency conversion (default: 512).
        """
        tiff_path = Path(tiff_path)
        roi_zip_path = Path(roi_zip_path)
        output_dir = Path(output_dir)

        if not tiff_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {tiff_path}")
        if not roi_zip_path.exists():
            raise FileNotFoundError(f"ROI zip file not found: {roi_zip_path}")

        # Read TIFF stack
        stack = tifffile.imread(tiff_path)
        if len(stack.shape) != 3:
            raise ValueError(f"Expected 3D TIFF stack (slices, height, width), got shape {stack.shape}")
        slices, height, width = stack.shape

        # Check if TIFF stack has non-zero values
        if np.all(stack == 0):
            if verbosity >= VerbosityLevel.BASIC:
                print(f"Warning: TIFF stack at {tiff_path} contains all zero values")
            return

        # Read ROIs - fromfile returns a list directly, not a context manager
        rois = ImagejRoi.fromfile(roi_zip_path)
        
        # Ensure rois is a list
        if isinstance(rois, ImagejRoi):
            rois = [rois]
        
        rect_rois = [roi for roi in rois if roi.roitype == ROI_TYPE.RECT]
        n_rois = len(rect_rois)
        if n_rois == 0:
            raise ValueError(f"No rectangular ROIs found in {roi_zip_path}")

        if verbosity >= VerbosityLevel.BASIC:
            print(f"Loaded TIFF stack: {slices} slices, {height}x{width}")
            print(f"Processing {n_rois} rectangular ROIs from {roi_zip_path}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Output directory: {output_dir}")

        # Create MTF output directory at base_dir level
        base_dir = output_dir.parent
        mtf_output_dir = base_dir / "MTF_calculation"
        
        # Compute summed image for MTF analysis (only if needed)
        mtf_rois = [roi for roi in rect_rois if "mtf" in (roi.name or "").lower()]
        if mtf_rois:
            summed_image = np.sum(stack, axis=0).astype(np.float64)
            # Replace inf and nan with nan for proper handling
            summed_image = np.where(np.isinf(summed_image), np.nan, summed_image)
            mtf_output_dir.mkdir(parents=True, exist_ok=True)
            if verbosity >= VerbosityLevel.BASIC:
                print(f"MTF output directory: {mtf_output_dir}")

        # Initialize summary statistics list
        summary_stats = []
        
        # Calculate statistics for entire TIFF stack
        stack_float = stack.astype(np.float64)
        total_counts = np.sum(stack_float)
        total_pixels = height * width * slices
        summary_stats.append({
            "region": "FULL_TIFF",
            "total_counts": total_counts,
            "mean_counts": np.mean(stack_float),
            "std_counts": np.std(stack_float),
            "min_counts": np.min(stack_float),
            "max_counts": np.max(stack_float),
            "total_area_pixels": total_pixels,
            "area_per_slice": height * width,
            "n_slices": slices,
            "counts_per_slice_mean": total_counts / slices,
            "counts_per_slice_std": np.std([np.sum(stack[z]) for z in range(slices)])
        })

        # Process each rectangular ROI
        for i, roi in enumerate(tqdm(rect_rois, desc="Processing ROIs", disable=(verbosity == VerbosityLevel.QUIET))):
            roi_name = roi.name if roi.name else f"ROI_{i+1}"

            # Generate mask for rectangular ROI
            # Use integer coordinates directly from the ROI
            left = max(0, int(np.floor(roi.left)))
            top = max(0, int(np.floor(roi.top)))
            right = min(width, int(np.ceil(roi.right)))
            bottom = min(height, int(np.ceil(roi.bottom)))
            
            if right <= left or bottom <= top:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Skipping {roi_name}: invalid rectangle (left={left}, right={right}, top={top}, bottom={bottom})")
                continue

            # Create mask - note: numpy uses [row, col] indexing which is [y, x]
            mask = np.zeros((height, width), dtype=bool)
            mask[top:bottom, left:right] = True
            area = np.sum(mask)
            
            if area == 0:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Skipping {roi_name}: zero area mask")
                continue

            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Processing {roi_name}: bounds=[{left}:{right}, {top}:{bottom}], area={area}")

            # Calculate counts per slice and collect statistics
            results = []
            slice_counts = []
            all_roi_pixels = []
            
            for z in range(slices):
                # Get slice - ensure it's float to avoid overflow
                slice_img = stack[z].astype(np.float64)
                
                # Apply mask and calculate sum
                roi_pixels = slice_img[mask]
                sum_val = np.sum(roi_pixels)
                slice_counts.append(sum_val)
                all_roi_pixels.extend(roi_pixels.flatten())
                
                # Calculate error (sqrt of counts, treating as Poisson statistics)
                err_val = np.sqrt(max(sum_val, 0))
                
                results.append({
                    "stack": z + 1,  # 1-based indexing to match ImageJ
                    "counts": sum_val,
                    "err": err_val
                })

            # Save results to CSV
            df = pd.DataFrame(results)
            csv_path = output_dir / f"{roi_name}.csv"
            df.to_csv(csv_path, index=False)
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Saved: {csv_path} (mean counts: {df['counts'].mean():.2f})")

            # Add ROI statistics to summary
            all_roi_pixels = np.array(all_roi_pixels)
            summary_stats.append({
                "region": roi_name,
                "total_counts": np.sum(slice_counts),
                "mean_counts": np.mean(all_roi_pixels),
                "std_counts": np.std(all_roi_pixels),
                "min_counts": np.min(all_roi_pixels),
                "max_counts": np.max(all_roi_pixels),
                "total_area_pixels": area * slices,
                "area_per_slice": area,
                "n_slices": slices,
                "counts_per_slice_mean": np.mean(slice_counts),
                "counts_per_slice_std": np.std(slice_counts)
            })

            # Perform MTF analysis if "mtf" is in the ROI name
            if "mtf" in roi_name.lower():
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Performing MTF analysis for {roi_name}")
                
                try:
                    self._perform_mtf_analysis(
                        summed_image=summed_image,
                        roi_name=roi_name,
                        left=left,
                        right=right,
                        top=top,
                        bottom=bottom,
                        output_dir=mtf_output_dir,
                        pixel_size_um=pixel_size_um,
                        detector_pixels=detector_pixels,
                        orientation="top-bottom",
                        verbosity=verbosity
                    )
                except Exception as e:
                    if verbosity >= VerbosityLevel.BASIC:
                        print(f"Warning: MTF analysis failed for {roi_name}: {e}")

        # Save summary statistics CSV
        summary_df = pd.DataFrame(summary_stats)
        summary_csv_path = output_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Saved summary statistics: {summary_csv_path}")
            print(f"Completed: {n_rois} ROI spectra saved to {output_dir}")



    def _perform_mtf_analysis(self,
                            summed_image: np.ndarray,
                            roi_name: str,
                            left: int,
                            right: int,
                            top: int,
                            bottom: int,
                            output_dir: Path,
                            pixel_size_um: float,
                            detector_pixels: int,
                            orientation: str = "top-bottom",
                            verbosity: VerbosityLevel = VerbosityLevel.QUIET
                            ) -> None:
        """
        Performs MTF (Modulation Transfer Function) analysis on a knife-edge ROI.
        
        Args:
            summed_image: 2D array of the summed stack
            roi_name: Name of the ROI
            left, right, top, bottom: ROI boundaries
            output_dir: Directory to save MTF results
            pixel_size_um: Pixel size in micrometers
            detector_pixels: Number of detector pixels for frequency conversion
            orientation: Orientation of the edge ("top-bottom" or "left-right")
            verbosity: Verbosity level
        """
        # Extract ROI from summed image
        roi_image = summed_image[top:bottom, left:right]
        
        roi_height = bottom - top
        roi_width = right - left
        
        if orientation=="top-bottom":
            # Horizontal edge (top-bottom) - sum along axis=1 (columns)
            esf = np.nansum(roi_image, axis=1)
            orientation = "top-bottom"
            scan_length = roi_height
        elif orientation=="left-right":
            # Vertical edge (left-right) - sum along axis=0 (rows)
            esf = np.nansum(roi_image, axis=0)
            orientation = "left-right"
            scan_length = roi_width
        else:
            raise ValueError("Invalid orientation. Use 'top-bottom' or 'left-right'.")  
        
        x = np.arange(len(esf))
        
        # Remove any remaining nan/inf values
        valid_mask = np.isfinite(esf)
        if not np.any(valid_mask):
            raise ValueError("No valid data points in ROI after removing nan/inf")
        
        x_clean = x[valid_mask]
        esf_clean = esf[valid_mask]
        
        if len(x_clean) < 10:
            raise ValueError(f"Insufficient valid data points ({len(x_clean)}) for MTF analysis")
        
        # Define erfc model for fitting the Edge Spread Function
        def erfc_model(x, center, width, amplitude, offset):
            return amplitude * 0.5 * erfc((x - center) / (np.sqrt(2) * width)) + offset
        
        # Create lmfit Model
        model = Model(erfc_model)
        
        # Improved initial parameter guesses
        center_guess = len(esf_clean) / 2
        
        # Estimate edge position from where signal changes most
        # Find the steepest gradient
        gradient = np.abs(np.gradient(esf_clean))
        if np.max(gradient) > 0:
            center_guess = x_clean[np.argmax(gradient)]
        
        amplitude_guess = np.nanmax(esf_clean) - np.nanmin(esf_clean)
        offset_guess = np.nanmin(esf_clean)
        
        # Better width estimate: use a small fraction of scan length
        # Typical knife edge is 2-10 pixels wide
        width_guess = min(5.0, len(esf_clean) * 0.05)
        center_guess = 68 if detector_pixels == 256 else 100
        amplitude_guess = -300 if detector_pixels == 256 else -100
        
        params = model.make_params(
            center=center_guess,
            width=width_guess,
            amplitude=amplitude_guess,
            offset=offset_guess
        )
        
        # Set reasonable bounds to prevent unrealistic fits
        # params['width'].min = 0.5
        # params['width'].max = len(esf_clean) * 0.3  # Max 30% of scan range
        # params['amplitude'].min = 0
        params['center'].min = 0
        params['center'].max = len(esf_clean)
        
        # Perform the fit
        try:
            result = model.fit(esf_clean, params, x=x_clean)
        except Exception as e:
            raise ValueError(f"Fit failed: {e}")
        
        # Check fit quality
        if result.redchi > 100:
            if verbosity >= VerbosityLevel.BASIC:
                print(f"Warning: Poor fit quality (reduced χ² = {result.redchi:.2f})")
        
        # Calculate LSF by differentiating the fitted ESF
        fitted_esf = result.eval(x=x_clean)
        lsf = -np.gradient(fitted_esf, x_clean)
        
        # Compute MTF from FFT of LSF
        mtf = np.abs(np.fft.fft(lsf))
        
        # Frequency conversion
        dx_pixels = 1.0  # spacing in x_clean array (pixels)
        frequencies = np.fft.fftfreq(len(x_clean), d=dx_pixels)  # cycles per pixel
        
        positive_freq_idx = len(mtf) // 2
        mtf = mtf[:positive_freq_idx]
        frequencies = frequencies[:positive_freq_idx]
        
        # Normalize MTF
        if mtf[0] > 0:
            mtf = mtf / mtf[0]
        
        # Convert from cycles/pixel to lp/mm
        frequencies_lpmm = frequencies * 1000.0 / pixel_size_um
        
        # Additional diagnostic info
        nyquist_lpmm = 0.5 * 1000.0 / pixel_size_um
        edge_width_mm = result.params['width'].value * pixel_size_um / 1000.0
        
        # Save fit results
        fit_results = {
            'parameter': ['center', 'width', 'amplitude', 'offset', 'reduced_chi2', 
                        'pixel_size_um', 'nyquist_lpmm', 'roi_dimension_pixels',
                        'edge_width_mm', 'orientation'],
            'value': [
                result.params['center'].value,
                result.params['width'].value,
                result.params['amplitude'].value,
                result.params['offset'].value,
                result.redchi,
                pixel_size_um,
                nyquist_lpmm,
                scan_length,
                edge_width_mm,
                orientation
            ],
            'stderr': [
                result.params['center'].stderr if result.params['center'].stderr else np.nan,
                result.params['width'].stderr if result.params['width'].stderr else np.nan,
                result.params['amplitude'].stderr if result.params['amplitude'].stderr else np.nan,
                result.params['offset'].stderr if result.params['offset'].stderr else np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan
            ]
        }
        
        fit_df = pd.DataFrame(fit_results)
        fit_csv_path = output_dir / f"{roi_name}_fit_params.csv"
        fit_df.to_csv(fit_csv_path, index=False)
        
        # Save MTF data
        mtf_data = pd.DataFrame({
            'frequency_lpmm': frequencies_lpmm,
            'mtf': mtf
        })
        mtf_csv_path = output_dir / f"{roi_name}_mtf.csv"
        mtf_data.to_csv(mtf_csv_path, index=False)
        
        # Create diagnostic plots
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Plot 1: ESF and fit
        axes[0, 0].plot(x_clean, esf_clean, 'o', label='Data', markersize=3, alpha=0.6)
        axes[0, 0].plot(x_clean, fitted_esf, 'r-', label='Fit', linewidth=2)
        axes[0, 0].set_xlabel('Position (pixels)')
        axes[0, 0].set_ylabel('Intensity')
        axes[0, 0].set_title(f'Edge Spread Function - {roi_name}\n({orientation})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].text(0.05, 0.95, 
                        f"Width: {result.params['width'].value:.2f} px\n({edge_width_mm:.4f} mm)", 
                        transform=axes[0, 0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: LSF
        axes[0, 1].plot(x_clean, lsf, 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Position (pixels)')
        axes[0, 1].set_ylabel('Intensity Derivative')
        axes[0, 1].set_title('Line Spread Function')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: MTF
        axes[1, 0].plot(frequencies_lpmm, mtf, 'g-', linewidth=2)
        axes[1, 0].axhline(0.1, color='0.7', linestyle='--', zorder=-1, label='MTF=0.1')
        axes[1, 0].set_xlabel('Frequency (lp/mm)')
        axes[1, 0].set_ylabel('MTF')
        axes[1, 0].set_title('Modulation Transfer Function')
        max_freq = min(nyquist_lpmm, 5.0)
        axes[1, 0].set_xlim(0, max_freq)
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Fit statistics
        axes[1, 1].axis('off')
        fit_text = f"Fit Parameters ({orientation}):\n\n"
        fit_text += f"Center: {result.params['center'].value:.2f}"
        if result.params['center'].stderr:
            fit_text += f" ± {result.params['center'].stderr:.2f}"
        fit_text += " px\n"
        
        fit_text += f"Width: {result.params['width'].value:.2f}"
        if result.params['width'].stderr:
            fit_text += f" ± {result.params['width'].stderr:.2f}"
        fit_text += f" px\n       ({edge_width_mm:.4f} mm)\n"
        
        fit_text += f"Amplitude: {result.params['amplitude'].value:.1f}"
        if result.params['amplitude'].stderr:
            fit_text += f" ± {result.params['amplitude'].stderr:.1f}"
        fit_text += "\n"
        
        fit_text += f"Offset: {result.params['offset'].value:.1f}"
        if result.params['offset'].stderr:
            fit_text += f" ± {result.params['offset'].stderr:.1f}"
        fit_text += "\n\n"
        
        fit_text += f"Reduced χ²: {result.redchi:.4f}\n"
        fit_text += f"Pixel size: {pixel_size_um:.2f} μm\n"
        fit_text += f"ROI size: {scan_length} px\n"
        fit_text += f"Nyquist: {nyquist_lpmm:.3f} lp/mm"
        
        axes[1, 1].text(0.1, 0.5, fit_text, fontsize=9, verticalalignment='center', 
                        family='monospace', 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plot_path = output_dir / f"{roi_name}_mtf_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbosity >= VerbosityLevel.DETAILED:
            print(f"MTF Analysis Results for {roi_name}:")
            print(f"  Orientation: {orientation}")
            print(f"  Edge width: {result.params['width'].value:.2f} pixels ({edge_width_mm:.4f} mm)")
            print(f"  Reduced χ²: {result.redchi:.4f}")
            print(f"  Nyquist frequency: {nyquist_lpmm:.3f} lp/mm")
            print(f"  Saved: {fit_csv_path}")
            print(f"  Saved: {mtf_csv_path}")
            print(f"  Saved: {plot_path}")



    def collect_analysis_results(self, group_name: str = "pz", verbosity: VerbosityLevel = VerbosityLevel.QUIET) -> pd.DataFrame:
        """
        Collects MTF analysis and ROI summary statistics from the archive folder structure.
        Searches for MTF_calculation and ROI_spectra subfolders within group/suffix directories
        (e.g., archive/knife_edge/pz/*/suffix/[MTF_calculation|ROI_spectra] for groupby,
        or archive/suffix/[MTF_calculation|ROI_spectra] for non-groupby). The suffix is any
        folder containing MTF_calculation or ROI_spectra subfolders.

        Args:
            group_name (str): Name of the group directory (default: "pz"). Used for groupby structure.
            verbosity (VerbosityLevel): Controls the level of debugging output (QUIET, BASIC, DETAILED).

        Returns:
            pd.DataFrame: Combined DataFrame with MTF and ROI statistics, multi-indexed by
                        group value (if groupby), suffix, and mtf_size (for MTF data).
        """
        result_dfs = []
        archive_path = Path(self.archive).resolve()  # Ensure absolute path

        if verbosity >= VerbosityLevel.BASIC:
            print(f"Resolved archive path: {archive_path}")
            print(f"Is groupby: {self._is_groupby}")
            if self._is_groupby:
                print(f"Groupby subfolders: {[f.name for f in self._groupby_subfolders]}")

        # Helper function to find suffix folders
        def find_suffixes(base_path: Path) -> set:
            suffixes = set()
            if not base_path.exists():
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Base path does not exist: {base_path}")
                return suffixes
            for f in base_path.iterdir():
                if f.is_dir() and not f.name.startswith('.'):
                    mtf_exists = any((f / name).exists() for name in ["MTF_calculation", "MTF_Calculation", "mtf_calculation"])
                    roi_exists = any((f / name).exists() for name in ["ROI_spectra", "ROI_Spectra", "roi_spectra"])
                    if mtf_exists or roi_exists:
                        suffixes.add(f.name)
            return suffixes

        # Check if groupby structure exists
        group_path = archive_path / group_name
        process_groupby = group_path.exists()

        # Process non-groupby structure only if groupby is not present
        if not process_groupby:
            if verbosity >= VerbosityLevel.BASIC:
                print("Checking non-groupby structure")

            suffix_names = find_suffixes(archive_path)
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Non-groupby directories checked: {[f.name for f in archive_path.iterdir() if f.is_dir()]}")
                print(f"Non-groupby suffixes found: {suffix_names}")

            for suffix in suffix_names:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"  Processing non-groupby suffix: {suffix}")

                # Collect MTF analysis results
                mtf_path = archive_path / suffix / "MTF_calculation"
                mtf_pattern = str(mtf_path / "*_fit_params.csv")
                mtf_files = glob.glob(mtf_pattern)
                
                if not mtf_files:
                    mtf_path_alt = archive_path / suffix / "MTF_Calculation"
                    mtf_pattern_alt = str(mtf_path_alt / "*_fit_params.csv")
                    mtf_files = glob.glob(mtf_pattern_alt)
                    mtf_path = mtf_path_alt if mtf_files else mtf_path
                
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"    MTF path: {mtf_path}")
                    print(f"    MTF pattern: {mtf_pattern}")
                    print(f"    MTF files: {mtf_files if mtf_files else 'None'}")

                for fname in mtf_files:
                    fname_path = Path(fname)
                    try:
                        df = pd.read_csv(fname)
                        if df.empty:
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"    Empty MTF file: {fname}")
                            continue
                        
                        # Extract width parameter
                        mtf_data = df.loc[df['parameter'] == 'width', ['value', 'stderr']].copy()
                        if mtf_data.empty:
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"    No 'width' parameter in MTF file: {fname}")
                            continue
                        
                        # Extract mtf_size from filename (e.g., '256' or '512')
                        mtf_size = fname_path.stem.split('_')[1]  # e.g., mtf_256_fit_params -> 256
                        mtf_data['group_value'] = 0.0
                        mtf_data['suffix'] = suffix
                        mtf_data['mtf_size'] = mtf_size
                        mtf_data['metric'] = 'mtf_width'
                        result_dfs.append(mtf_data)
                        if verbosity >= VerbosityLevel.BASIC:
                            print(f"    Added MTF data from {fname} (mtf_size: {mtf_size})")
                    except Exception as e:
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"    Error reading MTF file {fname}: {e}")

                # Collect ROI summary statistics
                roi_path = archive_path / suffix / "ROI_spectra"
                roi_pattern = str(roi_path / "summary_statistics.csv")
                roi_files = glob.glob(roi_pattern)
                
                if not roi_files:
                    roi_path_alt = archive_path / suffix / "ROI_Spectra"
                    roi_pattern_alt = str(roi_path_alt / "summary_statistics.csv")
                    roi_files = glob.glob(roi_pattern_alt)
                    roi_path = roi_path_alt if roi_files else roi_path
                
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"    ROI path: {roi_path}")
                    print(f"    ROI pattern: {roi_pattern}")
                    print(f"    ROI files: {roi_files if roi_files else 'None'}")

                for fname in roi_files:
                    fname_path = Path(fname)
                    try:
                        df = pd.read_csv(fname)
                        if df.empty:
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"    Empty ROI file: {fname}")
                            continue
                        
                        # Extract ROI data (row 1)
                        if len(df) <= 1:
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"    Insufficient data in ROI file: {fname}")
                            continue
                        
                        roi_data = df.loc[1, ['total_counts', 'mean_counts', 'std_counts', 
                                            'min_counts', 'max_counts', 'counts_per_slice_mean', 
                                            'counts_per_slice_std']].copy()
                        roi_data = pd.DataFrame([roi_data])
                        roi_data['group_value'] = 0.0
                        roi_data['suffix'] = suffix
                        roi_data['mtf_size'] = 'none'  # No mtf_size for ROI
                        roi_data['metric'] = 'roi_summary'
                        result_dfs.append(roi_data)
                        if verbosity >= VerbosityLevel.BASIC:
                            print(f"    Added ROI data from {fname}")
                    except Exception as e:
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"    Error reading ROI file {fname}: {e}")

        # Process groupby structure
        if process_groupby:
            if verbosity >= VerbosityLevel.BASIC:
                print(f"Checking groupby structure with group '{group_name}'")

            subfolders = [f for f in group_path.iterdir() if f.is_dir()]
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Group subfolders: {[f.name for f in subfolders]}")

            for subfolder in subfolders:
                try:
                    group_value = float(subfolder.name)
                except ValueError:
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  Skipping non-numeric subfolder: {subfolder.name}")
                    continue

                if verbosity >= VerbosityLevel.BASIC:
                    print(f"  Processing group: {subfolder.name}")

                suffix_names = find_suffixes(subfolder)
                if not suffix_names:
                    if verbosity >= VerbosityLevel.BASIC:
                        print(f"  No suffix folders found in {subfolder}")
                    continue

                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  Found suffixes in {subfolder.name}: {suffix_names}")

                for suffix in suffix_names:
                    if verbosity >= VerbosityLevel.BASIC:
                        print(f"    Processing groupby suffix: {suffix}")

                    # Collect MTF analysis results
                    mtf_path = subfolder / suffix / "MTF_calculation"
                    mtf_pattern = str(mtf_path / "*_fit_params.csv")
                    mtf_files = glob.glob(mtf_pattern)
                    
                    if not mtf_files:
                        mtf_path_alt = subfolder / suffix / "MTF_Calculation"
                        mtf_pattern_alt = str(mtf_path_alt / "*_fit_params.csv")
                        mtf_files = glob.glob(mtf_pattern_alt)
                        mtf_path = mtf_path_alt if mtf_files else mtf_path
                    
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"      MTF path: {mtf_path}")
                        print(f"      MTF pattern: {mtf_pattern}")
                        print(f"      MTF files: {mtf_files if mtf_files else 'None'}")

                    for fname in mtf_files:
                        fname_path = Path(fname)
                        try:
                            df = pd.read_csv(fname)
                            if df.empty:
                                if verbosity >= VerbosityLevel.DETAILED:
                                    print(f"      Empty MTF file: {fname}")
                                continue
                            
                            # Extract width parameter
                            mtf_data = df.loc[df['parameter'] == 'width', ['value', 'stderr']].copy()
                            if mtf_data.empty:
                                if verbosity >= VerbosityLevel.DETAILED:
                                    print(f"      No 'width' parameter in MTF file: {fname}")
                                continue
                            
                            # Extract mtf_size from filename
                            mtf_size = fname_path.stem.split('_')[1]  # e.g., mtf_256_fit_params -> 256
                            mtf_data['group_value'] = group_value
                            mtf_data['suffix'] = suffix
                            mtf_data['mtf_size'] = mtf_size
                            mtf_data['metric'] = 'mtf_width'
                            result_dfs.append(mtf_data)
                            if verbosity >= VerbosityLevel.BASIC:
                                print(f"      Added MTF data from {fname} (mtf_size: {mtf_size})")
                        except Exception as e:
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"      Error reading MTF file {fname}: {e}")

                    # Collect ROI summary statistics
                    roi_path = subfolder / suffix / "ROI_spectra"
                    roi_pattern = str(roi_path / "summary_statistics.csv")
                    roi_files = glob.glob(roi_pattern)
                    
                    if not roi_files:
                        roi_path_alt = subfolder / suffix / "ROI_Spectra"
                        roi_pattern_alt = str(roi_path_alt / "summary_statistics.csv")
                        roi_files = glob.glob(roi_pattern_alt)
                        roi_path = roi_path_alt if roi_files else roi_path
                    
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"      ROI path: {roi_path}")
                        print(f"      ROI pattern: {roi_pattern}")
                        print(f"      ROI files: {roi_files if roi_files else 'None'}")

                    for fname in roi_files:
                        fname_path = Path(fname)
                        try:
                            df = pd.read_csv(fname)
                            if df.empty:
                                if verbosity >= VerbosityLevel.DETAILED:
                                    print(f"      Empty ROI file: {fname}")
                                continue
                            
                            # Extract ROI data (row 1)
                            if len(df) <= 1:
                                if verbosity >= VerbosityLevel.DETAILED:
                                    print(f"      Insufficient data in ROI file: {fname}")
                                continue
                            
                            roi_data = df.loc[1, ['total_counts', 'mean_counts', 'std_counts', 
                                                'min_counts', 'max_counts', 'counts_per_slice_mean', 
                                                'counts_per_slice_std']].copy()
                            roi_data = pd.DataFrame([roi_data])
                            roi_data['group_value'] = group_value
                            roi_data['suffix'] = suffix
                            roi_data['mtf_size'] = 'none'  # No mtf_size for ROI
                            roi_data['metric'] = 'roi_summary'
                            result_dfs.append(roi_data)
                            if verbosity >= VerbosityLevel.BASIC:
                                print(f"      Added ROI data from {fname}")
                        except Exception as e:
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"      Error reading ROI file {fname}: {e}")

        if not result_dfs:
            if verbosity >= VerbosityLevel.BASIC:
                print("No data collected. Returning empty DataFrame.")
            return pd.DataFrame()

        # Combine all results
        combined_df = pd.concat(result_dfs, ignore_index=True, sort=False)
        
        # Convert numeric columns to float
        numeric_columns = ['value', 'stderr', 'total_counts', 'mean_counts', 'std_counts',
                        'min_counts', 'max_counts', 'counts_per_slice_mean', 
                        'counts_per_slice_std']
        for col in numeric_columns:
            if col in combined_df:
                combined_df[col] = combined_df[col].astype(float)

        # Debug: Print DataFrame before unstack
        if verbosity >= VerbosityLevel.DETAILED:
            print("DataFrame before unstack:")
            print(combined_df)

        # Create multi-index and pivot to organize by metric and mtf_size
        combined_df = combined_df.set_index(['group_value', 'suffix', 'mtf_size', 'metric'])
        try:
            combined_df = combined_df.unstack(level=['metric', 'mtf_size']).reset_index()
        except ValueError as e:
            if verbosity >= VerbosityLevel.BASIC:
                print(f"Error during unstack: {e}")
                print("Index value counts:")
                print(combined_df.index.value_counts())
            raise

        # Flatten column names
        combined_df.columns = [f"{col[0]}_{col[1]}_{col[2]}" if col[1] else col[0] 
                            for col in combined_df.columns]
        
        # Sort by group_value and suffix
        combined_df = combined_df.sort_values(['group_value', 'suffix'])
        
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Collected data for {len(combined_df)} rows")
            print(f"Columns: {list(combined_df.columns)}")
        
        return combined_df
    
    def _run_pixel2photon(self, tpx3_dir: Path, photon_files_dir: Path,
                        params_file: Path, n_threads: int,
                        parameters: Dict[str, Any],
                        verbosity: VerbosityLevel) -> None:
        """Run pixel2photon processing on TPX3 files and add neutron_id from traced photons.
        
        Args:
            tpx3_dir: Directory containing .tpx3 files.
            photon_files_dir: Directory to save .empirphot files.
            params_file: Path to the parameter settings JSON file.
            n_threads: Number of threads to use for parallel processing.
            parameters: Dictionary of parameters for pixel2photon.
            verbosity: Controls the level of output during processing.
        """
        tpx3_files = sorted([f for f in tpx3_dir.glob("*.tpx3")])
        if not tpx3_files:
            raise FileNotFoundError(f"No .tpx3 files found in {tpx3_dir}")
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Processing {len(tpx3_files)} .tpx3 files to photon files...")
        
        # Find traced photons directory
        traced_photons_dir = tpx3_dir.parent / "TracedPhotons"
        if not traced_photons_dir.exists():
            if verbosity > VerbosityLevel.BASIC:
                print(f"Warning: TracedPhotons directory not found at {traced_photons_dir}")
                print("Proceeding without neutron_id mapping")
            traced_photons_dir = None
        
        pids = []
        file_cnt = 0
        result_all = 0
        
        for tpx3_file in tqdm(tpx3_files, desc="Processing pixel2photon", disable=(verbosity == VerbosityLevel.QUIET)):
            file_base = tpx3_file.stem
            file_cnt += 1
            output_file = photon_files_dir / f"{file_base}.empirphot"
            
            cmd = [
                str(self.empir_dirpath / "bin/empir_pixel2photon_tpx3spidr"),
                "-i", str(tpx3_file),
                "-o", str(output_file),
                "--paramsFile", str(params_file)
            ]
            
            if parameters.get("pixel2photon", {}).get("TDC1", False):
                cmd.append("-T")
            
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Running: {' '.join(cmd)}")
                process = subprocess.Popen(cmd)
            else:
                process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            pids.append(process)
            
            if len(pids) >= n_threads:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Reached {n_threads} files, waiting for processing...")
                for process in pids:
                    result = process.wait()
                    if result != 0:
                        result_all = 1
                        if verbosity > VerbosityLevel.BASIC:
                            print(f"Error occurred while processing a file!")
                pids = []
                if verbosity > VerbosityLevel.BASIC:
                    print("Processed files, continuing...")
        
        for process in pids:
            result = process.wait()
            if result != 0:
                result_all = 1
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Error occurred while processing a file!")
        
        if result_all != 0:
            if verbosity > VerbosityLevel.BASIC:
                print("Errors occurred during pixel2photon processing")
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Finished processing {file_cnt} .tpx3 files")
        
        # Now add neutron_id to the empirphot files
        if traced_photons_dir is not None:
            self._add_neutron_id_to_photon_files(
                photon_files_dir=photon_files_dir,
                traced_photons_dir=traced_photons_dir,
                verbosity=verbosity
            )


    def _add_neutron_id_to_photon_files(self, photon_files_dir: Path, 
                                        traced_photons_dir: Path,
                                        verbosity: VerbosityLevel) -> None:
        """Add neutron_id column to empirphot files by matching with traced photons.
        
        Args:
            photon_files_dir: Directory containing .empirphot files.
            traced_photons_dir: Directory containing traced_*.csv files.
            verbosity: Controls the level of output during processing.
        """
        if verbosity > VerbosityLevel.BASIC:
            print("\nAdding neutron_id to photon files...")
        
        empirphot_files = sorted([f for f in photon_files_dir.glob("*.empirphot")])
        if not empirphot_files:
            if verbosity > VerbosityLevel.BASIC:
                print("No .empirphot files found to process")
            return
        
        # Parse TPX3 filename to extract the data file index
        # Format: traced_data_X_partY.tpx3 or traced_data_X.tpx3
        def extract_data_index(filename: str) -> int:
            """Extract data index from TPX3 filename."""
            # Remove extension
            stem = filename.replace('.empirphot', '').replace('.tpx3', '')
            
            # Handle partitioned files: traced_data_10_part003
            if '_part' in stem:
                stem = stem.split('_part')[0]
            
            # Extract the number: traced_data_10 -> 10
            parts = stem.split('_')
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
            
            raise ValueError(f"Could not extract data index from filename: {filename}")
        
        for empirphot_file in tqdm(empirphot_files, desc="Adding neutron_id", 
                                disable=(verbosity == VerbosityLevel.QUIET)):
            try:
                # Extract the data file index
                data_index = extract_data_index(empirphot_file.stem)
                
                # Find corresponding traced photons file
                traced_file = traced_photons_dir / f"traced_sim_data_{data_index}.csv"
                
                if not traced_file.exists():
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Warning: Traced file not found for {empirphot_file.name}: {traced_file}")
                    continue
                
                # Read the empirphot file
                try:
                    empirphot_df = pd.read_csv(empirphot_file)
                except Exception as e:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"Error reading {empirphot_file.name}: {e}")
                    continue
                
                if empirphot_df.empty:
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Skipping empty file: {empirphot_file.name}")
                    continue
                
                # Read the traced photons file
                try:
                    traced_df = pd.read_csv(traced_file)
                except Exception as e:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"Error reading {traced_file.name}: {e}")
                    continue
                
                # Filter to only rows that are in TPX3
                if 'in_tpx3' in traced_df.columns:
                    traced_df = traced_df[traced_df['in_tpx3']].copy()
                else:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"Warning: 'in_tpx3' column not found in {traced_file.name}")
                
                # Check if neutron_id exists
                if 'neutron_id' not in traced_df.columns:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"Warning: 'neutron_id' column not found in {traced_file.name}")
                    continue
                
                # The empirphot file should have the same number of rows as traced photons with in_tpx3=True
                # They should be in the same order (chronological by toa2)
                if len(empirphot_df) != len(traced_df):
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"Warning: Row count mismatch for {empirphot_file.name}")
                        print(f"  empirphot: {len(empirphot_df)} rows, traced (in_tpx3): {len(traced_df)} rows")
                    
                    # Try to match based on available columns if counts don't match
                    # This is a fallback - ideally counts should match
                    if 'pixel_x' in empirphot_df.columns and 'pixel_x' in traced_df.columns:
                        # Attempt merge on pixel_x, pixel_y, and toa (if available)
                        merge_cols = ['pixel_x', 'pixel_y']
                        if 'toa' in empirphot_df.columns and 'toa2' in traced_df.columns:
                            # Rename toa2 to toa for merging
                            traced_df = traced_df.rename(columns={'toa2': 'toa'})
                            merge_cols.append('toa')
                        
                        empirphot_df = empirphot_df.merge(
                            traced_df[merge_cols + ['neutron_id']],
                            on=merge_cols,
                            how='left'
                        )
                    else:
                        if verbosity > VerbosityLevel.BASIC:
                            print(f"  Cannot merge: missing required columns")
                        continue
                else:
                    # Simple case: same number of rows, assume same order
                    empirphot_df['neutron_id'] = traced_df['neutron_id'].values
                
                # Save the updated empirphot file
                empirphot_df.to_csv(empirphot_file, index=False)
                
                if verbosity >= VerbosityLevel.DETAILED:
                    neutron_count = empirphot_df['neutron_id'].notna().sum()
                    print(f"Added neutron_id to {empirphot_file.name}: {neutron_count} valid IDs")
            
            except Exception as e:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Error processing {empirphot_file.name}: {e}")
                continue
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Finished adding neutron_id to {len(empirphot_files)} photon files")

    def _run_photon2event(self, photon_files_dir: Path, event_files_dir: Path, params_file: Path, n_threads: int, verbosity: VerbosityLevel) -> None:
        """Run photon2event processing on .empirphot files.
        
            args:
                photon_files_dir: Directory containing .empirphot files.
                event_files_dir: Directory to save .empirevent files.
                params_file: Path to the parameter settings JSON file.
                n_threads: Number of threads to use for parallel processing.
                verbosity: Controls the level of output during processing.
        """
        empirphot_files = sorted(photon_files_dir.glob("*.empirphot"))
        if not empirphot_files:
            raise FileNotFoundError(f"No .empirphot files found in {photon_files_dir}")

        if verbosity > VerbosityLevel.BASIC:
            print(f"Processing {len(empirphot_files)} .empirphot files to event files...")

        pids = []
        file_cnt = 0
        result_all = 0

        for empirphot_file in tqdm(empirphot_files, desc="Processing photon2event", disable=(verbosity == VerbosityLevel.QUIET)):
            file_base = empirphot_file.stem
            file_cnt += 1
            output_file = event_files_dir / f"{file_base}.empirevent"
            cmd = [
                str(self.empir_dirpath / "bin/empir_photon2event"),
                "-i", str(empirphot_file),
                "-o", str(output_file),
                "--paramsFile", str(params_file)
            ]
            
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Running: {' '.join(cmd)}")
                process = subprocess.Popen(cmd)
            else:
                process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            pids.append(process)
            
            if len(pids) >= n_threads:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Reached {n_threads} files, waiting for processing...")
                for process in pids:
                    result = process.wait()
                    if result != 0:
                        result_all = 1
                        if verbosity > VerbosityLevel.BASIC:
                            print(f"Error occurred while processing a file!")
                pids = []
                if verbosity > VerbosityLevel.BASIC:
                    print("Processed files, continuing...")

        for process in pids:
            result = process.wait()
            if result != 0:
                result_all = 1
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Error occurred while processing a file!")

        if result_all != 0:
            raise RuntimeError("Errors occurred during photon2event processing")

        if verbosity > VerbosityLevel.BASIC:
            print(f"Finished processing {file_cnt} .empirphot files")


    def _run_event2image(self, event_files_dir: Path, final_dir: Path, params_file: Path, 
                        n_threads: int, sum_image: bool = False,
                        verbosity: VerbosityLevel = VerbosityLevel.BASIC, 
                        ) -> None:
        """Run event2image processing on all .empirevent files to produce a single image.
        
        Args:
            event_files_dir: Directory containing .empirevent files.
            final_dir: Directory to save the final .empirimage file.
            params_file: Path to the parameter settings JSON file.
            n_threads: Number of threads to use for parallel processing (not used in this implementation).
            sum_image: If True, generates both time-binned image and sum image. If False, only generates time-binned image.
            verbosity: Controls the level of output during processing.

        """
        empirevent_files = sorted(event_files_dir.glob("*.empirevent"))
        if not empirevent_files:
            raise FileNotFoundError(f"No .empirevent files found in {event_files_dir}")
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Processing {len(empirevent_files)} .empirevent files into a single image...")
        
        # Generate the main time-binned image
        output_file = final_dir / "image"
        cmd = [
            str(self.empir_dirpath / "bin/empir_event2image"),
            "-I", str(event_files_dir),
            "-o", str(output_file),
            "--paramsFile", str(params_file)
        ]
        
        if verbosity >= VerbosityLevel.DETAILED:
            print(f"Running: {' '.join(cmd)}")
            process = subprocess.run(cmd, check=True)
        else:
            process = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        if process.returncode != 0:
            raise RuntimeError("Error occurred during event2image processing")
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Finished producing time-binned image: {output_file}")
        
        # Generate sum image if requested
        if sum_image:
            if verbosity > VerbosityLevel.BASIC:
                print(f"Generating sum image (without time binning)...")
            
            # Load params and remove time binning parameters
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            # Remove time binning parameters if they exist
            if "event2image" in params:
                params["event2image"].pop("time_res_s", None)
                params["event2image"].pop("time_limit", None)
            
            # Create temporary params file for sum image
            sum_params_file = final_dir / ".sum_params_temp.json"
            with open(sum_params_file, 'w') as f:
                json.dump(params, f, indent=2)
            
            # Generate sum image
            sum_output_file = final_dir / "sum_image"
            sum_cmd = [
                str(self.empir_dirpath / "bin/empir_event2image"),
                "-I", str(event_files_dir),
                "-o", str(sum_output_file),
                "--paramsFile", str(sum_params_file)
            ]
            
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Running: {' '.join(sum_cmd)}")
                sum_process = subprocess.run(sum_cmd, check=True)
            else:
                sum_process = subprocess.run(sum_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            if sum_process.returncode != 0:
                raise RuntimeError("Error occurred during sum image generation")
            
            # Clean up temporary params file
            sum_params_file.unlink()
            
            if verbosity > VerbosityLevel.BASIC:
                print(f"Finished producing sum image: {sum_output_file}")

            
    def _run_export_photons(self, process_dir: Path, verbosity: VerbosityLevel = VerbosityLevel.QUIET):
        """
        Exports .empirphot files from photonFiles subfolder to CSV files in ExportedPhotons subfolder.
        
        Args:
            process_dir: Path to the processing directory (supports suffix).
            verbosity: Controls the level of output during processing.
        """
        photon_files_dir = process_dir / "photonFiles"
        if not photon_files_dir.exists():
            raise FileNotFoundError(f"{photon_files_dir} does not exist.")
        
        exported_photons_dir = process_dir / "ExportedPhotons"
        exported_photons_dir.mkdir(parents=True, exist_ok=True)
        
        empirphot_files = sorted(photon_files_dir.glob("*.empirphot"))
        if not empirphot_files:
            raise FileNotFoundError(f"No .empirphot files found in {photon_files_dir}")
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Exporting {len(empirphot_files)} .empirphot files to CSV...")
        
        for empirphot_file in tqdm(empirphot_files, desc="Exporting photons", disable=(verbosity == VerbosityLevel.QUIET)):
            try:
                photon_result_csv = exported_photons_dir / f"exported_{empirphot_file.stem}.csv"
                cmd = [
                    str(self.empir_dirpath / "empir_export_photons"),
                    str(empirphot_file),
                    str(photon_result_csv),
                    "csv"
                ]
                
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Running: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                else:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                try:
                    df = pd.read_csv(photon_result_csv)
                    df.columns = ["x", "y", "toa", "tof"]
                    df["x"] = df["x"].astype(float)
                    df["y"] = df["y"].astype(float)
                    df["toa"] = df["toa"].astype(float)
                    df["tof"] = pd.to_numeric(df["tof"], errors="coerce")
                    df.to_csv(photon_result_csv, index=False)
                    
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"✔ Exported and modified {empirphot_file.name} → {photon_result_csv.name}")
                        
                except Exception as e:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"⚠️ Error modifying headers for {photon_result_csv.name}: {e}")
                    
            except Exception as e:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"❌ Error exporting {empirphot_file.name}: {e}")
        
        if verbosity > VerbosityLevel.BASIC:
            print("✅ Finished exporting and modifying all photon files!")

    def _run_export_events(self, process_dir: Path, verbosity: VerbosityLevel = VerbosityLevel.QUIET):
        """
        Exports .empirevent files from eventFiles subfolder to CSV files in ExportedEvents subfolder.
        
        Args:
            process_dir: Path to the processing directory (supports suffix).
            verbosity: Controls the level of output during processing.
        """
        event_files_dir = process_dir / "eventFiles"
        if not event_files_dir.exists():
            raise FileNotFoundError(f"{event_files_dir} does not exist.")
        
        exported_events_dir = process_dir / "ExportedEvents"
        exported_events_dir.mkdir(parents=True, exist_ok=True)
        
        empirevent_files = sorted(event_files_dir.glob("*.empirevent"))
        if not empirevent_files:
            raise FileNotFoundError(f"No .empirevent files found in {event_files_dir}")
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Exporting {len(empirevent_files)} .empirevent files to CSV...")
        
        for empirevent_file in tqdm(empirevent_files, desc="Exporting events", disable=(verbosity == VerbosityLevel.QUIET)):
            try:
                event_result_csv = exported_events_dir / f"{empirevent_file.stem}.csv"
                cmd = [
                    str(self.empir_dirpath / "empir_export_events"),
                    str(empirevent_file),
                    str(event_result_csv),
                    "csv"
                ]
                
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Running: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                else:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                try:
                    import pandas as pd
                    df = pd.read_csv(event_result_csv)
                    df.columns = ["x", "y", "t", "n", "PSD", "tof"]
                    df["tof"] = df["tof"].astype(float)
                    df["PSD"] = df["PSD"].astype(float)
                    df.to_csv(event_result_csv, index=False)
                    
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"✔ Exported and modified {empirevent_file.name} → {event_result_csv.name}")
                        
                except Exception as e:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"⚠️ Error modifying headers for {event_result_csv.name}: {e}")
                    
            except Exception as e:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"❌ Error exporting {empirevent_file.name}: {e}")
        
        if verbosity > VerbosityLevel.BASIC:
            print("✅ Finished exporting and modifying all event files!")

    def process(self, 
                params: Union[str, Dict[str, Any]] = None,
                n_threads: int = 1,
                suffix: str = "",
                pixel2photon: bool = True,
                photon2event: bool = True,
                event2image: bool = False,
                sum_image: bool = False,
                export_photons: bool = True,
                export_events: bool = False,
                roifile: Optional[str] = None,
                verbosity: VerbosityLevel = VerbosityLevel.BASIC,
                clean: bool = True,
                **kwargs) -> None:
        """
        Process TPX3 files through the EMPIR pipeline.
        
        If roifile is provided, sets event2image=True and runs _run_roi_analysis on the output TIFF.
        
        Args:
            params: Either a path to a parameterSettings.json file, a JSON string, or a dictionary
            n_threads: Number of threads for parallel processing
            suffix: Optional suffix for creating a subfolder (ignored for groupby structures)
            pixel2photon: If True, runs empir_pixel2photon_tpx3spidr
            photon2event: If True, runs empir_photon2event
            event2image: If True, runs empir_event2image
            sum_image: If True, generates both time-binned image and sum image
            export_photons: If True, exports photons to CSV
            export_events: If True, exports events to CSV
            roifile: Optional path to roi.zip file for ROI analysis
            verbosity: Controls output level
            clean: If True, deletes existing processed files
            **kwargs: Additional parameters
        """
        # If roifile is provided, ensure event2image is True
        if roifile is not None:
            event2image = True
            if not Path(roifile).exists():
                raise FileNotFoundError(f"ROI file not found: {roifile}")

        # Auto-detect and handle groupby structure
        if self._is_groupby:
            if verbosity > VerbosityLevel.BASIC:
                print("Detected groupby structure, processing all groups...")
            return self._process_grouped(
                params=params,
                n_threads=n_threads,
                suffix=suffix,
                pixel2photon=pixel2photon,
                photon2event=photon2event,
                event2image=event2image,
                sum_image=sum_image,
                export_photons=export_photons,
                export_events=export_events,
                roifile=roifile,
                verbosity=verbosity,
                clean=clean,
                **kwargs
            )
        
        # Call the single processing method
        return self._process_single(
            params=params,
            n_threads=n_threads,
            suffix=suffix,
            pixel2photon=pixel2photon,
            photon2event=photon2event,
            event2image=event2image,
            sum_image=sum_image,
            export_photons=export_photons,
            export_events=export_events,
            roifile=roifile,
            verbosity=verbosity,
            clean=clean,
            **kwargs
        )

    def _process_grouped(self, 
                        params: Union[str, Dict[str, Any]] = None,
                        n_threads: int = 1,
                        suffix: str = "",
                        pixel2photon: bool = True,
                        photon2event: bool = True,
                        event2image: bool = False,
                        sum_image: bool = False,
                        export_photons: bool = True,
                        export_events: bool = False,
                        roifile: Optional[str] = None,
                        verbosity: VerbosityLevel = VerbosityLevel.BASIC,
                        clean: bool = True,
                        **kwargs) -> None:
        """
        Process TPX3 files for all groups in a groupby structure.
        
        Args:
            params: Parameters for processing (same as process())
            n_threads: Number of threads for parallel processing
            suffix: Optional suffix for creating a subfolder within each group
            pixel2photon: If True, runs empir_pixel2photon_tpx3spidr
            photon2event: If True, runs empir_photon2event
            event2image: If True, runs empir_event2image
            sum_image: If True, generates both time-binned image and sum image
            export_photons: If True, exports photons to CSV
            export_events: If True, exports events to CSV
            verbosity: Controls output level
            clean: If True, deletes existing processed files before processing
            **kwargs: Additional parameters
        
        Raises:
            ValueError: If not a groupby structure
        """
        if not self._is_groupby:
            raise ValueError("This archive is not a groupby structure. Use process() instead.")
        
        if not self._groupby_subfolders:
            raise ValueError("No valid group subfolders found.")
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"\n{'='*60}")
            print(f"Processing {len(self._groupby_subfolders)} groups")
            print(f"Column: {self._groupby_metadata.get('column', 'unknown')}")
            print(f"{'='*60}\n")
        
        original_archive = self.archive
        original_photon_files_dir = self.photon_files_dir
        
        # Resolve roifile to absolute path
        if roifile:
            roifile = str(Path(roifile).resolve())
        
        group_iter = enumerate(self._groupby_subfolders)
        if verbosity >= VerbosityLevel.BASIC:
            group_iter = enumerate(tqdm(self._groupby_subfolders, 
                                        desc="Processing groups",
                                        position=0,
                                        leave=True))
        
        for i, group_folder in group_iter:
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"\n{'─'*60}")
                print(f"Group {i+1}/{len(self._groupby_subfolders)}: {group_folder.name}")
                print(f"{'─'*60}")
            
            self.archive = group_folder
            if suffix:
                self.photon_files_dir = self.archive / suffix.strip("_") / "photonFiles"
            else:
                self.photon_files_dir = self.archive / "photonFiles"
            self.photon_files_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                tpx3_dir = group_folder / "tpx3Files"
                if not tpx3_dir.exists() or not list(tpx3_dir.glob("*.tpx3")):
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  No TPX3 files found in {group_folder.name}, skipping")
                    continue
                
                self._process_single(
                    params=params,
                    n_threads=n_threads,
                    suffix=suffix,
                    pixel2photon=pixel2photon,
                    photon2event=photon2event,
                    event2image=event2image,
                    sum_image=sum_image,
                    export_photons=export_photons,
                    export_events=export_events,
                    roifile=roifile,
                    verbosity=VerbosityLevel.QUIET,
                    clean=clean,
                    **kwargs
                )
                
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"✓ Completed group '{group_folder.name}'")
            
            except Exception as e:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"\n✗ Error processing group '{group_folder.name}': {e}")
                if verbosity >= VerbosityLevel.DETAILED:
                    import traceback
                    traceback.print_exc()
            
            finally:
                self.archive = original_archive
                self.photon_files_dir = original_photon_files_dir
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"\n{'='*60}")
            print(f"✓ Completed all {len(self._groupby_subfolders)} groups")
            print(f"{'='*60}\n")

    def _process_single(self, 
                        params: Union[str, Dict[str, Any]] = None,
                        n_threads: int = 1,
                        suffix: str = "",
                        pixel2photon: bool = True,
                        photon2event: bool = True,
                        event2image: bool = False,
                        sum_image: bool = False,
                        export_photons: bool = True,
                        export_events: bool = False,
                        roifile: Optional[str] = None,
                        verbosity: VerbosityLevel = VerbosityLevel.BASIC,
                        clean: bool = True,
                        **kwargs) -> None:
        """
        Process TPX3 files through the EMPIR pipeline.
        
        Automatically detects groupby structures and processes all groups if found.
        
        Args:
            params: Either a path to a parameterSettings.json file, a JSON string, or a dictionary
            n_threads: Number of threads for parallel processing
            suffix: Optional suffix for creating a subfolder (ignored for groupby structures)
            pixel2photon: If True, runs empir_pixel2photon_tpx3spidr
            photon2event: If True, runs empir_photon2event
            event2image: If True, runs empir_event2image
            sum_image: If True, generates both time-binned image and sum image
            export_photons: If True, exports photons to CSV
            export_events: If True, exports events to CSV
            verbosity: Controls output level
            clean: If True, deletes existing processed files
            **kwargs: Additional parameters
        """
            
        import time
        start_time = time.time()

        base_dir = self.archive
        if suffix:
            process_dir = base_dir / suffix.strip("_")
            process_dir.mkdir(parents=True, exist_ok=True)
            tpx3_dir = base_dir / "tpx3Files"  # Use existing tpx3Files from group
        else:
            process_dir = base_dir
            tpx3_dir = base_dir / "tpx3Files"

        if pixel2photon == True and (not tpx3_dir.exists() or not list(tpx3_dir.glob("*.tpx3"))):
            raise FileNotFoundError(f"No .tpx3 files found in {tpx3_dir}")

        photon_files_dir = process_dir / "photonFiles"
        event_files_dir = process_dir / "eventFiles"
        final_dir = process_dir / "final"
        # photon_files_dir.mkdir(parents=True, exist_ok=True)
        # event_files_dir.mkdir(parents=True, exist_ok=True)
        # final_dir.mkdir(parents=True, exist_ok=True)

        if clean:
            if pixel2photon==True:
                for file in photon_files_dir.glob("*.empirphot"):
                    file.unlink(missing_ok=True)
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Deleted existing file: {file.name}")
            if photon2event==True:
                for file in event_files_dir.glob("*.empirevent"):
                    file.unlink(missing_ok=True)
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Deleted existing file: {file.name}")
            if event2image==True:
                for file in final_dir.glob("*"):
                    file.unlink(missing_ok=True)
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Deleted existing file: {file.name}")
            if export_photons==True:
                exported_photons_dir = process_dir / "ExportedPhotons"
                for file in exported_photons_dir.glob("*.csv"):
                    file.unlink(missing_ok=True)
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Deleted existing file: {file.name}")
            if export_events==True:
                exported_events_dir = process_dir / "ExportedEvents"
                for file in exported_events_dir.glob("*.csv"):
                    file.unlink(missing_ok=True)
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Deleted existing file: {file.name}")

        if params is None:
            parameters = self.default_params.get("in_focus", {})
        elif isinstance(params, str):
            if params in ["in_focus", "out_of_focus", "hitmap"]:
                parameters = self.default_params.get(params, {})
            elif params.endswith('.json'):
                if not os.path.exists(params):
                    raise FileNotFoundError(f"Parameter file {params} not found")
                with open(params, 'r') as f:
                    parameters = json.load(f)
            else:
                try:
                    parameters = json.loads(params)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON string provided for parameters")
        elif isinstance(params, dict):
            parameters = params
        else:
            raise ValueError("params must be 'in_focus', 'out_of_focus', 'hitmap', a JSON file path, JSON string, or dictionary")

        if kwargs:
            if "pixel2photon" in parameters:
                parameters["pixel2photon"].update({k: v for k, v in kwargs.items() if k in ["dSpace", "dTime", "nPxMin", "nPxMax", "TDC1"]})
            if "photon2event" in parameters:
                parameters["photon2event"].update({k: v for k, v in kwargs.items() if k in ["dSpace_px", "dTime_s", "durationMax_s", "dTime_ext"]})
            if "event2image" in parameters:
                parameters["event2image"].update({k: v for k, v in kwargs.items() if k in ["size_x", "size_y", "nPhotons_min", "nPhotons_max", "time_res_s", "time_limit", "psd_min", "time_extTrigger"]})

        params_file = process_dir / ".parameterSettings.json"
        with open(params_file, 'w') as f:
            json.dump(parameters, f, indent=4)
        if verbosity >= VerbosityLevel.DETAILED:
            print(f"Parameters written to {params_file}")

        if pixel2photon:
            self._run_pixel2photon(tpx3_dir, photon_files_dir, params_file, n_threads, parameters, verbosity)
        
        if photon2event:
            self._run_photon2event(photon_files_dir, event_files_dir, params_file, n_threads, verbosity)
        
        if export_photons:
            self._run_export_photons(process_dir, verbosity=verbosity)
        
        if export_events:
            self._run_export_events(process_dir, verbosity=verbosity)
        
        if event2image:
            self._run_event2image(event_files_dir, final_dir, params_file, n_threads, sum_image, verbosity)
        
        if roifile:
            tiff_path = final_dir / "image"
            output_dir = process_dir / "ROI_spectra"
            if not tiff_path.exists():
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Warning: TIFF file not found at {tiff_path}, skipping ROI analysis")
            else:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Running ROI analysis on {tiff_path} with ROI file {roifile}")
                self._run_roi_analysis(tiff_path, roifile, output_dir, verbosity=verbosity)
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Total processing time: {time.time() - start_time:.2f} seconds")