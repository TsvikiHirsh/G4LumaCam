import logging
# Suppress all INFO messages globally
logging.disable(logging.INFO)
from rayoptics.environment import OpticalModel, PupilSpec, FieldSpec, WvlSpec, InteractiveLayout
from rayoptics.environment import RayFanFigure, SpotDiagramFigure, Fit, open_model
from rayoptics.gui import roafile
from rayoptics.elem.elements import Element
from rayoptics.raytr.trace import apply_paraxial_vignetting, trace_base
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional
from pathlib import Path
from multiprocessing import Pool
from functools import partial   
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from rayoptics.raytr import analyses
from lmfit import Parameters, minimize, MinimizerException
from copy import deepcopy
import glob
import importlib.resources
from enum import IntEnum
from io import StringIO
from contextlib import redirect_stdout
import tempfile
import os
import struct
import json
import shutil

class VerbosityLevel(IntEnum):
    """Verbosity levels for simulation output."""
    QUIET = 0    # Show nothing except progress bar
    BASIC = 1    # Show progress bar and basic info
    DETAILED = 2 # Show everything

class DetectorModel(IntEnum):
    """Physical models for photon detection and sensor response.

    Available Models:
    -----------------
    IMAGE_INTENSIFIER : Default model
        Simulates MCP-based image intensifier coupled to event camera.
        - Circular blob with uniform radius
        - Exponential delay for photon conversion
        - Independent pixel deadtime
        Parameters: blob, blob_variance, decay_time, deadtime, min_tot

    GAUSSIAN_DIFFUSION : Charge diffusion model
        Models charge spreading in solid-state detectors with Gaussian PSF.
        - Gaussian charge distribution instead of uniform blob
        - Charge-weighted TOT accumulation
        - Better for CCD/CMOS direct detection
        Parameters: blob (as sigma), deadtime, min_tot, charge_coupling (0-1)

    DIRECT_DETECTION : Simple direct detection
        Minimal model for bare sensors without intensification.
        - Single pixel per photon
        - Simple deadtime only
        - Fast computation for large datasets
        Parameters: deadtime, min_tot

    WAVELENGTH_DEPENDENT : Advanced intensifier with spectral response
        Energy/wavelength-dependent quantum efficiency and gain.
        - Wavelength-dependent blob size
        - QE curve affects detection probability
        - Non-uniform gain distribution
        Parameters: blob, decay_time, deadtime, min_tot, qe_wavelength (nm array), qe_values (efficiency array)

    AVALANCHE_GAIN : Stochastic gain model
        Models avalanche photodiodes or PMTs with Poisson gain statistics.
        - Poisson-distributed gain per photon
        - Afterpulsing effects
        - Gain-dependent TOT
        Parameters: mean_gain, gain_variance, afterpulse_prob, afterpulse_delay, deadtime, min_tot

    IMAGE_INTENSIFIER_GAIN : Gain-dependent intensifier (RECOMMENDED)
        Realistic MCP image intensifier with variable gain control.
        - Blob size scales with gain: σ ∝ (gain)^0.4
        - Gaussian photon distribution
        - Charge-weighted TOT calculation
        - Based on MCP physics literature
        Parameters: gain (default 5000), sigma_0, gain_exponent, deadtime (475 ns), tot_mode

    TIMEPIX3_CALIBRATED : Timepix3-specific model
        Calibrated for Timepix3 detector response.
        - Logarithmic TOT: TOT = a + b × ln(Q)
        - Per-pixel calibration variation
        - 475 ns deadtime (TPX3 spec)
        - Based on Poikela et al. 2014
        Parameters: gain, tot_a, tot_b, charge_ref, pixel_variation, deadtime (475 ns)

    PHYSICAL_MCP : Full physics simulation
        High-fidelity MCP simulation with complete physics.
        - Poisson gain statistics
        - Multi-exponential phosphor decay
        - Energy-dependent QE
        - MCP pore-level simulation
        Parameters: gain, gain_noise_factor, phosphor_type, decay_fast, decay_slow, mcp_voltage
    """
    IMAGE_INTENSIFIER = 0      # Default: circular blob + exponential decay
    GAUSSIAN_DIFFUSION = 1     # Gaussian charge spreading
    DIRECT_DETECTION = 2       # Single pixel, simple deadtime
    WAVELENGTH_DEPENDENT = 3   # Spectral QE and wavelength-dependent response
    AVALANCHE_GAIN = 4         # Poisson gain statistics with afterpulsing
    IMAGE_INTENSIFIER_GAIN = 5 # Gain-dependent blob (RECOMMENDED for TPX3)
    TIMEPIX3_CALIBRATED = 6    # TPX3-specific calibration
    PHYSICAL_MCP = 7           # Full physics MCP simulation

def _process_ray_chunk_standalone(chunk, opm_file_path, wvl_values, verbosity=0):
    """
    Process a chunk of rays using an optical model loaded from file.
    
    This standalone function is designed to be pickle-safe for multiprocessing 
    by loading the optical model from a file path rather than passing the object directly.
    
    Parameters:
    -----------
    chunk : list
        List of ray tuples: (position, direction, wavelength)
    opm_file_path : str
        Path to the optical model file (.roa)
    wvl_values : numpy.ndarray
        Wavelength specification array for the optical model
    verbosity : int
        Verbosity level for logging (0=QUIET, 1=BASIC, 2=DETAILED)
        
    Returns:
    --------
    list or None
        List of traced ray results, or None for failed traces
    """
    try:
        # Log start of chunk processing
        if verbosity >= 2:
            print(f"Worker processing chunk of {len(chunk)} rays with opm_file: {opm_file_path}")

        # Load the optical model from file
        try:
            opt_model = open_model(opm_file_path)
        except Exception as e:
            if verbosity >= 2:
                print(f"Error loading optical model from {opm_file_path}: {str(e)}")
            return [None] * len(chunk)
        
        # Set the spectral region
        try:
            opt_model.optical_spec.spectral_region = WvlSpec(wvl_values, ref_wl=1)
            if verbosity >= 2:
                print(f"Spectral region set with wvl_values: {wvl_values}")
        except Exception as e:
            if verbosity >= 2:
                print(f"Error setting spectral region with wvl_values={wvl_values}: {str(e)}")
            return [None] * len(chunk)
        
        # Trace the rays
        try:
            result = analyses.trace_list_of_rays(
                opt_model,
                chunk,
                output_filter="last",
                rayerr_filter="summary"
            )
            if verbosity >= 2:
                print(f"Successfully traced {len(result)} rays")
            return result
        except Exception as e:
            if verbosity >= 2:
                print(f"Error tracing rays: {str(e)}")
            return [None] * len(chunk)
            
    except Exception as e:
        if verbosity >= 2:
            print(f"Unexpected error in _process_ray_chunk_standalone: {str(e)}")
        return [None] * len(chunk)

class Lens:
    """
    Lens defining object with integrated data management.
    """
    def __init__(self, archive: str = None, data: "pd.DataFrame" = None,
                kind: str = "nikkor_58mm", zfine: float = 12.75, zmx_file: str = None,
                focus_gaps: List[Tuple[int, float]] = None, dist_from_obj: float = None,
                gap_between_lenses: float = 15.0, dist_to_screen: float = 20.0, fnumber: float = 8.0,
                FOV: float = None, magnification: float = None,
                empir_dirpath: str = None,
                verbosity: VerbosityLevel = VerbosityLevel.BASIC):
        """
        Initialize a Lens object with optical model and data management.

        Args:
            archive (str, optional): Directory path for saving results.
            data (pd.DataFrame, optional): Optical photon data table.
            kind (str, optional): Lens type ('nikkor_58mm', 'microscope', 'zmx_file'). Defaults to 'nikkor_58mm'.
            zfine (float, optional): Initial zfine adjustment in mm relative to default settings.
            zmx_file (str, optional): Path to .zmx file for custom lens (required when kind='zmx_file').
            focus_gaps (List[Tuple[int, float]], optional): List of (gap_index, scaling_factor) for focus adjustment.
                Required for focus adjustments when using zmx_file.
            dist_from_obj (float, optional): Distance from object to first lens in mm.
                Defaults to lens-specific values: nikkor_58mm=461.535, microscope=41.0, zmx_file=100.0.
            gap_between_lenses (float, optional): Gap between lenses in mm. Defaults to 15.0.
            dist_to_screen (float, optional): Distance from last lens to screen in mm. Defaults to 20.0.
            fnumber (float, optional): F-number of the optical system. Defaults to 8.0.
            FOV (float, optional): Field of view in mm. Defaults to None. for 'nikor_58mm', FOV=120mm and for 'microscope', FOV=10mm, for 'zmx_file', FOV=60mm.
            magnification (float, optional): Manually define magnification. Defaults to None.
            verbosity (VerbosityLevel, optional): Verbosity level for logging. Defaults to VerbosityLevel.BASIC.
        Raises:
            ValueError: If invalid lens kind, missing zmx_file for 'zmx_file', or invalid parameters.

        Example:
            >>> # Simple usage - kind is auto-detected
            >>> lens = Lens(archive="path/to/archive", zmx_file="my_lens.zmx")
            >>>
            >>> # With focus gaps for refocusing
            >>> lens = Lens(archive="path/to/archive",
            ...             zmx_file="my_lens.zmx",
            ...             focus_gaps=[(12, 1.0)])
        """

        # Auto-detect kind from zmx_file parameter
        if zmx_file is not None and kind == "nikkor_58mm":  # Only auto-detect if kind is default
            kind = "zmx_file"

        self.kind = kind
        self.zfine = zfine
        self.zmx_file = zmx_file
        self.focus_gaps = focus_gaps
        self.FOV = FOV

        # Validate inputs
        if kind == "zmx_file" and zmx_file is None:
            raise ValueError("zmx_file must be provided when kind='zmx_file'")

        # Set default parameters based on lens kind
        if kind == "nikkor_58mm":
            self.dist_from_obj = dist_from_obj if dist_from_obj else 461.535
            self.gap_between_lenses = 0.0
            self.dist_to_screen = 0.0
            self.fnumber = fnumber if fnumber != 8.0 else 0.98
            self.default_focus_gaps = [(22, 2.68)]
            if self.FOV is None:
                self.FOV = 120.0
        elif kind == "microscope":
            self.dist_from_obj = dist_from_obj if dist_from_obj else 41.0
            self.gap_between_lenses = gap_between_lenses
            self.dist_to_screen = dist_to_screen
            self.fnumber = fnumber
            self.default_focus_gaps = [(24, None), (31, None)]
            if self.FOV is None:
                self.FOV = 10.0
        elif kind == "zmx_file":
            # Provide sensible defaults for zmx_file to prevent None values
            self.dist_from_obj = dist_from_obj if dist_from_obj is not None else 100.0  # Default distance
            self.gap_between_lenses = gap_between_lenses
            self.dist_to_screen = dist_to_screen
            self.fnumber = fnumber
            self.default_focus_gaps = focus_gaps or []
            if focus_gaps is None:
                print("Warning: focus_gaps not provided for zmx_file; focus adjustment will have no effect")
        else:
            raise ValueError(f"Unknown lens kind: {kind}, supported lenses are ['nikkor_58mm', 'microscope', 'zmx_file']")

        if archive is not None:
            self.archive = Path(archive)
            self.archive.mkdir(parents=True, exist_ok=True)

            sim_photons_dir = self.archive / "SimPhotons"
            csv_files = sorted(sim_photons_dir.glob("sim_data_*.csv"))

            valid_dfs = []
            for file in tqdm(csv_files, desc="Loading simulation data"):
                try:
                    if file.stat().st_size > 100:
                        df = pd.read_csv(file)
                        if not df.empty:
                            valid_dfs.append(df)
                except Exception as e:
                    print(f"⚠️ Skipping {file.name} due to error: {e}")
                    pass

            if valid_dfs:
                self.data = pd.concat(valid_dfs, ignore_index=True)
            else:
                print("No valid simulation data files found, initializing empty DataFrame.")
                self.data = pd.DataFrame()

        elif data is not None:
            self.data = data
            self.archive = Path("archive/test")
        else:
            raise ValueError("Either archive or data must be provided")

        # Initialize optical models
        self.opm0 = None
        self.opm = None
        if self.kind == "nikkor_58mm":
            self.opm0 = self.nikkor_58mm(dist_from_obj=self.dist_from_obj, fnumber=self.fnumber, save=False)
            self.opm = deepcopy(self.opm0)
            if zfine is not None:
                self.opm = self.refocus(zfine=zfine, save=False)
        elif self.kind == "microscope":
            self.opm0 = self.microscope_nikor_80_200mm_canon_50mm(focus=zfine or 0.0, save=False)
            self.opm = deepcopy(self.opm0)
            if zfine is not None:
                self.opm = self.refocus(zfine=zfine, save=False)
        elif self.kind == "zmx_file":
            self.opm0 = self.load_zmx_lens(zmx_file, focus=zfine, save=False)
            self.opm = deepcopy(self.opm0)
            if zfine is not None:
                self.opm = self.refocus(zfine=zfine, save=False)
        else:
            raise ValueError(f"Unknown lens kind: {self.kind}")

        # get the multiplication value for converting from mm to pixels
        if magnification is not None:
            self.reduction_ratio = magnification
        else:
            self.reduction_ratio = self.get_first_order_parameters().loc["Reduction Ratio","Value"]

        try:
            if empir_dirpath is not None:
                self.empir_dirpath = Path(empir_dirpath)
            else:
                try:
                    from G4LumaCam.config.paths import EMPIR_PATH
                    self.empir_dirpath = Path(EMPIR_PATH)
                except ImportError:
                    self.empir_dirpath = Path("./empir")

            # if not self.empir_dirpath.exists():
            #     raise FileNotFoundError(f"{self.empir_dirpath} does not exist.")
            
            required_files = {
                "empir_import_photons": "empir_import_photons",
            }
            
            self.executables = {}
            for attr_name, filename in required_files.items():
                file_path = self.empir_dirpath / filename
                if not file_path.exists():
                    raise FileNotFoundError(f"{filename} not found in {self.empir_dirpath}")
                self.executables[attr_name] = file_path
                setattr(self, attr_name, file_path)
        except Exception as e:
            if verbosity >= VerbosityLevel.BASIC:
                print(f"⚠️ Warning: Could not set empir_dirpath or find required files: {e}")


    def get_first_order_parameters(self, opm: "OpticalModel" = None) -> pd.DataFrame:
        """
        Calculate first-order optical parameters and return them as a DataFrame.

        Args:
            opm (OpticalModel, optional): Optical model to analyze. Defaults to self.opm0.

        Returns:
            pd.DataFrame: DataFrame with first-order parameters and user-friendly names.

        Raises:
            RuntimeError: If parameters cannot be retrieved.
        """
        if opm is None:
            opm = self.opm0
        pm = opm['parax_model']

        output = StringIO()
        with redirect_stdout(output):
            pm.first_order_data()
        output_str = output.getvalue()
        
        fod = {}
        multi_word_keys = ['pp sep', 'na obj', 'n obj', 'na img', 'n img', 'optical invariant']
        lines = output_str.strip().split('\n')
        for line in lines:
            try:
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    continue
                key, value = parts
                for mw_key in multi_word_keys:
                    if line.startswith(mw_key):
                        key = mw_key
                        value = line[len(mw_key):].strip()
                        break
                try:
                    fod[key] = float(value)
                except ValueError:
                    fod[key] = value
            except ValueError:
                continue
        
        if not fod:
            fod = {}
            try:
                fod['efl'] = pm.efl if hasattr(pm, 'efl') else float('nan')
                fod['f'] = pm.f if hasattr(pm, 'f') else float('nan')
                fod['f\''] = pm.f_prime if hasattr(pm, 'f_prime') else float('nan')
                fod['ffl'] = pm.ffl if hasattr(pm, 'ffl') else float('nan')
                fod['pp1'] = pm.pp1 if hasattr(pm, 'pp1') else float('nan')
                fod['bfl'] = pm.bfl if hasattr(pm, 'bfl') else float('nan')
                fod['ppk'] = pm.ppk if hasattr(pm, 'ppk') else float('nan')
                fod['pp sep'] = pm.pp_sep if hasattr(pm, 'pp_sep') else float('nan')
                fod['f/#'] = pm.f_number if hasattr(pm, 'f_number') else opm['optical_spec'].pupil.value
                fod['m'] = pm.magnification if hasattr(pm, 'magnification') else float('nan')
                fod['red'] = pm.reduction if hasattr(pm, 'reduction') else float('nan')
                fod['obj_dist'] = pm.obj_dist if hasattr(pm, 'obj_dist') else opm.seq_model.gaps[0].thi
                fod['obj_ang'] = pm.obj_angle if hasattr(pm, 'obj_angle') else opm['optical_spec'].field_of_view.flds[-1]
                fod['enp_dist'] = pm.enp_dist if hasattr(pm, 'enp_dist') else float('nan')
                fod['enp_radius'] = pm.enp_radius if hasattr(pm, 'enp_radius') else float('nan')
                fod['na obj'] = pm.na_obj if hasattr(pm, 'na_obj') else float('nan')
                fod['n obj'] = pm.n_obj if hasattr(pm, 'n_obj') else 1.0
                fod['img_dist'] = pm.img_dist if hasattr(pm, 'img_dist') else float('nan')
                fod['img_ht'] = pm.img_height if hasattr(pm, 'img_height') else float('nan')
                fod['exp_dist'] = pm.exp_dist if hasattr(pm, 'exp_dist') else float('nan')
                fod['exp_radius'] = pm.exp_radius if hasattr(pm, 'exp_radius') else float('nan')
                fod['na img'] = pm.na_img if hasattr(pm, 'na_img') else float('nan')
                fod['n img'] = pm.n_img if hasattr(pm, 'n_img') else 1.0
                fod['optical invariant'] = pm.opt_inv if hasattr(pm, 'opt_inv') else float('nan')
            except Exception as e:
                raise RuntimeError(f"Failed to retrieve first-order parameters: {e}")

        param_names = {
            'efl': 'Effective Focal Length (mm)',
            'f': 'Focal Length (mm)',
            'f\'': 'Back Focal Length (mm)',
            'ffl': 'Front Focal Length (mm)',
            'pp1': 'Front Principal Point (mm)',
            'bfl': 'Back Focal Length to Image (mm)',
            'ppk': 'Back Principal Point (mm)',
            'pp sep': 'Principal Plane Separation (mm)',
            'f/#': 'F-Number',
            'm': 'Magnification',
            'red': 'Reduction Ratio',
            'obj_dist': 'Object Distance (mm)',
            'obj_ang': 'Object Field Angle (degrees)',
            'enp_dist': 'Entrance Pupil Distance (mm)',
            'enp_radius': 'Entrance Pupil Radius (mm)',
            'na obj': 'Object Numerical Aperture',
            'n obj': 'Object Space Refractive Index',
            'img_dist': 'Image Distance (mm)',
            'img_ht': 'Image Height (mm)',
            'exp_dist': 'Exit Pupil Distance (mm)',
            'exp_radius': 'Exit Pupil Radius (mm)',
            'na img': 'Image Numerical Aperture',
            'n img': 'Image Space Refrictive Index',
            'optical invariant': 'Optical Invariant'
        }
        
        df = pd.DataFrame.from_dict(fod, orient='index', columns=['Value'])
        df['Original Name'] = df.index
        df.index = [param_names.get(idx, idx) for idx in df.index]
        df = df[['Original Name', 'Value']]
        return df

    def load_zmx_lens(self, zmx_file: str, focus: float = None, dist_from_obj: float = None,
                      gap_between_lenses: float = None, dist_to_screen: float = None,
                      fnumber: float = None, save: bool = False) -> OpticalModel:
        """
        Load a lens from a .zmx file.

        Args:
            zmx_file (str): Path to the .zmx file.
            focus (float, optional): Initial focus adjustment in mm relative to default settings.
            dist_from_obj (float, optional): Distance from object to first lens in mm.
            gap_between_lenses (float, optional): Gap between lenses in mm.
            dist_to_screen (float, optional): Distance from last lens to screen in mm.
            fnumber (float, optional): F-number of the optical system.
            save (bool): Save the optical model to a file.

        Returns:
            OpticalModel: The loaded optical model.

        Raises:
            FileNotFoundError: If the .zmx file does not exist.
        """
        if not Path(zmx_file).exists():
            raise FileNotFoundError(f".zmx file not found: {zmx_file}")

        opm = OpticalModel()
        sm = opm.seq_model
        osp = opm.optical_spec
        opm.system_spec.title = f'Custom Lens from {Path(zmx_file).name}'
        opm.system_spec.dimensions = 'MM'
        opm.radius_mode = True
        sm.gaps[0].thi = dist_from_obj if dist_from_obj is not None else self.dist_from_obj
        osp.pupil = PupilSpec(osp, key=['image', 'f/#'], value=fnumber if fnumber is not None else self.fnumber)
        sm.do_apertures = False
        opm.add_from_file(zmx_file, t=gap_between_lenses if gap_between_lenses is not None else self.gap_between_lenses)
        if dist_to_screen is not None:
            sm.gaps[-1].thi = dist_to_screen
        elif self.dist_to_screen != 0.0:
            sm.gaps[-1].thi = self.dist_to_screen
        opm.update_model()
        
        if focus is not None and self.focus_gaps is not None:
            opm = self.refocus(opm=opm, zfine=focus, save=False)
        
        if save:
            output_path = self.archive / f"Custom_Lens_{Path(zmx_file).stem}.roa"
            opm.save_model(str(output_path))
        return opm

    def microscope_nikor_80_200mm_canon_50mm(self, focus: float = 0.0, dist_from_obj: float = 41.0,
                                             gap_between_lenses: float = 15.0, dist_to_screen: float = 20.0,
                                             fnumber: float = 8.0, save: bool = False) -> OpticalModel:
        """
        Create a microscope lens model with Nikkor 80-200mm f/2.8 and flipped Canon 50mm f/1.8 lenses.

        Args:
            focus (float): Focus adjustment in mm relative to default settings (gap 24 increases, gap 31 decreases). Defaults to 0.0.
            dist_from_obj (float): Distance from object to first lens in mm. Defaults to 35.0.
            gap_between_lenses (float): Gap between the two lenses in mm. Defaults to 15.0.
            dist_to_screen (float): Distance from second lens to screen in mm. Defaults to 20.0.
            fnumber (float): F-number of the optical system. Defaults to 8.0.
            save (bool): Save the optical model to a file.

        Returns:
            OpticalModel: The configured microscope optical model.

        Raises:
            ValueError: If parameters are invalid.
            FileNotFoundError: If .zmx files are not found.
        """
        if dist_from_obj <= 0:
            raise ValueError(f"dist_from_obj must be positive, got {dist_from_obj}")
        if gap_between_lenses < 0:
            raise ValueError(f"gap_between_lenses cannot be negative, got {gap_between_lenses}")
        if dist_to_screen < 0:
            raise ValueError(f"dist_to_screen cannot be negative, got {dist_to_screen}")
        if fnumber <= 0:
            raise ValueError(f"fnumber must be positive, got {fnumber}")

        opm = OpticalModel()
        sm = opm.seq_model
        osp = opm.optical_spec
        opm.system_spec.title = 'Microscope Lens Model'
        opm.system_spec.dimensions = 'MM'
        opm.radius_mode = True

        sm.gaps[0].thi = dist_from_obj
        osp.pupil = PupilSpec(osp, key=['object', 'f/#'], value=fnumber)
        osp.field_of_view = FieldSpec(osp, key=['object', 'height'], flds=[0., 1])  # Set field of view
        osp.spectral_region = WvlSpec([(486.1327, 0.5), (587.5618, 1.0), (656.2725, 0.5)], ref_wl=1)
        sm.do_apertures = False
        opm.update_model()

        package = 'lumacam.data'
        zmx_files = [
            'JP1985-040604_Example01P_50mm_1.2f.zmx',
            'JP2000-019398_Example01_Tale67_80_200_AF-S_2.4f.zmx',
        ]

        with importlib.resources.as_file(importlib.resources.files(package).joinpath(zmx_files[0])) as zmx_path:
            if not zmx_path.exists():
                raise FileNotFoundError(f".zmx file not found: {zmx_path}")
            opm.add_from_file(str(zmx_path), t=gap_between_lenses)

        with importlib.resources.as_file(importlib.resources.files(package).joinpath(zmx_files[1])) as zmx_path:
            if not zmx_path.exists():
                raise FileNotFoundError(f".zmx file not found: {zmx_path}")
            opm.add_from_file(str(zmx_path), t=dist_to_screen)

        opm.flip(1, 15)
        
        # Store default gap thicknesses for microscope
        self.default_focus_gaps = [(24, sm.gaps[24].thi), (31, sm.gaps[31].thi)]
        opm = self.refocus(opm=opm, zfine=focus, save=False)
        opm.update_model()
        self.opm0 = deepcopy(opm)
        

        if save:
            output_path = self.archive / "Microscope_Lens.roa"
            opm.save_model(str(output_path))
        return opm

    def nikkor_58mm(self, dist_from_obj: float = 461.535, fnumber: float = 0.98, save: bool = False) -> OpticalModel:
        """
        Create a Nikkor 58mm f/0.95 lens model from a .zmx file, correcting specific thicknesses to match the original model.

        Args:
            dist_from_obj (float): Distance from object to first lens in mm. Defaults to 461.535.
            fnumber (float): F-number of the optical system. Defaults to 0.98.
            save (bool): Save the optical model to a file.

        Returns:
            OpticalModel: The configured Nikkor 58mm optical model.

        Raises:
            FileNotFoundError: If the .zmx file is not found.
            ValueError: If the model has insufficient surfaces for correction.
        """
        zmx_path = str(importlib.resources.files('lumacam.data').joinpath('WO2019-229849_Example01P.zmx'))
        if not Path(zmx_path).exists():
            raise FileNotFoundError(f".zmx file not found: {zmx_path}")

        opm = OpticalModel()
        sm = opm.seq_model
        osp = opm.optical_spec
        opm.system_spec.title = 'WO2019-229849 Example 1 (Nikkor Z 58mm f/0.95 S)'
        opm.system_spec.dimensions = 'MM'
        opm.radius_mode = True

        # Load the .zmx file
        sm.gaps[0].thi = dist_from_obj
        osp.pupil = PupilSpec(osp, key=['object', 'f/#'], value=fnumber)
        osp.field_of_view = FieldSpec(osp, key=['object', 'height'], flds=[0., 60])
        osp.spectral_region = WvlSpec([(486.1327, 0.5), (587.5618, 1.0), (656.2725, 0.5)], ref_wl=1)
        sm.do_apertures = False
        opm.add_from_file(zmx_path)
        opm.update_model()

        # Correct specific thicknesses
        if len(sm.gaps) <= 30:
            raise ValueError(f"Insufficient gaps in .zmx file: {len(sm.gaps)} found, expected at least 31")

        # Correct surface 22 thickness (from 21.2900 mm to 2.68000 mm)
        if abs(sm.gaps[22].thi - 2.68) > 1e-6:
            sm.gaps[22].thi = 2.68

        # Correct surface 30 thickness (from 0.00000 mm to 1.00000 mm)
        if abs(sm.gaps[30].thi - 1.0) > 1e-6:
            sm.gaps[30].thi = 1.0

        opm.update_model()
        apply_paraxial_vignetting(opm)

        # Verify corrections
        if abs(sm.gaps[22].thi - 2.68) > 1e-6:
            print(f"Warning: Surface 22 thickness {sm.gaps[22].thi:.6f} does not match expected 2.68000 mm")
        if abs(sm.gaps[30].thi - 1.0) > 1e-6:
            print(f"Warning: Surface 30 thickness {sm.gaps[30].thi:.6f} does not match expected 1.00000 mm")

        if save:
            output_path = self.archive / "Nikkor_58mm.roa"
            opm.save_model(str(output_path))

        return opm

    def refocus(self, opm: "OpticalModel" = None, zscan: float = 0, zfine: float = 12.75, fnumber: float = None, save: bool = False) -> OpticalModel:
        """
        Refocus the lens by adjusting gaps relative to default settings.

        Args:
            opm (OpticalModel, optional): Optical model to refocus. Defaults to self.opm0.
            zscan (float): Distance to move the lens assembly in mm relative to default object distance. Defaults to 0.
            zfine (float): Focus adjustment in mm relative to default gap thicknesses (for microscope, gap 24 increases, gap 31 decreases). Defaults to 12.75
            fnumber (float, optional): New f-number for the lens.
            save (bool): Save the optical model to a file.

        Returns:
            OpticalModel: The refocused optical model.

        Raises:
            ValueError: If lens kind is unsupported or gap indices are invalid.
        """
        opm = deepcopy(self.opm0) if opm is None else deepcopy(opm)
        sm = opm.seq_model
        osp = opm.optical_spec
        
        if self.kind == "nikkor_58mm":
            if not self.default_focus_gaps:
                raise ValueError("Default focus gaps not set for nikkor_58mm")
            gap_index, default_thi = self.default_focus_gaps[0]
            if gap_index >= len(sm.gaps):
                raise ValueError(f"Invalid gap index {gap_index} for nikkor_58mm lens")
            if zfine != 0:
                new_thi = default_thi + zfine
                sm.gaps[gap_index].thi = new_thi
            sm.gaps[0].thi = self.dist_from_obj + zscan
            
        elif self.kind == "microscope":
            if len(self.default_focus_gaps) != 2:
                raise ValueError("Default focus gaps not set correctly for microscope")
            if zfine != 0:
                gap_index_24, default_thi_24 = self.default_focus_gaps[0]
                if gap_index_24 >= len(sm.gaps):
                    raise ValueError(f"Invalid gap index {gap_index_24} for microscope lens")
                if default_thi_24 is None:
                    raise ValueError(f"Default thickness not set for gap {gap_index_24}")
                new_thi_24 = default_thi_24 + zfine
                sm.gaps[gap_index_24].thi = new_thi_24

                gap_index_31, default_thi_31 = self.default_focus_gaps[1]
                if gap_index_31 >= len(sm.gaps):
                    raise ValueError(f"Invalid gap index {gap_index_31} for microscope lens")
                if default_thi_31 is None:
                    raise ValueError(f"Default thickness not set for gap {gap_index_31}")
                new_thi_31 = default_thi_31 - zfine
                sm.gaps[gap_index_31].thi = new_thi_31
            sm.gaps[0].thi = self.dist_from_obj + zscan
            
        elif self.kind == "zmx_file":
            if zfine != 0 and self.focus_gaps is not None:
                for gap_index, scaling_factor in self.focus_gaps:
                    if gap_index >= len(sm.gaps):
                        raise ValueError(f"Invalid gap index {gap_index} for zmx_file lens")
                    default_thi = sm.gaps[gap_index].thi
                    new_thi = default_thi + zfine * scaling_factor
                    sm.gaps[gap_index].thi = new_thi
            sm.gaps[0].thi = self.dist_from_obj + zscan
            
        else:
            raise ValueError(f"Unsupported lens kind: {self.kind}")
        
        if fnumber is not None:
            osp.pupil = PupilSpec(osp, key=['image', 'f/#'], value=fnumber)
        
        sm.do_apertures = False
        opm.update_model()
        apply_paraxial_vignetting(opm)
        
        self.opm = opm
        
        if save:
            fnumber_str = f"_f{fnumber:.2f}" if fnumber is not None else ""
            save_path = self.archive / f"refocus_zscan_{zscan}_zfine_{zfine}{fnumber_str}.roa"
            opm.save_model(save_path)
        
        return opm

    def _chunk_rays(self, rays, chunk_size):
        """
        Split rays into chunks for parallel processing while preserving ray identifiers.
        
        Parameters:
        -----------
        rays : list
            List of ray tuples: (position, direction, wavelength)
        chunk_size : int
            Number of rays per chunk
            
        Returns:
        --------
        list
            List of chunks, where each chunk is a list of ray tuples
        """
        return [rays[i:i+chunk_size] for i in range(0, len(rays), chunk_size)]


    def trace_rays(self, opm=None, opm_file=None, zscan=0, zfine=12.75, fnumber=None,
                    source=None, deadtime=None, blob=0.0, blob_variance=0.0, decay_time=10,
                    detector_model: Union[str, DetectorModel] = None,
                    model_params: dict = None,
                    join=False, print_stats=False, n_processes=None, chunk_size=1000,
                    progress_bar=True, timeout=3600, return_df=False, split_method="auto",
                    seed: int = None,
                    verbosity=VerbosityLevel.BASIC,
                    **kwargs  # Additional model parameters passed as kwargs
                    ) -> Optional[pd.DataFrame]:
        """
        Trace rays from simulation data files and save processed results, optionally applying pixel saturation and blob effect.
        
        If groupby() was called prior to this method, automatically performs grouped tracing for each group.
        Otherwise, performs standard single-archive tracing.

        This method processes ray data from CSV files in the 'SimPhotons' directory using either
        a provided optical model, a saved optical model file, or by creating a refocused version
        of the default optical model.
        
        The method supports two output workflows controlled by the 'source' parameter:
        1. "hits" workflow: Applies saturation (if deadtime/blob provided), generates TPX3 files
        2. "photons" workflow: Saves traced photons directly without saturation, ready for direct import

        Parameters:
        -----------
        opm : OpticalModel, optional
            Custom optical model to use. If provided, will be saved to a temporary file
            for multiprocessing compatibility. Cannot be used with opm_file.
        opm_file : str or Path, optional  
            Path to a saved optical model file (.roa). If provided, this model will be
            loaded in each worker process. Cannot be used with opm.
        zscan : float, default 0
            Distance to move the lens assembly in mm relative to default object distance.
            Only used if neither opm nor opm_file is provided.
        zfine : float, default 12.75
            Focus adjustment in mm relative to default gap thicknesses. Only used if 
            neither opm nor opm_file is provided.
        fnumber : float, optional
            New f-number for the lens. Applied to refocused model if neither opm nor
            opm_file is provided.
        source : str, optional
            Output workflow mode:
            - None (default): Auto-detect based on deadtime/blob parameters
            - If deadtime > 0 or blob > 0: uses "hits" workflow
            - Otherwise: uses "photons" workflow
            - "hits": Apply saturation and generate TPX3 files (requires deadtime and/or blob)
            - "photons": Save traced photons directly without saturation for direct import
        deadtime : float, optional
            Deadtime in nanoseconds for pixel saturation. Only used in "hits" workflow.
            If source=None and deadtime > 0, automatically selects "hits" workflow.
        blob : float, default 0.0
            Interaction radius in pixels for photon hits. If > 0, each photon hit affects
            all pixels within this radius. Only used in "hits" workflow.
            If source=None and blob > 0, automatically selects "hits" workflow.
        blob_variance : float, default 0.0
            Radius value subtracted from blob radius (in pixels). Only used if blob > 0.
            Each photon's actual blob radius is drawn uniformly from [blob - blob_variance, blob].
            Example: blob=5, blob_variance=2.5 → radius uniformly distributed in [2.5, 5.0] pixels.
        decay_time : float, default 100.0
            Exponential decay time constant in nanoseconds for blob activation timing.
            Single delay drawn per photon, applied to all pixels in its blob.
        detector_model : str or DetectorModel, default "image_intensifier"
            Physical model for photon detection (only used in "hits" workflow). Options:
            - "image_intensifier": MCP intensifier with circular blobs (default)
            - "gaussian_diffusion": Charge diffusion with Gaussian PSF
            - "direct_detection": Simple single-pixel detection
            - "wavelength_dependent": Spectral QE and wavelength-dependent gain
            - "avalanche_gain": Poisson gain with afterpulsing
        model_params : dict, optional
            Model-specific parameters (only used in "hits" workflow). Examples:
            - gaussian_diffusion: {'charge_coupling': 0.8}
            - wavelength_dependent: {'qe_wavelength': [400,500,600], 'qe_values': [0.1,0.3,0.2]}
            - avalanche_gain: {'mean_gain': 100, 'afterpulse_prob': 0.01}
        seed : int, optional
            Random seed for reproducibility. If None, uses random state. If specified, allows
            exact reconstruction of results across multiple runs.
        join : bool, default False
            If True, concatenates original simulation data with traced results.
            If False, returns only traced positions and identifiers.
        print_stats : bool, default False
            If True, prints detailed tracing statistics for each processed file
        n_processes : int, optional
            Number of processes for parallel execution. If None, uses CPU count.
            Set to 1 to disable multiprocessing for debugging.
        chunk_size : int, default 1000
            Number of rays per processing chunk. Larger chunks use more memory but
            may be more efficient. Smaller chunks provide better progress tracking.
        progress_bar : bool, default True
            If True, displays progress bars during file and chunk processing
        timeout : int, default 3600
            Maximum time in seconds for processing each file (currently unused)
        return_df : bool, default False
            If True, returns a combined DataFrame of all processed files.
            If False, returns None (files are still saved to disk).
        split_method : str, default "auto"
            TPX3 file splitting strategy (only used in "hits" workflow):
            - "auto": Groups neutron events to minimize file count (default)
            - "event": Creates one TPX3 file per neutron_id for event-by-event analysis
        verbosity : VerbosityLevel, default VerbosityLevel.BASIC
            Controls output detail level:
            - QUIET (0): Only essential error messages
            - BASIC (1): Progress bars + basic file info + statistics
            - DETAILED (2): All available information including warnings

        Returns:
        --------
        pd.DataFrame or None
            Combined DataFrame of all processed results if return_df=True, 
            otherwise None. Each row represents a traced ray.
        
        Raises:
        -------
        ValueError: If both opm and opm_file are provided, if parameters are invalid,
                    or if source="hits" but neither deadtime nor blob is provided.
        FileNotFoundError: If opm_file does not exist or if no valid simulation
                            data files are found.
        RuntimeError: If tracing fails for a file.
        
        Examples:
        ---------
        Hits workflow with saturation (auto-detected):
        >>> optics.trace_rays(deadtime=100, blob=5.0)  # Auto-detects "hits" workflow

        Hits workflow with Gaussian diffusion model:
        >>> optics.trace_rays(deadtime=100, blob=2.0, detector_model="gaussian_diffusion",
        ...                   model_params={'charge_coupling': 0.85})

        Direct detection (no blob):
        >>> optics.trace_rays(deadtime=300, detector_model="direct_detection")

        Wavelength-dependent QE:
        >>> optics.trace_rays(deadtime=600, blob=2.0, detector_model="wavelength_dependent",
        ...                   model_params={'qe_wavelength': [400,500,600],
        ...                                 'qe_values': [0.1,0.3,0.2]})

        Photons workflow (auto-detected):
        >>> optics.trace_rays()  # Auto-detects "photons" workflow (no saturation)

        Photons workflow with explicit source:
        >>> optics.trace_rays(source="photons")  # Direct import, no saturation
        """
        # Auto-detect source if not specified
        if source is None:
            if deadtime is not None and deadtime > 0:
                source = "hits"
            elif blob > 0:
                source = "hits"
            else:
                source = "photons"
            
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Auto-detected source: '{source}'")
        
        # Validate source parameter
        if source not in ["hits", "photons"]:
            raise ValueError(f"Invalid source: '{source}'. Must be 'hits' or 'photons'")
        
        # Validate hits workflow requirements
        if source == "hits":
            if (deadtime is None or deadtime <= 0) and blob <= 0:
                raise ValueError(
                    "source='hits' requires either deadtime > 0 or blob > 0 for saturation. "
                    "Use source='photons' for direct import without saturation."
                )
        
        # Check if groupby was called - if so, delegate to grouped tracing
        if hasattr(self, '_groupby_dir') and hasattr(self, '_groupby_labels'):
            return self._trace_rays_grouped(
                opm=opm,
                opm_file=opm_file,
                zscan=zscan,
                zfine=zfine,
                fnumber=fnumber,
                source=source,
                deadtime=deadtime,
                blob=blob,
                blob_variance=blob_variance,
                decay_time=decay_time,
                seed=seed,
                join=join,
                print_stats=print_stats,
                n_processes=n_processes,
                chunk_size=chunk_size,
                progress_bar=progress_bar,
                timeout=timeout,
                return_df=return_df,
                split_method=split_method,
                verbosity=verbosity
            )
        
        # Otherwise, perform standard single-archive tracing
        return self._trace_rays_single(
            opm=opm,
            opm_file=opm_file,
            zscan=zscan,
            zfine=zfine,
            fnumber=fnumber,
            source=source,
            deadtime=deadtime,
            blob=blob,
            blob_variance=blob_variance,
            decay_time=decay_time,        
            seed=seed,    
            join=join,
            print_stats=print_stats,
            n_processes=n_processes,
            chunk_size=chunk_size,
            progress_bar=progress_bar,
            timeout=timeout,
            return_df=return_df,
            split_method=split_method,        
            verbosity=verbosity
        )


    def _trace_rays_single(self, opm=None, opm_file=None, zscan=0, zfine=0, fnumber=None,
                        source="photons", deadtime=None, blob=0.0, blob_variance=0.0, decay_time=100, 
                        seed: int = None, join=False, print_stats=False, n_processes=None, chunk_size=1000, 
                        progress_bar=True, timeout=3600, return_df=False, 
                        split_method="auto",
                        verbosity=VerbosityLevel.BASIC,  
                        ) -> pd.DataFrame or None:
        """
        Internal method for single-archive ray tracing (non-grouped).
        See trace_rays() for full documentation.

        Parameters:
        -----------
        (Same as trace_rays)
        
        Returns:
        --------
        pd.DataFrame or None
            Combined DataFrame of all processed results if return_df=True,
            otherwise None. Each row represents a traced ray.
        """
        # Validate input parameters
        if opm is not None and opm_file is not None:
            raise ValueError("Cannot specify both 'opm' and 'opm_file' parameters. Choose one.")
        
        if opm_file is not None and not Path(opm_file).exists():
            raise FileNotFoundError(f"Optical model file not found: {opm_file}")
        
        if source not in ["hits", "photons"]:
            raise ValueError(f"Invalid source: '{source}'. Must be 'hits' or 'photons'")
        
        if source == "hits":
            if (deadtime is None or deadtime <= 0) and blob <= 0:
                raise ValueError("source='hits' requires either deadtime > 0 or blob > 0")
        
        if deadtime is not None and deadtime <= 0:
            raise ValueError(f"deadtime must be positive, got {deadtime}")
        
        if blob < 0:
            raise ValueError(f"blob must be non-negative, got {blob}")

        # Set up directories
        sim_photons_dir = self.archive / "SimPhotons"
        traced_photons_dir = self.archive / "TracedPhotons"
        traced_photons_dir.mkdir(parents=True, exist_ok=True)

        # Find all non-empty sim_data_*.csv files
        csv_files = sorted(sim_photons_dir.glob("sim_data_*.csv"))
        valid_files = [f for f in csv_files if f.stat().st_size > 100]

        if not valid_files:
            if verbosity > VerbosityLevel.BASIC:
                print("No valid simulation data files found in 'SimPhotons' directory.")
                print(f"Searched in: {sim_photons_dir}")
                print("Expected files matching pattern: sim_data_*.csv")
            return None

        if verbosity > VerbosityLevel.BASIC:
            print(f"Found {len(valid_files)} valid simulation files to process")
            print(f"Workflow: {source}")

        # Handle optical model setup
        temp_opm_file = None
        opm_file_path = None
        
        try:
            if opm is not None:
                # Save provided OPM to temporary file for multiprocessing
                temp_dir = self.archive / "TempOpm"
                temp_dir.mkdir(exist_ok=True)
                temp_opm_file = temp_dir / "temp_opm_for_tracing.roa"
                opm.save_model(str(temp_opm_file))
                opm_file_path = str(temp_opm_file)
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Saved temporary OPM to: {temp_opm_file}")
                    
            elif opm_file is not None:
                # Use provided file path
                opm_file_path = str(opm_file)
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Using OPM file: {opm_file_path}")
                    
            else:
                # Create refocused OPM and save to temporary file
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Creating refocused OPM with zscan={zscan}, zfine={zfine}, fnumber={fnumber}")
                
                refocused_opm = self.refocus(zscan=zscan, zfine=zfine, fnumber=fnumber, save=False)
                
                temp_dir = self.archive / "TempOpm" 
                temp_dir.mkdir(exist_ok=True)
                filename = f"temp_refocus_zscan_{zscan}_zfine_{zfine}"
                if fnumber is not None:
                    filename += f"_f{fnumber:.2f}"
                filename += ".roa"
                temp_opm_file = temp_dir / filename
                refocused_opm.save_model(str(temp_opm_file))
                opm_file_path = str(temp_opm_file)
                
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Saved refocused OPM to: {temp_opm_file}")

            all_results = []

            # Progress bar for file processing
            file_desc = f"Processing {len(valid_files)} files"
            file_iter = tqdm(valid_files, desc=file_desc, 
                            disable=not progress_bar or verbosity == VerbosityLevel.QUIET)
            
            for file_idx, csv_file in enumerate(file_iter):
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"\nProcessing file: {csv_file.name}")

                # Load and validate data
                try:
                    df = pd.read_csv(csv_file)
                except Exception as e:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"Error reading {csv_file.name}: {str(e)}")
                    continue
                    
                if df.empty:
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Skipping empty file: {csv_file.name}")
                    continue

                # Validate required columns
                required_cols = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'wavelength']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"Skipping {csv_file.name}: missing columns {missing_cols}")
                    continue

                # Check for pulse_id when split_method="event"
                if source == "hits" and split_method == "event" and 'pulse_id' not in df.columns:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"Warning: {csv_file.name} missing pulse_id column required for split_method='event'")

                # Validate nz and pz columns
                nz_pz_cols = ['nz', 'pz']
                missing_nz_pz = [col for col in nz_pz_cols if col not in df.columns]
                if missing_nz_pz:
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"Warning: {csv_file.name} missing columns {missing_nz_pz}. Setting nz and pz to NaN.")
                    for col in missing_nz_pz:
                        df[col] = np.nan

                # Get wavelengths for the optical model
                wvl = df["wavelength"].value_counts().to_frame().reset_index()
                wvl["count"] = 1
                wvl_values = wvl.values

                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  Rays to process: {len(df)}")
                    print(f"  Unique wavelengths: {len(wvl_values)}")
                    print(f"  Chunk size: {chunk_size}")
                    print(f"  Processes: {n_processes or 'auto'}")

                # Convert DataFrame to ray format
                rays = [
                    (np.array([row.x, row.y, row.z], dtype=np.float64),
                    np.array([row.dx, row.dy, row.dz], dtype=np.float64),
                    np.array([row.wavelength], dtype=np.float64))
                    for row in df.itertuples()
                ]

                # Split rays into chunks
                chunks = self._chunk_rays(rays, chunk_size)
                index_chunks = [list(range(i, min(i + chunk_size, len(rays)))) for i in range(0, len(rays), chunk_size)]
                
                rays = None  # Clear memory

                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  Created {len(chunks)} chunks for processing")

                # Process chunks in parallel
                process_chunk = partial(
                    _process_ray_chunk_standalone,
                    opm_file_path=opm_file_path,
                    wvl_values=wvl_values,
                    verbosity=verbosity
                )
                
                try:
                    if n_processes == 1:
                        # Sequential processing for debugging
                        if verbosity >= VerbosityLevel.DETAILED:
                            print("  Using sequential processing")
                        results_with_indices = []
                        chunk_iter = tqdm(enumerate(zip(chunks, index_chunks)), 
                                        total=len(chunks),
                                        desc=f"Tracing rays ({csv_file.name})",
                                        disable=not progress_bar or verbosity == VerbosityLevel.QUIET)
                        
                        for chunk_idx, (chunk, indices) in chunk_iter:
                            chunk_result = process_chunk(chunk)
                            results_with_indices.extend(self._align_chunk_results(
                                chunk_result, indices, chunk_idx, verbosity))
                    else:
                        # Parallel processing
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"  Using parallel processing with {n_processes or 'auto'} processes")
                        
                        with Pool(processes=n_processes) as pool:
                            results_with_indices = []
                            chunk_iter = tqdm(
                                enumerate(zip(pool.imap(process_chunk, chunks), index_chunks)),
                                total=len(chunks),
                                desc=f"Tracing rays ({csv_file.name})",
                                disable=not progress_bar or verbosity == VerbosityLevel.QUIET
                            )
                            
                            for chunk_idx, (chunk_result, indices) in chunk_iter:
                                results_with_indices.extend(self._align_chunk_results(
                                    chunk_result, indices, chunk_idx, verbosity))
                            
                            pool.close()
                            pool.join()
                            
                except Exception as e:
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Error in processing {csv_file.name}: {str(e)}")
                        print("Consider using n_processes=1 for debugging")
                    raise

                # Create result DataFrame from processed chunks
                result_df = self._create_result_dataframe(results_with_indices, df, join, verbosity)

                # Verify alignment by checking row count
                if len(result_df) != len(df):
                    if verbosity > VerbosityLevel.QUIET:
                        print(f"  ERROR: Row count mismatch. Expected {len(df)}, got {len(result_df)}")
                    continue

                # Verify ID alignment
                if verbosity >= VerbosityLevel.DETAILED:
                    if 'id' in result_df.columns and 'id' in df.columns:
                        id_matches = (result_df['id'].values == df['id'].values).sum()
                        neutron_matches = (result_df['neutron_id'].values == df['neutron_id'].values).sum() if 'neutron_id' in result_df.columns else 0
                        pulse_matches = (result_df['pulse_id'].values == df['pulse_id'].values).sum() if 'pulse_id' in result_df.columns else 0
                        
                        if id_matches != len(df):
                            print(f"  WARNING: ID alignment check: {id_matches}/{len(df)} match")
                        if neutron_matches != len(df):
                            print(f"  WARNING: neutron_id alignment: {neutron_matches}/{len(df)} match")
                        if pulse_matches != len(df):
                            print(f"  WARNING: pulse_id alignment: {pulse_matches}/{len(df)} match")
                        
                        if id_matches == len(df) and neutron_matches == len(df) and pulse_matches == len(df):
                            print(f"  ✓ Perfect ID alignment verified: all {len(df)} rows match")

                # Process based on source workflow
                if source == "hits":
                    # Hits workflow: apply saturation and write TPX3 files
                    # Initialize in_tpx3 column as False (will be marked True for surviving rows)
                    result_df['in_tpx3'] = False
                    
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  Applying saturation with deadtime={deadtime} ns, blob={blob} pixels, decay_time={decay_time}ns")
                    
                    # Ensure required columns for saturation
                    required_cols = ['x2', 'y2', 'z2', 'toa2', 'id', 'neutron_id']
                    missing_cols = [col for col in required_cols if col not in result_df.columns]
                    if missing_cols:
                        if verbosity > VerbosityLevel.QUIET:
                            print(f"Cannot apply saturation to {csv_file.name}: missing columns {missing_cols}")
                        continue
                    
                    # Store original indices before saturation
                    result_df['_original_index'] = result_df.index
                    
                    # Call saturate_photons
                    saturated_df = self.saturate_photons(
                        data=result_df,
                        deadtime=deadtime,
                        blob=blob,
                        blob_variance=blob_variance,
                        seed=seed,
                        output_format="photons",
                        min_tot=1.0,
                        decay_time=decay_time,
                        detector_model=detector_model,
                        model_params=model_params,
                        verbosity=verbosity,
                        **kwargs  # Pass through additional model parameters
                    )
                    
                    if saturated_df is None or saturated_df.empty:
                        if verbosity > VerbosityLevel.QUIET:
                            print(f"  Saturation produced no results for {csv_file.name}")
                        result_df['in_tpx3'] = False
                    else:
                        # Mark rows that survived saturation
                        if '_original_index' in saturated_df.columns:
                            survived_indices = saturated_df['_original_index'].values
                            result_df.loc[survived_indices, 'in_tpx3'] = True
                        
                        # Update result_df with saturated data
                        result_df = saturated_df
                        
                        # Sort by time to restore chronological order
                        result_df = result_df.sort_values('toa2').reset_index(drop=True)
                        
                        # Remove temporary index column
                        if '_original_index' in result_df.columns:
                            result_df = result_df.drop(columns=['_original_index'])
                        
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"  After saturation and sorting: {len(result_df)} rows")
                            tpx3_count = result_df['in_tpx3'].sum() if 'in_tpx3' in result_df.columns else len(result_df)
                            print(f"  Rows marked for TPX3: {tpx3_count}")

                    # Filter columns to keep for hits workflow
                    desired_columns = ['pixel_x', 'pixel_y', 'toa2', 'photon_count', 'time_diff', 
                                    'id', 'neutron_id', 'pulse_id', 'pulse_time_ns', 'in_tpx3']
                    
                    columns_to_keep = [col for col in desired_columns if col in result_df.columns]
                    result_df = result_df[columns_to_keep]
                    
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  Filtered to columns: {columns_to_keep}")

                    # Save results to file
                    output_file = traced_photons_dir / f"traced_{csv_file.name}"
                    result_df.to_csv(output_file, index=False)

                    # Write TPX3 files
                    file_index = int(csv_file.stem.split('_')[-1])
                    
                    if 'in_tpx3' in result_df.columns:
                        tpx3_data = result_df[result_df['in_tpx3']].copy()
                    else:
                        tpx3_data = result_df.copy()
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"  Warning: 'in_tpx3' column not found, using all rows for TPX3")
                    
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  Writing {len(tpx3_data)} rows to TPX3 file")
                    
                    self._write_tpx3(
                        traced_data=tpx3_data,
                        chip_index=0,
                        verbosity=verbosity,
                        sensor_size=256,
                        split_method=split_method,
                        clean=(file_idx == 0),
                        file_index=file_index
                    )
                
                else:  # source == "photons"
                    # Photons workflow: save traced photons with tof for direct import
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  Preparing photons for direct import (no saturation)")
                    
                    # Calculate tof: time from pulse start to photon arrival, in seconds
                    if 'pulse_time_ns' in result_df.columns and 'toa2' in result_df.columns:
                        result_df['tof'] = (result_df['toa2'] - result_df['pulse_time_ns']) / 1e9  # ns to seconds
                    else:
                        result_df['tof'] = 0.0  # Fallback if pulse_time_ns not available
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"  Warning: pulse_time_ns not available, setting tof=0")
                    
                    # Filter columns for photons workflow - keep format compatible with empir_import_photons
                    desired_columns = ['pixel_x', 'pixel_y', 'toa2', 'tof', 
                                    'id', 'neutron_id', 'pulse_id', 'pulse_time_ns']
                    
                    columns_to_keep = [col for col in desired_columns if col in result_df.columns]
                    result_df = result_df[columns_to_keep]
                    
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  Filtered to columns: {columns_to_keep}")

                    # Save results to file
                    output_file = traced_photons_dir / f"traced_{csv_file.name}"
                    result_df.to_csv(output_file, index=False)
                    
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  Saved {len(result_df)} traced photons ready for import")
                    
                    # Convert to photonFiles format immediately
                    photon_files_dir = self.archive / "photonFiles"
                    photon_files_dir.mkdir(parents=True, exist_ok=True)
                    
                    self._run_import_photons(
                        traced_file=output_file,
                        photon_files_dir=photon_files_dir,
                        verbosity=verbosity
                    )

                # Print statistics if requested
                if print_stats and verbosity > VerbosityLevel.BASIC:
                    self._print_tracing_stats(csv_file.name, df, result_df)

                if return_df:
                    all_results.append(result_df)

            # Return combined results if requested
            if return_df and all_results:
                combined_df = pd.concat(all_results, ignore_index=True)
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"\nReturning combined DataFrame with {len(combined_df)} rows")
                return combined_df

            if verbosity >= VerbosityLevel.DETAILED:
                print(f"\nProcessing complete. Results saved to: {traced_photons_dir}")
            
            return None

        finally:
            pass


    def _trace_rays_grouped(self, opm=None, opm_file=None, zscan=0, zfine=0, fnumber=None,
                            source="photons", deadtime=None, blob=0.0, blob_variance=0.0, decay_time=100, 
                            seed: int = None, join=False, print_stats=False, n_processes=None, chunk_size=1000, 
                            progress_bar=True, timeout=3600, return_df=False, split_method="auto",
                            verbosity=VerbosityLevel.BASIC) -> pd.DataFrame or None:
        """
        Internal method for grouped ray tracing. 
        Trace rays for each group created by groupby() with all trace_rays options.
        
        All parameters are identical to trace_rays(). See trace_rays() for full documentation.
        
        Raises:
            ValueError: If groupby() hasn't been called first
        """
        if not hasattr(self, '_groupby_dir') or not hasattr(self, '_groupby_labels'):
            raise ValueError("Must call groupby() before trace_rays() for grouped operation")
        
        groupby_dir = self._groupby_dir
        labels = self._groupby_labels
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"\n{'='*60}")
            print(f"Tracing rays for {len(labels)} groups in: {groupby_dir.name}")
            print(f"Workflow: {source}")
            print(f"{'='*60}\n")
        
        # Store original archive
        original_archive = self.archive
        all_group_results = []
        
        # Iterate through each group
        for i, label in enumerate(tqdm(labels, desc="Processing groups", disable=(verbosity == VerbosityLevel.QUIET))):
            bin_dir = groupby_dir / label
            
            if not bin_dir.exists():
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Skipping non-existent group: {label}")
                continue
            
            # Check if SimPhotons folder exists and has data
            simphotons_dir = bin_dir / "SimPhotons"
            if not simphotons_dir.exists() or not list(simphotons_dir.glob("sim_data_*.csv")):
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Skipping group '{label}': no simulation data")
                continue
            
            if verbosity > VerbosityLevel.BASIC:
                print(f"\n{'─'*60}")
                print(f"Group {i+1}/{len(labels)}: {label}")
                print(f"{'─'*60}")
            
            # Temporarily change archive to this group's directory
            self.archive = bin_dir
            
            try:
                # Reload data for this group
                csv_files = sorted(simphotons_dir.glob("sim_data_*.csv"))
                valid_dfs = []
                for csv_file in csv_files:
                    try:
                        if csv_file.stat().st_size > 100:
                            df = pd.read_csv(csv_file)
                            if not df.empty:
                                valid_dfs.append(df)
                    except Exception as e:
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"Warning: Skipping {csv_file.name}: {e}")
                
                if valid_dfs:
                    self.data = pd.concat(valid_dfs, ignore_index=True)
                    
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  Loaded {len(self.data)} photons for group '{label}'")
                    
                    # Trace rays for this group with all parameters
                    group_result = self._trace_rays_single(
                        opm=opm,
                        opm_file=opm_file,
                        zscan=zscan,
                        zfine=zfine,
                        fnumber=fnumber,
                        source=source,
                        deadtime=deadtime,
                        blob=blob,
                        blob_variance=blob_variance,
                        decay_time=decay_time,          
                        seed=seed,              
                        join=join,
                        print_stats=print_stats,
                        n_processes=n_processes,
                        chunk_size=chunk_size,
                        progress_bar=progress_bar,
                        timeout=timeout,
                        return_df=return_df,
                        split_method=split_method,
                        verbosity=verbosity
                    )
                    
                    if return_df and group_result is not None:
                        all_group_results.append(group_result)
                    
                    if verbosity > VerbosityLevel.BASIC:
                        print(f"✓ Completed group '{label}'")
                else:
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  No valid data in group '{label}'")
            
            except Exception as e:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"✗ Error processing group '{label}': {e}")
                if verbosity >= VerbosityLevel.DETAILED:
                    import traceback
                    traceback.print_exc()
            
            finally:
                # Restore original archive
                self.archive = original_archive
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"\n{'='*60}")
            print(f"✓ Completed all {len(labels)} groups")
            print(f"{'='*60}\n")
        
        # Return combined results if requested
        if return_df and all_group_results:
            combined_df = pd.concat(all_group_results, ignore_index=True)
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"\nReturning combined DataFrame with {len(combined_df)} total rows from all groups")
            return combined_df
        
        return None


    def _run_import_photons(self, traced_file: Path, photon_files_dir: Path, 
                                        verbosity: VerbosityLevel) -> None:
        """
        Convert a single traced photon CSV file to empirphot binary format (.empirphot).
        
        This method reads a traced photon CSV file, formats it according to EMPIR requirements,
        and uses empir_import_photons to create the binary .empirphot file.
        
        The input format expects:
        - pixel_x, pixel_y: Pixel coordinates (already in pixels)
        - toa2: Time of arrival in nanoseconds (will be converted to seconds)
        - pulse_time_ns: Trigger/pulse time in nanoseconds (for calculating t_relToExtTrigger)
        
        Args:
            traced_file: Path to the traced photon CSV file (e.g., traced_sim_data_0.csv)
            photon_files_dir: Directory where .empirphot files will be saved
            verbosity: Controls output level
        
        Raises:
            ValueError: If required columns are missing from traced photon file
        """
        try:
            # Read traced photons
            df = pd.read_csv(traced_file)
            
            if df.empty:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"    Skipping empty traced file: {traced_file.name}")
                return
            
            # Validate required columns
            required_cols = ['pixel_x', 'pixel_y', 'toa2']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Traced photon file {traced_file.name} missing required columns: {missing_cols}\n"
                    f"Available columns: {list(df.columns)}\n"
                    f"Expected columns: {required_cols}"
                )
            
            # Drop rows with NaN in essential columns
            df = df[['pixel_x', 'pixel_y', 'toa2', 'pulse_time_ns']].dropna()
            
            # Convert times from nanoseconds to seconds
            df['t_s'] = df['toa2'] * 1e-9
            
            # Calculate time relative to external trigger (pulse_time_ns)
            if 'pulse_time_ns' in df.columns:
                df['t_relToExtTrigger_s'] = (df['toa2'] - df['pulse_time_ns']) * 1e-9
            else:
                # Fallback: use absolute time if pulse_time_ns not available
                df['t_relToExtTrigger_s'] = df['t_s']
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"    Warning: pulse_time_ns not found, using absolute time for t_relToExtTrigger")
            
            # Prepare DataFrame in EMPIR import format
            # Column names must match exactly what empir_import_photons expects
            import_df = pd.DataFrame({
                'x [px]': df['pixel_x'].astype(np.float64),
                'y [px]': df['pixel_y'].astype(np.float64),
                't [s]': df['t_s'].astype(np.float64),
                't_relToExtTrigger [s]': df['t_relToExtTrigger_s'].astype(np.float64)
            })
            
            
            if import_df.empty:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"    No photons in valid time range (0-1s) for {traced_file.name}")
                return
            
            # Sort by time
            import_df = import_df.sort_values('t [s]')
            
            # Create ImportedPhotons directory for intermediate CSV
            imported_photons_dir = self.archive / "ImportedPhotons"
            imported_photons_dir.mkdir(exist_ok=True)
            
            # Extract the data index from filename
            stem = traced_file.stem
            if stem.startswith('traced_sim_data_'):
                data_index = stem.replace('traced_sim_data_', '')
            elif stem.startswith('traced_data_'):
                data_index = stem.replace('traced_data_', '')
            else:
                # Fallback: try to extract number from end
                data_index = ''.join(filter(str.isdigit, stem))
            
            # Save intermediate CSV for empir_import_photons
            output_csv = imported_photons_dir / f"imported_traced_data_{data_index}.csv"
            import_df.to_csv(output_csv, index=False)
            
            # Output empirphot binary file
            output_file = photon_files_dir / f"traced_data_{data_index}.empirphot"
            
            # Use empir_import_photons to create the binary .empirphot file
            import subprocess
            cmd = [
                str(self.empir_import_photons),
                str(output_csv),
                str(output_file),
                "csv"
            ]
            
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"    Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            else:
                result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"empir_import_photons failed for {traced_file.name}")
            
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"    ✓ Converted {traced_file.name} → {output_file.name} ({len(import_df)} photons)")
        
        except Exception as e:
            if verbosity > VerbosityLevel.BASIC:
                print(f"    ✗ Error converting {traced_file.name}: {e}")
            raise

    def _write_tpx3(
        self,
        traced_data: pd.DataFrame = None,
        chip_index: int = 0,
        verbosity: int = 1,
        sensor_size: int = 256,
        split_method: str = "auto",
        clean: bool = True,
        file_index: int = None
    ):
        """
        Convert traced photon data to valid TPX3 binary files, following the SERVAL TPX3 raw file format.
        """
        # Constants
        TICK_NS = 1.5625  # ToA tick size (1.5625 ns)
        MAX_TDC_TIMESTAMP_S = (2**32) * 25e-9  # ~107.37 seconds
        MAX_CHUNK_BYTES = 65535
        TIMER_TICK_NS = 409.6  # GTS timer tick
        PACKET_SIZE = 8
        
        def encode_gts_pair(timer_value):
            """Encode Global Timestamp (GTS) packet pair."""
            timer_value = int(timer_value) & ((1 << 48) - 1)
            lsb_timer = timer_value & 0xFFFFFFFF
            lsb_word = (0x4 << 60) | (0x4 << 56) | (lsb_timer << 16)
            msb_word = (0x4 << 60) | (0x5 << 56) | ((timer_value >> 32) & 0xFFFF) << 16
            return struct.pack("<Q", lsb_word) + struct.pack("<Q", msb_word)
        
        def encode_tdc_packet(trigger_time_ns, trigger_counter, tdc_channel=1, edge_type='rising'):
            """
            Encode TDC trigger packet following TPX3 format.
            
            From C++ decoder:
            coarsetime = (temp >> 12) & 0xFFFFFFFF  (32 bits in 25ns units)
            tmpfine = (temp >> 5) & 0xF  (4 bits, clock phase 1-12)
            trigtime_fine = (temp & 0x0E00) | (((tmpfine-1) << 9) / 12)
            tdc_time = coarsetime*25e-9 + trigtime_fine*(25/4096)*1e-9
            """
            # Determine header
            if tdc_channel == 1:
                header = 0x6F if edge_type == 'rising' else 0x6A
            else:
                header = 0x6E if edge_type == 'rising' else 0x6B
            
            # Coarse time: 25ns resolution (32 bits)
            coarsetime = int(trigger_time_ns / 25.0) & 0xFFFFFFFF
            
            # Fine time: sub-25ns component
            fine_ns = trigger_time_ns - (coarsetime * 25.0)
            
            # Convert to 12-bit trigtime_fine (0-4095 representing 0-25ns)
            trigtime_fine = int(round(fine_ns * 4096.0 / 25.0)) & 0xFFF
            
            # Extract upper 3 bits (bits 9-11)
            fine_upper = (trigtime_fine >> 9) & 0x7
            
            # Extract lower 9 bits
            fine_lower = trigtime_fine & 0x1FF
            
            # Reverse tmpfine encoding: fine_lower ≈ ((tmpfine-1) << 9) / 12
            # So: tmpfine ≈ (fine_lower * 12 / 512) + 1
            if fine_lower == 0:
                tmpfine = 1
            else:
                tmpfine = int(round((fine_lower * 12.0 / 512.0) + 1.0))
                tmpfine = max(1, min(12, tmpfine))
            
            # Build 64-bit TDC packet
            tdc_word = (
                (int(header) << 56) |
                ((trigger_counter & 0xFFF) << 44) |
                ((coarsetime & 0xFFFFFFFF) << 12) |
                ((fine_upper & 0x7) << 9) |
                ((tmpfine & 0xF) << 5)
            )
            
            return struct.pack("<Q", tdc_word)
        
        if traced_data is None or len(traced_data) == 0:
            if verbosity >= 2:
                print("No traced photon data provided")
            return
        
        df = traced_data.copy()
        
        # Sort by neutron_id and toa2
        if "neutron_id" in df.columns:
            df = df.sort_values(by=["neutron_id", "toa2"]).reset_index(drop=True)
        
        if verbosity >= 2:
            print(f"\nProcessing {len(df)} events for TPX3 export")
        
        # Validate required columns
        required = ["pixel_x", "pixel_y", "toa2", "time_diff", "pulse_id", "pulse_time_ns"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            if verbosity >= 2:
                print(f"Missing required columns: {missing}")
            return
        
        # Filter out invalid rows (NaN values and out-of-range coordinates)
        valid_mask = (
            df["pixel_x"].notna() & 
            df["pixel_y"].notna() & 
            df["toa2"].notna() & 
            df["time_diff"].notna() &
            (df["pixel_x"] >= 0) & (df["pixel_x"] < sensor_size) &
            (df["pixel_y"] >= 0) & (df["pixel_y"] < sensor_size)
        )
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            n_nan = df[["pixel_x", "pixel_y", "toa2", "time_diff"]].isna().any(axis=1).sum()
            n_oob = ((df["pixel_x"] < 0) | (df["pixel_x"] >= sensor_size) | 
                     (df["pixel_y"] < 0) | (df["pixel_y"] >= sensor_size)).sum()
            if verbosity >= 2:
                print(f"  Filtering out {n_invalid} invalid rows:")
                print(f"    - {n_nan} with NaN values")
                print(f"    - {n_oob} with out-of-range coordinates (must be 0-{sensor_size-1})")
            df = df[valid_mask].reset_index(drop=True)
        
        if len(df) == 0:
            if verbosity >= 1:
                print("  No valid data after filtering")
            return
        
        # Setup output directory
        out_dir = self.archive / "tpx3Files"
        if out_dir.exists() and clean and (file_index is None or file_index == 0):
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = f"traced_data_{file_index}" if file_index is not None else "traced_data"
        
        # Build trigger time dictionary
        trigger_time_dict = {}
        for pulse_id in df['pulse_id'].dropna().unique():
            pulse_rows = df[df['pulse_id'] == pulse_id]
            if not pulse_rows.empty:
                trigger_time_ns = pulse_rows['pulse_time_ns'].iloc[0]
                if trigger_time_ns > MAX_TDC_TIMESTAMP_S * 1e9:
                    if verbosity >= 2:
                        print(f"  Warning: pulse_id {pulse_id} time exceeds TDC range")
                    trigger_time_ns = MAX_TDC_TIMESTAMP_S * 1e9
                trigger_time_dict[int(pulse_id)] = trigger_time_ns
        
        if verbosity >= 2:
            print(f"  Triggers: {len(trigger_time_dict)}")
            if trigger_time_dict:
                sample = list(trigger_time_dict.items())[:3]
                print(f"  Sample: {[(k, f'{v:.1f}ns') for k, v in sample]}")
        
        # Extract and convert data (no clipping - already filtered)
        px = df["pixel_x"].to_numpy().astype(np.int64)
        py = df["pixel_y"].to_numpy().astype(np.int64)
        toa_ns = df["toa2"].to_numpy().astype(float)
        tot_ns = np.maximum(df["time_diff"].to_numpy().astype(float), 1.0)
        
        # Convert ToA to 1.5625ns ticks
        toa_ticks = np.round(toa_ns / TICK_NS).astype(np.int64)
        
        # Decompose ToA into packet fields
        spidr_time = ((toa_ticks >> 18) & 0xFFFF).astype(np.int64)  # 16 bits
        coarse_toa = ((toa_ticks >> 4) & 0x3FFF).astype(np.int64)
        ftoa = (15 - (toa_ticks & 0xF)).astype(np.int64)
        ftoa = np.clip(ftoa, 0, 15)
        
        # Convert ToT to ticks (10-bit, ~1ns resolution)
        tot_ticks = np.clip(np.round(tot_ns).astype(np.int64), 1, 0x3FF)
        
        # Encode PixAddr using TPX3 hierarchical addressing scheme
        # C++ decoder extracts from 64-bit word at these positions:
        #   dcol = (word >> 52) & 0x7F    (bits 58-52, 7 bits)
        #   spix = (word >> 45) & 0x3F    (bits 50-45, 6 bits)
        #   pix  = (word >> 44) & 0x7     (bits 46-44, 3 bits)
        # Then reconstructs: X = dcol + pix/4,  Y = spix + (pix & 3)
        #
        # So we need to encode directly into those bit positions:
        pix = (((px & 0x1) << 2) | (py & 0x3)).astype(np.uint64)  # 3 bits: X[0] and Y[1:0]
        dcol = (px >> 1).astype(np.uint64)  # 7 bits: X / 2
        spix = (py >> 2).astype(np.uint64)  # 6 bits: Y / 4
        
        # Timer for GTS
        timer_ticks = (toa_ticks * TICK_NS / TIMER_TICK_NS).astype(np.int64)
        
        if verbosity >= 2:
            print(f"  ToA: {toa_ns.min():.1f} - {toa_ns.max():.1f} ns")
            print(f"  ToT: {tot_ns.min():.1f} - {tot_ns.max():.1f} ns")
            print(f"  Pixels: x[{px.min()}, {px.max()}], y[{py.min()}, {py.max()}]")
            
            # Verify encoding
            print(f"  Encoding verification:")
            for i in [0, min(5, len(px)-1)]:
                # Simulate what decoder will extract
                decoded_dcol = dcol[i]
                decoded_spix = spix[i]
                decoded_pix = pix[i]
                decoded_x = decoded_dcol + (decoded_pix >> 2)
                decoded_y = decoded_spix + (decoded_pix & 0x3)
                match = "✓" if (decoded_x == px[i] and decoded_y == py[i]) else "✗"
                print(f"    ({px[i]},{py[i]}) → dcol={dcol[i]}, spix={spix[i]}, pix={pix[i]} → ({decoded_x},{decoded_y}) {match}")
        
        # Encode pixel packets
        pixel_packets = []
        for j in range(len(df)):
            # Build 64-bit pixel packet
            # Based on reverse engineering from actual decoder output:
            # The PixAddr field needs special encoding
            # Treating it as a 16-bit field at bits 59-44
            
            # Standard TPX3: PixAddr encodes doublecolumn, superpixel, pixel
            # dcol (7 bits) | spix (6 bits) | pix (3 bits) = 16 bits
            pix_addr_16bit = ((int(dcol[j]) & 0x7F) << 9) | ((int(spix[j]) & 0x3F) << 3) | (int(pix[j]) & 0x7)
            
            pixel_word = (
                (0xB << 60) |
                ((pix_addr_16bit & 0xFFFF) << 44) |
                ((int(coarse_toa[j]) & 0x3FFF) << 30) |
                ((int(tot_ticks[j]) & 0x3FF) << 20) |
                ((int(ftoa[j]) & 0xF) << 16) |
                (int(spidr_time[j]) & 0xFFFF)
            )
            pixel_packets.append(struct.pack("<Q", pixel_word))
        
        # Determine file groups
        file_groups = []
        
        if split_method == "event" and "neutron_id" in df.columns:
            neutron_ids = df["neutron_id"].to_numpy()
            pulse_ids = df["pulse_id"].to_numpy()
            
            for nid in np.unique(neutron_ids):
                indices = np.where(neutron_ids == nid)[0]
                if len(indices) > 0:
                    pulse_id = pulse_ids[indices[0]]
                    unique_pulses = np.unique(pulse_ids[indices])
                    if len(unique_pulses) > 1 and verbosity >= 2:
                        print(f"  Warning: neutron {nid} spans multiple pulses: {unique_pulses}")
                    
                    trigger_time_ns = trigger_time_dict.get(int(pulse_id))
                    
                    file_groups.append({
                        'start_idx': int(indices[0]),
                        'end_idx': int(indices[-1] + 1),
                        'neutron_id': int(nid),
                        'pulse_id': int(pulse_id),
                        'trigger_time_ns': trigger_time_ns
                    })
            
            if verbosity >= 2:
                print(f"  Files: {len(file_groups)} (one per neutron)")
                
        elif "neutron_id" in df.columns:
            neutron_ids = df["neutron_id"].to_numpy()
            pulse_ids = df["pulse_id"].to_numpy()
            
            boundaries = [0]
            for i in range(1, len(neutron_ids)):
                if neutron_ids[i] != neutron_ids[i-1]:
                    boundaries.append(i)
            boundaries.append(len(df))
            
            current_start = 0
            current_size = 16
            current_triggers = set()
            
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                
                n_size = (end - start) * PACKET_SIZE
                n_pulses = set(pulse_ids[start:end])
                new_triggers = n_pulses - current_triggers
                n_size += len(new_triggers) * 8
                
                if current_size + n_size > MAX_CHUNK_BYTES and start > current_start:
                    group_pulses = set(pulse_ids[current_start:start])
                    group_triggers = {
                        int(pid): trigger_time_dict.get(int(pid))
                        for pid in group_pulses
                        if int(pid) in trigger_time_dict
                    }
                    
                    file_groups.append({
                        'start_idx': current_start,
                        'end_idx': start,
                        'neutron_id': None,
                        'pulse_ids': group_pulses,
                        'trigger_times': group_triggers
                    })
                    
                    current_start = start
                    current_size = 16 + n_size
                    current_triggers = n_pulses.copy()
                else:
                    current_size += n_size
                    current_triggers.update(n_pulses)
            
            if current_start < len(df):
                group_pulses = set(pulse_ids[current_start:])
                group_triggers = {
                    int(pid): trigger_time_dict.get(int(pid))
                    for pid in group_pulses
                    if int(pid) in trigger_time_dict
                }
                
                file_groups.append({
                    'start_idx': current_start,
                    'end_idx': len(df),
                    'neutron_id': None,
                    'pulse_ids': group_pulses,
                    'trigger_times': group_triggers
                })
            
            if verbosity >= 2:
                print(f"  Files: {len(file_groups)} (auto-grouped)")
        else:
            file_groups.append({
                'start_idx': 0,
                'end_idx': len(df),
                'neutron_id': None,
                'pulse_id': None,
                'trigger_time_ns': None
            })
        
        # Write files
        files_written = []
        for file_idx, group in enumerate(file_groups):
            start_idx = group['start_idx']
            end_idx = group['end_idx']
            
            # Initialize with GTS pair
            if split_method == "event":
                trigger_ns = group.get('trigger_time_ns')
                timer = int(trigger_ns / TIMER_TICK_NS) if trigger_ns else timer_ticks[start_idx]
            else:
                triggers = group.get('trigger_times', {})
                timer = int(min(triggers.values()) / TIMER_TICK_NS) if triggers else timer_ticks[start_idx]
            
            content = encode_gts_pair(timer)
            
            # Add TDC triggers
            n_triggers = 0
            
            if split_method == "event":
                trigger_ns = group.get('trigger_time_ns')
                pulse_id = group.get('pulse_id')
                
                if trigger_ns is not None and pulse_id is not None:
                    tdc_packet = encode_tdc_packet(trigger_ns, pulse_id, 1, 'rising')
                    content += tdc_packet
                    n_triggers = 1
                    
                    if verbosity >= 2:
                        first_toa = toa_ns[start_idx]
                        dt = first_toa - trigger_ns
                        print(f"  File {file_idx + 1}: TDC @ {trigger_ns:.1f}ns, first hit @ {first_toa:.1f}ns (Δt={dt:.1f}ns)")
            else:
                triggers = group.get('trigger_times', {})
                
                for pulse_id, trigger_ns in sorted(triggers.items()):
                    if trigger_ns is not None:
                        tdc_packet = encode_tdc_packet(trigger_ns, pulse_id, 1, 'rising')
                        content += tdc_packet
                        n_triggers += 1
                
                if verbosity >= 2 and n_triggers > 0:
                    print(f"  File {file_idx + 1}: {n_triggers} TDC trigger(s)")
            
            # Add pixel packets
            for j in range(start_idx, end_idx):
                content += pixel_packets[j]
            
            # Determine filename
            neutron_id = group.get('neutron_id')
            if split_method == "event" and neutron_id is not None:
                out_path = out_dir / f"{base_name}_neutron{neutron_id:06d}.tpx3"
            elif len(file_groups) > 1:
                out_path = out_dir / f"{base_name}_part{file_idx + 1:03d}.tpx3"
            else:
                out_path = out_dir / f"{base_name}.tpx3"
            
            # Write file
            with open(out_path, "wb") as fh:
                file_size = len(content)
                header = struct.pack("<4sBBH", b"TPX3", chip_index & 0xFF, 0, file_size & 0xFFFF)
                fh.write(header)
                fh.write(content)
            
            files_written.append((out_path, end_idx - start_idx, n_triggers))
            
            if verbosity >= 2:
                info = f"neutron {neutron_id}" if neutron_id else f"part {file_idx + 1}"
                trig_info = f", {n_triggers} trig" if n_triggers else ""
                print(f"  Wrote {out_path.name}: {end_idx - start_idx} events{trig_info}")
        
        if verbosity >= 2:
            total_trigs = sum(t for _, _, t in files_written)
            if len(files_written) == 2:
                print(f"✅ {files_written[0][0].name}: {files_written[0][1]} events, {files_written[0][2]} triggers")
            else:
                print(f"✅ {len(files_written)} files, {total_trigs} triggers total")

    def _align_chunk_results(self, chunk_result, indices, chunk_idx, verbosity):
        """
        Align chunk processing results with original row indices.
        
        Parameters:
        -----------
        chunk_result : list or None
            Results from processing a chunk of rays
        indices : list
            Original row indices for this chunk
        chunk_idx : int
            Index of the current chunk (for error reporting)
        verbosity : VerbosityLevel
            Current verbosity level
            
        Returns:
        --------
        list
            List of (result, index) tuples properly aligned
        """
        if chunk_result is None:
            chunk_result = [None] * len(indices)
        elif len(chunk_result) != len(indices):
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"    Warning: Chunk {chunk_idx} returned {len(chunk_result)} "
                    f"results but expected {len(indices)}")
            if len(chunk_result) < len(indices):
                chunk_result = chunk_result + [None] * (len(indices) - len(chunk_result))
            else:
                chunk_result = chunk_result[:len(indices)]
        
        return list(zip(chunk_result, indices))

    def _create_result_dataframe(self, results_with_indices, original_df, join, verbosity):
        """
        Create a DataFrame from traced ray results with 1:1 row correspondence to original data.
        
        CRITICAL: Maintains exact row order and count. Row i in output = Row i in input.
        If a photon fails to trace, that row will have NaN for x2, y2, z2 but keep original IDs.
        
        Parameters:
        -----------
        results_with_indices : list
            List of (trace_result, original_row_index) tuples
        original_df : pd.DataFrame
            Original simulation data
        join : bool
            If True, include original ray definition columns (x, y, z, dx, dy, dz, wavelength)
        verbosity : VerbosityLevel
            Logging verbosity
            
        Returns:
        --------
        pd.DataFrame
            Result DataFrame with same length and order as original_df
        """
        # Sort by original index to maintain input order
        results_with_indices.sort(key=lambda x: x[1])
        
        # Verify we have exactly one result per input row
        expected_indices = set(range(len(original_df)))
        actual_indices = {idx for _, idx in results_with_indices}
        
        if len(results_with_indices) != len(original_df):
            if verbosity > VerbosityLevel.BASIC:
                print(f"    WARNING: Result count mismatch. Expected {len(original_df)}, got {len(results_with_indices)}")
            
            # Find and fill missing indices
            missing_indices = expected_indices - actual_indices
            if missing_indices:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"    Filling {len(missing_indices)} missing results with NaN")
                for idx in missing_indices:
                    results_with_indices.append((None, idx))
            
            # Remove extra indices
            extra_indices = actual_indices - expected_indices
            if extra_indices:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"    Removing {len(extra_indices)} extra results")
                results_with_indices = [(res, idx) for res, idx in results_with_indices if idx in expected_indices]
            
            # Re-sort after additions/removals
            results_with_indices.sort(key=lambda x: x[1])
        
        # Sanity check: verify we now have the right indices
        final_indices = [idx for _, idx in results_with_indices]
        if final_indices != list(range(len(original_df))):
            if verbosity > VerbosityLevel.QUIET:
                print(f"    ERROR: Index mismatch after alignment! Expected 0..{len(original_df)-1}")
            # Force correct order by rebuilding
            index_to_result = {idx: res for res, idx in results_with_indices}
            results_with_indices = [(index_to_result.get(i), i) for i in range(len(original_df))]
        
        # Build output DataFrame row by row
        result_rows = []
        
        for trace_result, orig_idx in results_with_indices:
            # Extract traced position (or NaN if failed)
            if trace_result is not None:
                try:
                    ray, path_length, wvl = trace_result
                    position = ray[0]
                    x2, y2, z2 = float(position[0]), float(position[1]), float(position[2])
                except Exception as e:
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"    Error extracting trace result for row {orig_idx}: {str(e)}")
                    x2, y2, z2 = np.nan, np.nan, np.nan
            else:
                x2, y2, z2 = np.nan, np.nan, np.nan
            
            # Start with traced coordinates
            row = {'x2': x2, 'y2': y2, 'z2': z2}
            
            # Copy ID and metadata columns from original data for THIS specific row
            orig_row = original_df.iloc[orig_idx]
            
            for col in ['id', 'neutron_id', 'pulse_id', 'parent_id', 'nz', 'pz', 'pulse_time_ns']:
                if col in original_df.columns:
                    row[col] = orig_row[col]
            
            # Copy timing
            if 'toa' in original_df.columns:
                row['toa2'] = orig_row['toa']
            
            # If join=True, also copy ray definition columns
            if join:
                for col in ['x', 'y', 'z', 'dx', 'dy', 'dz', 'wavelength']:
                    if col in original_df.columns:
                        row[col] = orig_row[col]
            
            result_rows.append(row)
        
        # Create DataFrame - row i corresponds to original_df.iloc[i]
        result_df = pd.DataFrame(result_rows)
        
        # Verify final alignment
        if verbosity >= VerbosityLevel.DETAILED:
            if 'id' in result_df.columns and 'id' in original_df.columns:
                matches = (result_df['id'].values == original_df['id'].values).sum()
                if matches != len(result_df):
                    print(f"    ERROR: Only {matches}/{len(result_df)} IDs match after _create_result_dataframe!")

        # Convert position to pixels
        result_df["pixel_x"] = np.ceil((result_df["x2"]*self.reduction_ratio + 0.5*self.FOV)*256/self.FOV)
        result_df["pixel_y"] = np.ceil((result_df["y2"]*self.reduction_ratio + 0.5*self.FOV)*256/self.FOV)
        
        return result_df

    # ==================== Detector Model Helper Methods ====================

    def _apply_image_intensifier_model(self, cx, cy, photon_toa, blob, blob_variance, decay_time):
        """
        Image intensifier model: circular blob with exponential delay.

        Returns:
        --------
        tuple: (covered_x, covered_y, activation_time, pixel_weights)
            - covered_x, covered_y: arrays of pixel coordinates
            - activation_time: time when pixels are activated
            - pixel_weights: array of weights (all 1.0 for this model)
        """
        # Draw single exponential delay for this photon's entire blob
        activation_delay = np.random.exponential(decay_time)
        activation_time = photon_toa + activation_delay

        # Draw blob radius for this photon
        if blob_variance > 0 and blob > 0:
            min_radius = blob - blob_variance
            actual_blob = np.random.uniform(min_radius, blob)
        else:
            actual_blob = blob

        # Find all pixels covered by the circular blob
        if actual_blob > 0:
            i_min = int(np.floor(cx - actual_blob - 0.5))
            i_max = int(np.ceil(cx + actual_blob + 0.5))
            j_min = int(np.floor(cy - actual_blob - 0.5))
            j_max = int(np.ceil(cy + actual_blob + 0.5))

            # Create grid of pixel centers
            x_grid = np.arange(i_min, i_max + 1)
            y_grid = np.arange(j_min, j_max + 1)
            xx, yy = np.meshgrid(x_grid, y_grid)
            xx = xx.flatten()
            yy = yy.flatten()

            # Find closest point in each pixel to circle center
            closest_x = np.clip(cx, xx - 0.5, xx + 0.5)
            closest_y = np.clip(cy, yy - 0.5, yy + 0.5)
            dist2 = (closest_x - cx) ** 2 + (closest_y - cy) ** 2
            mask = dist2 <= actual_blob ** 2

            covered_x = xx[mask]
            covered_y = yy[mask]
        else:
            # No blob: only the pixel containing the photon center
            covered_x = np.array([int(np.floor(cx))])
            covered_y = np.array([int(np.floor(cy))])

        # All pixels have equal weight
        pixel_weights = np.ones(len(covered_x))

        return covered_x, covered_y, activation_time, pixel_weights

    def _apply_gaussian_diffusion_model(self, cx, cy, photon_toa, sigma, model_params):
        """
        Gaussian charge diffusion model: charge spreads with Gaussian PSF.

        Returns:
        --------
        tuple: (covered_x, covered_y, activation_time, pixel_weights)
            - covered_x, covered_y: arrays of pixel coordinates
            - activation_time: time when charge arrives (no delay)
            - pixel_weights: Gaussian-weighted charge distribution
        """
        charge_coupling = model_params.get('charge_coupling', 1.0)  # 0-1, fraction of charge collected
        activation_time = photon_toa  # Direct detection, no conversion delay

        if sigma > 0:
            # Sample pixels within 3 sigma
            search_radius = 3.0 * sigma
            i_min = int(np.floor(cx - search_radius - 0.5))
            i_max = int(np.ceil(cx + search_radius + 0.5))
            j_min = int(np.floor(cy - search_radius - 0.5))
            j_max = int(np.ceil(cy + search_radius + 0.5))

            # Create grid of pixel centers
            x_grid = np.arange(i_min, i_max + 1)
            y_grid = np.arange(j_min, j_max + 1)
            xx, yy = np.meshgrid(x_grid, y_grid)
            xx = xx.flatten()
            yy = yy.flatten()

            # Calculate Gaussian weights based on distance to pixel centers
            dx = xx - cx
            dy = yy - cy
            dist2 = dx**2 + dy**2
            weights = np.exp(-dist2 / (2 * sigma**2))

            # Apply charge coupling efficiency
            weights *= charge_coupling

            # Keep only pixels with significant charge (> 1% of peak)
            threshold = 0.01 * charge_coupling
            mask = weights > threshold

            covered_x = xx[mask]
            covered_y = yy[mask]
            pixel_weights = weights[mask]
        else:
            # No diffusion: single pixel
            covered_x = np.array([int(np.floor(cx))])
            covered_y = np.array([int(np.floor(cy))])
            pixel_weights = np.array([charge_coupling])

        return covered_x, covered_y, activation_time, pixel_weights

    def _apply_direct_detection_model(self, cx, cy, photon_toa):
        """
        Direct detection model: single pixel, no blob.

        Returns:
        --------
        tuple: (covered_x, covered_y, activation_time, pixel_weights)
            - covered_x, covered_y: single pixel containing photon
            - activation_time: direct photon arrival time
            - pixel_weights: weight of 1.0
        """
        covered_x = np.array([int(np.floor(cx))])
        covered_y = np.array([int(np.floor(cy))])
        activation_time = photon_toa
        pixel_weights = np.ones(1)

        return covered_x, covered_y, activation_time, pixel_weights

    def _apply_wavelength_dependent_model(self, cx, cy, photon_toa, wavelength, blob, decay_time, model_params):
        """
        Wavelength-dependent model: spectral QE and wavelength-dependent blob size.

        Returns:
        --------
        tuple: (covered_x, covered_y, activation_time, pixel_weights) or None
            - Returns None if photon is not detected (QE check fails)
            - Otherwise returns pixel coverage with wavelength-scaled blob
        """
        # Get QE curve from model_params
        qe_wavelength = np.array(model_params.get('qe_wavelength', [400, 500, 600]))  # nm
        qe_values = np.array(model_params.get('qe_values', [0.3, 0.3, 0.3]))  # quantum efficiency

        # Interpolate QE at photon wavelength
        qe_at_wavelength = np.interp(wavelength, qe_wavelength, qe_values, left=0.0, right=0.0)

        # Check if photon is detected based on QE
        if np.random.random() > qe_at_wavelength:
            return None  # Photon not detected

        # Scale blob size with wavelength (longer wavelength -> larger diffraction)
        wavelength_factor = model_params.get('wavelength_scaling', wavelength / 500.0)  # normalized to 500nm
        scaled_blob = blob * wavelength_factor

        # Apply exponential delay
        activation_delay = np.random.exponential(decay_time)
        activation_time = photon_toa + activation_delay

        # Use circular blob (similar to image intensifier)
        if scaled_blob > 0:
            i_min = int(np.floor(cx - scaled_blob - 0.5))
            i_max = int(np.ceil(cx + scaled_blob + 0.5))
            j_min = int(np.floor(cy - scaled_blob - 0.5))
            j_max = int(np.ceil(cy + scaled_blob + 0.5))

            x_grid = np.arange(i_min, i_max + 1)
            y_grid = np.arange(j_min, j_max + 1)
            xx, yy = np.meshgrid(x_grid, y_grid)
            xx = xx.flatten()
            yy = yy.flatten()

            closest_x = np.clip(cx, xx - 0.5, xx + 0.5)
            closest_y = np.clip(cy, yy - 0.5, yy + 0.5)
            dist2 = (closest_x - cx) ** 2 + (closest_y - cy) ** 2
            mask = dist2 <= scaled_blob ** 2

            covered_x = xx[mask]
            covered_y = yy[mask]
        else:
            covered_x = np.array([int(np.floor(cx))])
            covered_y = np.array([int(np.floor(cy))])

        pixel_weights = np.ones(len(covered_x))

        return covered_x, covered_y, activation_time, pixel_weights

    def _apply_avalanche_gain_model(self, cx, cy, photon_toa, blob, model_params):
        """
        Avalanche gain model: Poisson-distributed gain with afterpulsing.

        Returns:
        --------
        tuple: (covered_x, covered_y, activation_time, pixel_weights, afterpulse_events)
            - covered_x, covered_y: pixels affected by primary pulse
            - activation_time: primary detection time
            - pixel_weights: gain-weighted response
            - afterpulse_events: list of (time, x, y) for afterpulses
        """
        mean_gain = model_params.get('mean_gain', 100)
        gain_variance = model_params.get('gain_variance', 20)
        afterpulse_prob = model_params.get('afterpulse_prob', 0.01)
        afterpulse_delay_mean = model_params.get('afterpulse_delay', 200.0)  # ns

        # Draw gain from Poisson-like distribution (using Gamma approximation)
        if gain_variance > 0:
            # Gamma distribution: shape = mean^2/variance, scale = variance/mean
            shape = mean_gain**2 / gain_variance**2
            scale = gain_variance**2 / mean_gain
            gain = np.random.gamma(shape, scale)
        else:
            gain = mean_gain

        activation_time = photon_toa

        # Determine affected pixels (small blob due to avalanche region)
        if blob > 0:
            i_min = int(np.floor(cx - blob - 0.5))
            i_max = int(np.ceil(cx + blob + 0.5))
            j_min = int(np.floor(cy - blob - 0.5))
            j_max = int(np.ceil(cy + blob + 0.5))

            x_grid = np.arange(i_min, i_max + 1)
            y_grid = np.arange(j_min, j_max + 1)
            xx, yy = np.meshgrid(x_grid, y_grid)
            xx = xx.flatten()
            yy = yy.flatten()

            closest_x = np.clip(cx, xx - 0.5, xx + 0.5)
            closest_y = np.clip(cy, yy - 0.5, yy + 0.5)
            dist2 = (closest_x - cx) ** 2 + (closest_y - cy) ** 2
            mask = dist2 <= blob ** 2

            covered_x = xx[mask]
            covered_y = yy[mask]
        else:
            covered_x = np.array([int(np.floor(cx))])
            covered_y = np.array([int(np.floor(cy))])

        # Weight by gain (normalized)
        pixel_weights = np.full(len(covered_x), gain / 100.0)

        # Generate afterpulses
        afterpulse_events = []
        if np.random.random() < afterpulse_prob:
            delay = np.random.exponential(afterpulse_delay_mean)
            afterpulse_events.append((photon_toa + delay, cx, cy))

        return covered_x, covered_y, activation_time, pixel_weights, afterpulse_events

    def _apply_image_intensifier_gain_model(self, cx, cy, photon_toa, blob, decay_time, model_params):
        """
        Gain-dependent image intensifier model with physics-based blob scaling.

        Physics:
        - Blob size: σ = σ₀ × (gain/gain_ref)^exponent
        - Gaussian photon distribution
        - Charge-weighted TOT
        - Based on MCP physics (Photonis specs, Siegmund et al.)

        Returns:
        --------
        tuple: (covered_x, covered_y, activation_time, pixel_weights)
        """
        # Model parameters (with defaults from literature)
        gain = model_params.get('gain', 5000)  # MCP gain (typical: 10³-10⁴)
        sigma_0 = model_params.get('sigma_0', 1.0)  # Base blob sigma at gain_ref (pixels)
        gain_ref = model_params.get('gain_ref', 1000)  # Reference gain
        gain_exponent = model_params.get('gain_exponent', 0.4)  # Scaling exponent (literature: 0.3-0.5)

        # Calculate gain-dependent blob size: σ ∝ (gain)^exponent
        sigma_pixels = sigma_0 * (gain / gain_ref) ** gain_exponent

        # Override blob if explicitly provided
        if blob > 0:
            sigma_pixels = blob

        # Draw exponential delay for phosphor emission
        activation_delay = np.random.exponential(decay_time)
        activation_time = photon_toa + activation_delay

        # Sample pixels within 3σ (covers 99.7% of photons)
        if sigma_pixels > 0:
            search_radius = 3.0 * sigma_pixels
            i_min = int(np.floor(cx - search_radius - 0.5))
            i_max = int(np.ceil(cx + search_radius + 0.5))
            j_min = int(np.floor(cy - search_radius - 0.5))
            j_max = int(np.ceil(cy + search_radius + 0.5))

            # Create pixel grid
            x_grid = np.arange(i_min, i_max + 1)
            y_grid = np.arange(j_min, j_max + 1)
            xx, yy = np.meshgrid(x_grid, y_grid)
            xx = xx.flatten()
            yy = yy.flatten()

            # Gaussian weights: I(r) = exp(-r²/2σ²)
            dx = xx - cx
            dy = yy - cy
            dist2 = dx**2 + dy**2
            weights = np.exp(-dist2 / (2 * sigma_pixels**2))

            # Keep pixels with significant charge (> 1% of peak)
            threshold = 0.01
            mask = weights > threshold

            covered_x = xx[mask]
            covered_y = yy[mask]
            pixel_weights = weights[mask]

            # Normalize weights to conserve total photon count
            pixel_weights = pixel_weights / pixel_weights.sum() if pixel_weights.sum() > 0 else pixel_weights
        else:
            # No blob: single pixel
            covered_x = np.array([int(np.floor(cx))])
            covered_y = np.array([int(np.floor(cy))])
            pixel_weights = np.ones(1)

        return covered_x, covered_y, activation_time, pixel_weights

    def _apply_timepix3_calibrated_model(self, cx, cy, photon_toa, model_params):
        """
        Timepix3-calibrated model with logarithmic TOT response.

        Physics:
        - TOT = a + b × ln(Q/Q_ref)  [Poikela et al. 2014]
        - Per-pixel calibration variation
        - 475 ns deadtime (TPX3 spec)

        Returns:
        --------
        tuple: (covered_x, covered_y, activation_time, pixel_weights, tot_calibration)
            - tot_calibration: dict with 'tot_a' and 'tot_b' for this event
        """
        # Model parameters (from Timepix3 literature)
        gain = model_params.get('gain', 5000)
        sigma_pixels = model_params.get('sigma_pixels', 1.5)  # Blob size
        tot_a = model_params.get('tot_a', 30.0)  # TOT offset (ns)
        tot_b = model_params.get('tot_b', 50.0)  # TOT slope (ns/decade)
        pixel_variation = model_params.get('pixel_variation', 0.05)  # 5% per-pixel variation

        # Per-pixel calibration variation (simulate detector non-uniformity)
        tot_a_actual = tot_a * (1 + np.random.normal(0, pixel_variation))
        tot_b_actual = tot_b * (1 + np.random.normal(0, pixel_variation))

        # Direct detection (no phosphor delay for TPX3)
        activation_time = photon_toa

        # Gaussian blob (similar to gain model but with fixed sigma)
        if sigma_pixels > 0:
            search_radius = 3.0 * sigma_pixels
            i_min = int(np.floor(cx - search_radius - 0.5))
            i_max = int(np.ceil(cx + search_radius + 0.5))
            j_min = int(np.floor(cy - search_radius - 0.5))
            j_max = int(np.ceil(cy + search_radius + 0.5))

            x_grid = np.arange(i_min, i_max + 1)
            y_grid = np.arange(j_min, j_max + 1)
            xx, yy = np.meshgrid(x_grid, y_grid)
            xx = xx.flatten()
            yy = yy.flatten()

            dx = xx - cx
            dy = yy - cy
            dist2 = dx**2 + dy**2
            weights = np.exp(-dist2 / (2 * sigma_pixels**2))

            threshold = 0.01
            mask = weights > threshold

            covered_x = xx[mask]
            covered_y = yy[mask]
            pixel_weights = weights[mask] * gain  # Scale by gain

        else:
            covered_x = np.array([int(np.floor(cx))])
            covered_y = np.array([int(np.floor(cy))])
            pixel_weights = np.array([gain])

        # Return calibration parameters for TOT calculation
        return covered_x, covered_y, activation_time, pixel_weights, {'tot_a': tot_a_actual, 'tot_b': tot_b_actual}

    def _apply_physical_mcp_model(self, cx, cy, photon_toa, model_params):
        """
        Full physics MCP simulation with Poisson gain statistics.

        Physics:
        - Poisson electron multiplication
        - Multi-exponential phosphor decay
        - Energy-dependent effects

        Returns:
        --------
        tuple: (covered_x, covered_y, activation_time, pixel_weights)
        """
        # Model parameters
        gain_mean = model_params.get('gain', 5000)
        gain_noise_factor = model_params.get('gain_noise_factor', 1.3)  # Excess noise factor
        phosphor_type = model_params.get('phosphor_type', 'p43')  # P20, P43, P46
        decay_fast = model_params.get('decay_fast', 50.0)  # Fast component (ns)
        decay_slow = model_params.get('decay_slow', 500.0)  # Slow component (ns)
        fast_fraction = model_params.get('fast_fraction', 0.7)  # Fraction in fast component

        # Poisson gain with excess noise
        # Use Gamma distribution: mean=gain_mean, variance=gain_mean*noise_factor
        if gain_noise_factor > 1.0:
            shape = gain_mean / gain_noise_factor
            scale = gain_noise_factor
            actual_gain = np.random.gamma(shape, scale)
        else:
            actual_gain = gain_mean

        # Multi-exponential phosphor decay
        if np.random.random() < fast_fraction:
            activation_delay = np.random.exponential(decay_fast)
        else:
            activation_delay = np.random.exponential(decay_slow)

        activation_time = photon_toa + activation_delay

        # Blob size depends on actual gain (physics-based)
        sigma_pixels = 1.0 * (actual_gain / 1000) ** 0.4

        # Gaussian distribution
        if sigma_pixels > 0:
            search_radius = 3.0 * sigma_pixels
            i_min = int(np.floor(cx - search_radius - 0.5))
            i_max = int(np.ceil(cx + search_radius + 0.5))
            j_min = int(np.floor(cy - search_radius - 0.5))
            j_max = int(np.ceil(cy + search_radius + 0.5))

            x_grid = np.arange(i_min, i_max + 1)
            y_grid = np.arange(j_min, j_max + 1)
            xx, yy = np.meshgrid(x_grid, y_grid)
            xx = xx.flatten()
            yy = yy.flatten()

            dx = xx - cx
            dy = yy - cy
            dist2 = dx**2 + dy**2
            weights = np.exp(-dist2 / (2 * sigma_pixels**2))

            threshold = 0.01
            mask = weights > threshold

            covered_x = xx[mask]
            covered_y = yy[mask]
            pixel_weights = weights[mask] * actual_gain / 1000  # Normalized weights
        else:
            covered_x = np.array([int(np.floor(cx))])
            covered_y = np.array([int(np.floor(cy))])
            pixel_weights = np.array([actual_gain / 1000])

        return covered_x, covered_y, activation_time, pixel_weights


    def saturate_photons(self, data: pd.DataFrame = None, deadtime: float = 600.0, blob: float = 0.0,
                        blob_variance: float = 0.0, output_format: str = "photons", min_tot: float = 20.0,
                        decay_time: float = 100.0, seed: int = None,
                        detector_model: Union[str, DetectorModel] = None,
                        model_params: dict = None,
                        verbosity: VerbosityLevel = VerbosityLevel.BASIC,
                        **kwargs  # Additional model parameters (e.g., gain=5000, tot_mode="logarithmic")
                        ) -> Union[pd.DataFrame, None]:
        """
        Process traced photons with selectable physical detector models.

        This method simulates various photon detection scenarios using different physical models.
        Select the appropriate model using the detector_model parameter.

        Parameters:
        -----------
        data : pd.DataFrame, optional
            DataFrame containing photon data to process. If None, loads from 'TracedPhotons' directory.
            Must have columns: pixel_x, pixel_y, toa2, id, neutron_id, pulse_id, pulse_time_ns
        deadtime : float, default 600.0
            Deadtime in nanoseconds for pixel saturation. During this window after activation,
            additional photons update TOT but don't create new activation.
        blob : float, default 0.0
            Blob size parameter (interpretation depends on detector_model):
            - image_intensifier: Maximum blob radius in pixels
            - gaussian_diffusion: Gaussian sigma in pixels
            - direct_detection: Ignored
            - wavelength_dependent: Base blob radius (scaled by wavelength)
            - avalanche_gain: Blob radius for gain distribution
        blob_variance : float, default 0.0
            Blob radius variance (for image_intensifier model).
            Actual radius drawn uniformly from [blob - blob_variance, blob].
        output_format : str, default "photons"
            Output format: "photons" for photon-averaged output with nz, pz columns.
        min_tot : float, default 20.0
            Minimum Time-Over-Threshold in nanoseconds.
        decay_time : float, default 100.0
            Exponential decay time constant in nanoseconds (image_intensifier model).
        seed : int, optional
            Random seed for reproducibility.
        detector_model : str or DetectorModel, default "image_intensifier"
            Physical model to use for photon detection. String options (lowercase):
            - "image_intensifier": MCP intensifier with circular blobs (default)
            - "gaussian_diffusion": Charge diffusion with Gaussian PSF
            - "direct_detection": Simple single-pixel detection
            - "wavelength_dependent": Spectral QE and wavelength-dependent gain
            - "avalanche_gain": Poisson gain with afterpulsing
        model_params : dict, optional
            Additional model-specific parameters. See documentation for details.
            Examples:
            - gaussian_diffusion: {'charge_coupling': 0.8}
            - wavelength_dependent: {'qe_wavelength': [400, 500, 600], 'qe_values': [0.1, 0.3, 0.2]}
            - avalanche_gain: {'mean_gain': 100, 'gain_variance': 20, 'afterpulse_prob': 0.01}
        verbosity : VerbosityLevel, default VerbosityLevel.BASIC
            Controls output detail level.

        Returns:
        --------
        pd.DataFrame or None
            Columns: pixel_x, pixel_y, toa2, photon_count, time_diff, id, neutron_id, pulse_id, pulse_time_ns, nz, pz
            - pixel_x, pixel_y: integer pixel positions
            - toa2: time of arrival in nanoseconds (first activation time)
            - time_diff: time-over-threshold in nanoseconds (first to last photon in deadtime)
            - photon_count: number of photon blobs that hit this pixel during deadtime

        Examples:
        ---------
        # Image intensifier (default)
        df = lens.saturate_photons(blob=2.0, decay_time=100.0, deadtime=600.0)

        # Gaussian diffusion model
        df = lens.saturate_photons(
            detector_model="gaussian_diffusion",
            blob=1.5,  # sigma
            model_params={'charge_coupling': 0.85}
        )

        # Direct detection (no blob)
        df = lens.saturate_photons(
            detector_model="direct_detection",
            deadtime=300.0
        )

        # Wavelength-dependent response
        df = lens.saturate_photons(
            detector_model="wavelength_dependent",
            blob=2.0,
            model_params={
                'qe_wavelength': [400, 450, 500, 550, 600],
                'qe_values': [0.1, 0.25, 0.35, 0.30, 0.15]
            }
        )
        """
        # Set default detector model if None
        if detector_model is None:
            detector_model = "image_intensifier"  # Default model

        # Merge kwargs into model_params
        if model_params is None:
            model_params = {}
        else:
            model_params = model_params.copy()  # Don't modify original dict

        # Kwargs take precedence over model_params
        model_params.update(kwargs)

        # Convert string to DetectorModel enum
        if isinstance(detector_model, str):
            model_map = {
                'image_intensifier': DetectorModel.IMAGE_INTENSIFIER,
                'gaussian_diffusion': DetectorModel.GAUSSIAN_DIFFUSION,
                'direct_detection': DetectorModel.DIRECT_DETECTION,
                'wavelength_dependent': DetectorModel.WAVELENGTH_DEPENDENT,
                'avalanche_gain': DetectorModel.AVALANCHE_GAIN,
                'image_intensifier_gain': DetectorModel.IMAGE_INTENSIFIER_GAIN,
                'timepix3_calibrated': DetectorModel.TIMEPIX3_CALIBRATED,
                'physical_mcp': DetectorModel.PHYSICAL_MCP
            }
            detector_model_lower = detector_model.lower()
            if detector_model_lower not in model_map:
                raise ValueError(f"Unknown detector model: '{detector_model}'. "
                               f"Valid options: {list(model_map.keys())}")
            detector_model = model_map[detector_model_lower]
        elif not isinstance(detector_model, DetectorModel):
            raise TypeError(f"detector_model must be a string or DetectorModel enum, got {type(detector_model)}")

        if blob < 0:
            raise ValueError(f"blob must be non-negative, got {blob}")
        if blob_variance < 0:
            raise ValueError(f"blob_variance must be non-negative, got {blob_variance}")
        if blob_variance > blob:
            raise ValueError(f"blob_variance ({blob_variance}) cannot be larger than blob ({blob})")
        if deadtime is not None and deadtime <= 0:
            raise ValueError(f"deadtime must be positive, got {deadtime}")

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            if verbosity > VerbosityLevel.BASIC:
                print(f"Random seed set to: {seed}")

        # Set up input data
        if data is not None:
            dfs = [(data, None)]
            save_results = False
        else:
            traced_photons_dir = self.archive / "TracedPhotons"
            saturated_photons_dir = self.archive / "SaturatedPhotons"
            sim_photons_dir = self.archive / "SimPhotons"
            saturated_photons_dir.mkdir(parents=True, exist_ok=True)

            csv_files = sorted(traced_photons_dir.glob("traced_sim_data_*.csv"))
            dfs = []
            for f in csv_files:
                if f.stat().st_size > 100:
                    sim_file = sim_photons_dir / f.name.replace("traced_", "")
                    sim_df = None
                    if sim_file.exists():
                        try:
                            sim_df = pd.read_csv(sim_file)
                            if sim_df.empty:
                                sim_df = None
                            else:
                                # Ensure nz and pz exist
                                for col in ['nz', 'pz']:
                                    if col not in sim_df.columns:
                                        if verbosity > VerbosityLevel.BASIC:
                                            print(f"Warning: SimPhotons file {sim_file.name} missing {col}. Setting to NaN.")
                                        sim_df[col] = np.nan
                        except Exception as e:
                            if verbosity > VerbosityLevel.BASIC:
                                print(f"Error reading SimPhotons file {sim_file.name}: {str(e)}")
                            sim_df = None
                    
                    try:
                        df = pd.read_csv(f)
                        if not df.empty:
                            dfs.append((df, sim_df))
                    except Exception as e:
                        if verbosity > VerbosityLevel.BASIC:
                            print(f"Error reading {f.name}: {str(e)}")
                        continue

            if not dfs:
                if verbosity > VerbosityLevel.BASIC:
                    print("No valid traced photon files found in 'TracedPhotons' directory.")
                return None
            save_results = True

        if verbosity > VerbosityLevel.BASIC and save_results:
            print(f"Found {len(dfs)} valid traced photon files to process")

        all_results = []
        file_desc = f"Processing {len(dfs)} files" if save_results else "Processing provided data"
        file_iter = tqdm(dfs, desc=file_desc, disable=not (verbosity > VerbosityLevel.BASIC))

        for i, (df, sim_df) in enumerate(file_iter):
            file_name = f"provided_data_{i}.csv" if not save_results else Path(df.name).name
            
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"\nProcessing: {file_name}")
                print(f"  Input shape: {df.shape}")

            if df.empty:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Skipping empty data: {file_name}")
                continue

            # Validate required columns
            required_cols = ['pixel_x', 'pixel_y', 'toa2', 'id', 'neutron_id', 'pulse_id', 'pulse_time_ns']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Skipping {file_name}: missing columns {missing_cols}")
                continue

            # Clean data
            df = df.dropna(subset=required_cols).reset_index(drop=True)
            if df.empty:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Skipping {file_name}: no valid data after removing NaNs")
                continue

            # Sort by arrival time
            if not (df['toa2'].diff().dropna() >= 0).all():
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Warning: Sorting {file_name} by toa2")
                df = df.sort_values('toa2').reset_index(drop=True)

            # Extract data arrays (pixel_x, pixel_y are already in integer pixel units)
            px_float = df['pixel_x'].to_numpy()
            py_float = df['pixel_y'].to_numpy()
            toa = df['toa2'].to_numpy()
            photon_ids = df['id'].to_numpy()
            neutron_ids = df['neutron_id'].to_numpy()
            pulse_ids = df['pulse_id'].to_numpy()
            pulse_times = df['pulse_time_ns'].to_numpy()

            # Get nz, pz from sim_df or df
            if sim_df is not None and len(sim_df) == len(df):
                nz = sim_df['nz'].to_numpy() if 'nz' in sim_df.columns else np.full(len(df), np.nan)
                pz = sim_df['pz'].to_numpy() if 'pz' in sim_df.columns else np.full(len(df), np.nan)
            else:
                nz = df['nz'].to_numpy() if 'nz' in df.columns else np.full(len(df), np.nan)
                pz = df['pz'].to_numpy() if 'pz' in df.columns else np.full(len(df), np.nan)

            if verbosity >= VerbosityLevel.DETAILED:
                print(f"  Pixel coordinate range: x=[{px_float.min():.1f}, {px_float.max():.1f}], "
                    f"y=[{py_float.min():.1f}, {py_float.max():.1f}]")

            # Apply the physical detector model with TOT tracking
            result_rows = []
            pixel_state = {}  # Key: (px_i, py_i), Value: {'first_toa': float, 'last_toa': float, 'photon_count': int, 'idx': int}

            # Initialize model parameters
            if model_params is None:
                model_params = {}

            if verbosity >= VerbosityLevel.DETAILED:
                model_name = detector_model.name
                if detector_model == DetectorModel.IMAGE_INTENSIFIER:
                    variance_msg = f", variance ±{blob_variance} pixels" if blob_variance > 0 else ""
                    print(f"  Model: {model_name} - blob radius {blob} pixels{variance_msg}, deadtime {deadtime}ns, decay {decay_time}ns")
                elif detector_model == DetectorModel.GAUSSIAN_DIFFUSION:
                    coupling = model_params.get('charge_coupling', 1.0)
                    print(f"  Model: {model_name} - Gaussian σ={blob} pixels, coupling={coupling}, deadtime {deadtime}ns")
                elif detector_model == DetectorModel.DIRECT_DETECTION:
                    print(f"  Model: {model_name} - single pixel detection, deadtime {deadtime}ns")
                elif detector_model == DetectorModel.WAVELENGTH_DEPENDENT:
                    print(f"  Model: {model_name} - wavelength-dependent QE, base blob {blob} pixels, deadtime {deadtime}ns")
                elif detector_model == DetectorModel.AVALANCHE_GAIN:
                    mean_gain = model_params.get('mean_gain', 100)
                    print(f"  Model: {model_name} - avalanche gain (mean={mean_gain}), deadtime {deadtime}ns")
                elif detector_model == DetectorModel.IMAGE_INTENSIFIER_GAIN:
                    gain = model_params.get('gain', 5000)
                    print(f"  Model: {model_name} - gain-dependent MCP (gain={gain}), deadtime {deadtime}ns")
                elif detector_model == DetectorModel.TIMEPIX3_CALIBRATED:
                    gain = model_params.get('gain', 5000)
                    tot_a = model_params.get('tot_a', 30.0)
                    tot_b = model_params.get('tot_b', 50.0)
                    print(f"  Model: {model_name} - TPX3 calibrated (gain={gain}, TOT: {tot_a}+{tot_b}×ln(Q)), deadtime {deadtime}ns")
                elif detector_model == DetectorModel.PHYSICAL_MCP:
                    gain = model_params.get('gain', 5000)
                    phosphor = model_params.get('phosphor_type', 'p43')
                    print(f"  Model: {model_name} - full physics MCP (gain={gain}, phosphor={phosphor}), deadtime {deadtime}ns")

            # Collect afterpulses for AVALANCHE_GAIN model
            afterpulse_queue = []

            for idx in range(len(df)):
                cx = px_float[idx]
                cy = py_float[idx]
                photon_toa = toa[idx]

                # Dispatch to appropriate detector model
                if detector_model == DetectorModel.IMAGE_INTENSIFIER:
                    result = self._apply_image_intensifier_model(cx, cy, photon_toa, blob, blob_variance, decay_time)
                    if result is None:
                        continue
                    covered_x, covered_y, activation_time, pixel_weights = result

                elif detector_model == DetectorModel.GAUSSIAN_DIFFUSION:
                    result = self._apply_gaussian_diffusion_model(cx, cy, photon_toa, blob, model_params)
                    if result is None:
                        continue
                    covered_x, covered_y, activation_time, pixel_weights = result

                elif detector_model == DetectorModel.DIRECT_DETECTION:
                    result = self._apply_direct_detection_model(cx, cy, photon_toa)
                    if result is None:
                        continue
                    covered_x, covered_y, activation_time, pixel_weights = result

                elif detector_model == DetectorModel.WAVELENGTH_DEPENDENT:
                    # Get wavelength from data if available
                    wavelength = df.iloc[idx].get('wavelength', 500.0)  # default to 500nm if not available
                    result = self._apply_wavelength_dependent_model(cx, cy, photon_toa, wavelength, blob, decay_time, model_params)
                    if result is None:
                        continue  # Photon not detected due to QE
                    covered_x, covered_y, activation_time, pixel_weights = result

                elif detector_model == DetectorModel.AVALANCHE_GAIN:
                    result = self._apply_avalanche_gain_model(cx, cy, photon_toa, blob, model_params)
                    if result is None:
                        continue
                    covered_x, covered_y, activation_time, pixel_weights, afterpulse_events = result
                    afterpulse_queue.extend(afterpulse_events)

                elif detector_model == DetectorModel.IMAGE_INTENSIFIER_GAIN:
                    result = self._apply_image_intensifier_gain_model(cx, cy, photon_toa, blob, decay_time, model_params)
                    if result is None:
                        continue
                    covered_x, covered_y, activation_time, pixel_weights = result

                elif detector_model == DetectorModel.TIMEPIX3_CALIBRATED:
                    result = self._apply_timepix3_calibrated_model(cx, cy, photon_toa, model_params)
                    if result is None:
                        continue
                    covered_x, covered_y, activation_time, pixel_weights, tot_cal = result
                    # Store TOT calibration for later use (could be used in finalization)

                elif detector_model == DetectorModel.PHYSICAL_MCP:
                    result = self._apply_physical_mcp_model(cx, cy, photon_toa, model_params)
                    if result is None:
                        continue
                    covered_x, covered_y, activation_time, pixel_weights = result

                else:
                    raise ValueError(f"Unknown detector model: {detector_model}")

                if len(covered_x) == 0:
                    continue

                # Process all pixels in blob (common deadtime logic for all models)
                for i, (px_i, py_i) in enumerate(zip(covered_x, covered_y)):
                    pixel_key = (int(px_i), int(py_i))
                    weight = pixel_weights[i] if i < len(pixel_weights) else 1.0

                    # Check if pixel is currently active (in deadtime)
                    if pixel_key in pixel_state:
                        pixel_info = pixel_state[pixel_key]
                        time_since_first = activation_time - pixel_info['first_toa']

                        if deadtime is not None and time_since_first <= deadtime:
                            # Pixel still in deadtime - update last_toa and increment count
                            pixel_info['last_toa'] = activation_time
                            pixel_info['photon_count'] += 1
                            # Accumulate weighted charge for models like GAUSSIAN_DIFFUSION
                            if 'total_charge' in pixel_info:
                                pixel_info['total_charge'] += weight
                            continue
                        else:
                            # Deadtime expired - finalize previous pixel event
                            self._finalize_pixel_event(result_rows, pixel_key, pixel_info,
                                                    photon_ids, neutron_ids, pulse_ids, pulse_times, nz, pz, min_tot)
                            # Remove from active state (will be re-added below)
                            del pixel_state[pixel_key]

                    # Start new pixel activation
                    pixel_state[pixel_key] = {
                        'first_toa': activation_time,
                        'last_toa': activation_time,
                        'photon_count': 1,
                        'idx': idx,  # Store index of first photon that activated this pixel
                        'total_charge': weight  # Track accumulated charge for weighted models
                    }

            # Process afterpulses for AVALANCHE_GAIN model
            if detector_model == DetectorModel.AVALANCHE_GAIN and afterpulse_queue:
                # Sort afterpulses by time
                afterpulse_queue.sort(key=lambda x: x[0])
                for afterpulse_time, ap_cx, ap_cy in afterpulse_queue:
                    # Treat afterpulse as a new photon
                    px_i = int(np.floor(ap_cx))
                    py_i = int(np.floor(ap_cy))
                    pixel_key = (px_i, py_i)

                    if pixel_key in pixel_state:
                        pixel_info = pixel_state[pixel_key]
                        time_since_first = afterpulse_time - pixel_info['first_toa']

                        if deadtime is not None and time_since_first <= deadtime:
                            pixel_info['last_toa'] = afterpulse_time
                            pixel_info['photon_count'] += 1
                            continue
                        else:
                            self._finalize_pixel_event(result_rows, pixel_key, pixel_info,
                                                    photon_ids, neutron_ids, pulse_ids, pulse_times, nz, pz, min_tot)
                            del pixel_state[pixel_key]

                    # Create new activation from afterpulse
                    # Use the original photon's index for metadata
                    pixel_state[pixel_key] = {
                        'first_toa': afterpulse_time,
                        'last_toa': afterpulse_time,
                        'photon_count': 1,
                        'idx': 0,  # Default index
                        'total_charge': 1.0
                    }
            
            # Finalize all remaining pixel events
            for pixel_key, pixel_info in pixel_state.items():
                self._finalize_pixel_event(result_rows, pixel_key, pixel_info,
                                        photon_ids, neutron_ids, pulse_ids, pulse_times, nz, pz, min_tot)

            # Create result DataFrame
            if not result_rows:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  No results after processing {file_name}")
                continue

            result_df = pd.DataFrame(result_rows)
            
            # Sort by time to restore chronological order
            result_df = result_df.sort_values('toa2').reset_index(drop=True)
            
            # Save results if processing files
            if save_results:
                output_file = saturated_photons_dir / f"saturated_{file_name}"
                result_df.to_csv(output_file, index=False)
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  Saved results to: {output_file}")

            # Print stats
            if verbosity > VerbosityLevel.BASIC:
                print(f"  Input photons: {len(df)}, Output events: {len(result_df)}")
                if actual_blob > 0:
                    ratio = len(result_df) / len(df) if len(df) > 0 else 0
                    print(f"  Expansion ratio (blob effect): {ratio:.2f}x")
                if 'photon_count' in result_df.columns:
                    avg_photons = result_df['photon_count'].mean()
                    print(f"  Average photons per pixel event: {avg_photons:.2f}")

            all_results.append(result_df)

        # Combine results
        if all_results:
            try:
                combined_df = pd.concat(all_results, ignore_index=True)
                if verbosity > VerbosityLevel.BASIC:
                    print(f"\nReturning combined DataFrame with {len(combined_df)} rows")
                return combined_df
            except ValueError as e:
                if verbosity > VerbosityLevel.BASIC:
                    print(f"Error combining results: {str(e)}")
                return None

        if verbosity > VerbosityLevel.BASIC:
            print("\nNo results to return")
        return None

    def _finalize_pixel_event(self, result_rows, pixel_key, pixel_info,
                            photon_ids, neutron_ids, pulse_ids, pulse_times, nz, pz, min_tot):
        """Helper function to finalize and add a pixel event to results.
        
        Output format for saturate_photons (photons format):
        - pixel_x, pixel_y: integer pixel positions
        - toa2: time of arrival in nanoseconds (first photon)
        - time_diff: time-over-threshold in nanoseconds (first to last photon)
        - photon_count: number of photon blobs that hit this pixel
        - id, neutron_id, pulse_id, pulse_time_ns: from first photon
        - nz, pz: from first photon
        """
        px_i, py_i = pixel_key
        first_toa = pixel_info['first_toa']
        last_toa = pixel_info['last_toa']
        photon_count = pixel_info['photon_count']
        first_idx = pixel_info['idx']
        
        tot_measured = max(last_toa - first_toa, min_tot)
        
        result_rows.append({
            'pixel_x': px_i,
            'pixel_y': py_i,
            'toa2': first_toa,
            'photon_count': photon_count,
            'time_diff': tot_measured,
            'id': photon_ids[first_idx],
            'neutron_id': neutron_ids[first_idx],
            'pulse_id': pulse_ids[first_idx],
            'pulse_time_ns': pulse_times[first_idx],
            'nz': nz[first_idx],
            'pz': pz[first_idx]
        })
        
    def _add_pixel_event(self, result_rows, px_i, py_i, window_events, df, 
                        photon_ids, neutron_ids, pulse_ids, pulse_times, nz, pz,
                        pixel_size, min_tot, output_format):
        """Helper function to add a pixel event from a deadtime window.
        
        Important: Each pixel event should preserve the identity of the FIRST photon
        that activated that specific pixel, not mix IDs from different neutrons.
        """
        # Extract timing and indices
        toas = [e[0] for e in window_events]
        indices = [e[1] for e in window_events]
        
        toa_first = toas[0]
        toa_last = toas[-1]
        tot_measured = max(toa_last - toa_first, min_tot)
        photon_count = len(window_events)
        
        # CRITICAL: Use the FIRST photon's properties for this pixel event
        # This ensures we don't mix IDs from different neutron events
        first_idx = indices[0]
        
        if output_format == "tpx3":
            # Convert pixel coordinates back to mm (pixel center)
            x_mm = (px_i + 0.5) * pixel_size + self.reduction_ratio
            y_mm = (py_i + 0.5) * pixel_size + self.reduction_ratio
            
            result_rows.append({
                'x': x_mm,
                'y': y_mm,
                'toa': toa_first,
                'tot': tot_measured,
                'pulse_time_ns': pulse_times[first_idx]
            })
        else:  # photons format
            # Use PIXEL coordinates, not first photon coordinates
            x_mm = (px_i + 0.5) * pixel_size + self.reduction_ratio
            y_mm = (py_i + 0.5) * pixel_size + self.reduction_ratio
            
            result_rows.append({
                'x2': x_mm,  # Pixel center position
                'y2': y_mm,  # Pixel center position
                'z2': 0.0,   # Sensor plane
                'pixel_x': px_i,
                'pixel_y': py_i,
                'id': photon_ids[first_idx],           # From first photon to hit this pixel
                'neutron_id': neutron_ids[first_idx],   # From first photon to hit this pixel
                'pulse_id': pulse_ids[first_idx],       # From first photon to hit this pixel
                'pulse_time_ns': pulse_times[first_idx], # From first photon to hit this pixel
                'toa2': toa_first,
                'photon_count': photon_count,
                'time_diff': tot_measured,
                'nz': nz[first_idx],                    # From first photon to hit this pixel
                'pz': pz[first_idx]                     # From first photon to hit this pixel
            })


    def groupby(self, column: str, low: float = None, high: float = None,
                step: float = None, bins: List[float] = None,
                labels: List[str] = None, verbosity: VerbosityLevel = VerbosityLevel.BASIC):
        """
        Group simulation data by a column and create subfolders with filtered data.

        Args:
            column (str): Column name to group by (e.g., 'nz', 'neutronEnergy', 'pulse_id', 'parentName')
            low (float, optional): Lower bound for binning (numerical columns only)
            high (float, optional): Upper bound for binning (numerical columns only)
            step (float, optional): Step size for bins (numerical columns only)
            bins (List[float], optional): Custom bin edges (numerical columns only, alternative to low/high/step)
            labels (List[str], optional): Custom labels for groups/bins
            verbosity (VerbosityLevel): Controls output level

        Returns:
            Lens: Self for method chaining

        Raises:
            ValueError: If column doesn't exist or invalid parameters

        Note:
            For categorical/string columns (e.g., parentName), groups are created automatically
            from unique values. The low/high/step/bins parameters are ignored for such columns.
        """
        if self.data.empty:
            raise ValueError("No simulation data available. Load data first.")

        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data. Available columns: {list(self.data.columns)}")

        # Determine if column is categorical/string or numerical
        is_categorical = pd.api.types.is_string_dtype(self.data[column]) or \
                        pd.api.types.is_object_dtype(self.data[column]) or \
                        pd.api.types.is_categorical_dtype(self.data[column])

        # Create bins or categories
        if is_categorical:
            # For categorical columns, use unique values as groups
            unique_values = self.data[column].dropna().unique()
            unique_values = sorted(unique_values)  # Sort for consistency

            if labels is None:
                labels = [str(val) for val in unique_values]

            if verbosity > VerbosityLevel.BASIC:
                print(f"Grouping by categorical column '{column}' with {len(labels)} unique values")

            bins = None  # No bins for categorical data
        else:
            # For numerical columns, create bins as before
            if bins is None:
                if low is None or high is None or step is None:
                    raise ValueError("For numerical columns, must provide either 'bins' or all of 'low', 'high', 'step'")
                bins = np.arange(low, high + step, step)

            if verbosity > VerbosityLevel.BASIC:
                print(f"Grouping by numerical column '{column}' with {len(bins)-1} bins")
        
        # Create groupby folder structure
        groupby_dir = self.archive / column
        groupby_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata about the groupby operation
        metadata = {
            "column": column,
            "bins": bins.tolist() if isinstance(bins, np.ndarray) else bins,
            "labels": labels,
            "is_categorical": is_categorical,
            "created": pd.Timestamp.now().isoformat(),
            "type": "groupby"
        }

        metadata_file = groupby_dir / ".groupby_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        if verbosity >= VerbosityLevel.DETAILED:
            print(f"Created groupby directory: {groupby_dir}")
            print(f"Saved metadata to: {metadata_file}")

        # Load all SimPhotons data if not already loaded
        sim_photons_dir = self.archive / "SimPhotons"
        if not sim_photons_dir.exists():
            raise FileNotFoundError(f"SimPhotons directory not found: {sim_photons_dir}")

        csv_files = sorted(sim_photons_dir.glob("sim_data_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No sim_data_*.csv files found in {sim_photons_dir}")

        # Load all data if self.data is empty or incomplete
        if len(csv_files) > 1 or self.data.empty:
            if verbosity > VerbosityLevel.BASIC:
                print(f"Loading {len(csv_files)} simulation files...")

            all_data = []
            for csv_file in tqdm(csv_files, desc="Loading data", disable=(verbosity == VerbosityLevel.QUIET)):
                try:
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        # Add file index to track which file each row came from
                        df['_source_file_idx'] = int(csv_file.stem.split('_')[-1])
                        all_data.append(df)
                except Exception as e:
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Warning: Failed to load {csv_file.name}: {e}")

            if not all_data:
                raise ValueError("No valid data loaded from SimPhotons")

            self.data = pd.concat(all_data, ignore_index=True)
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Loaded {len(self.data)} total rows from {len(all_data)} files")

        # Create bin/group labels
        if is_categorical:
            # For categorical data, map values directly to labels
            value_to_label = dict(zip(unique_values, labels))
            self.data['_bin_label'] = self.data[column].map(value_to_label)
        else:
            # For numerical data, use binning
            if labels is None:
                labels = [f"{bins[i]:.3f}" for i in range(len(bins)-1)]

            self.data['_bin_label'] = pd.cut(
                self.data[column],
                bins=bins,
                labels=labels,
                right=False,
                include_lowest=True
            )
        
        # Count rows per bin
        bin_counts = self.data['_bin_label'].value_counts().sort_index()
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"\nBin distribution for '{column}':")
            for label, count in bin_counts.items():
                print(f"  {label}: {count} photons")
        
        # Create subfolders and save filtered data
        for i, label in enumerate(tqdm(labels, desc="Creating groups", disable=(verbosity == VerbosityLevel.QUIET))):
            bin_dir = groupby_dir / label
            bin_dir.mkdir(parents=True, exist_ok=True)
            
            # Create SimPhotons subfolder
            bin_simphotons = bin_dir / "SimPhotons"
            bin_simphotons.mkdir(parents=True, exist_ok=True)
            
            # Filter data for this bin
            bin_data = self.data[self.data['_bin_label'] == label].copy()
            
            if bin_data.empty:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Warning: No data in bin '{label}'")
                continue
            
            # Group by source file index and save
            if '_source_file_idx' in bin_data.columns:
                for file_idx, group_data in bin_data.groupby('_source_file_idx'):
                    # Remove temporary columns
                    save_data = group_data.drop(columns=['_bin_label', '_source_file_idx'], errors='ignore')
                    output_file = bin_simphotons / f"sim_data_{int(file_idx)}.csv"
                    save_data.to_csv(output_file, index=False)
                    
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  Saved {len(save_data)} rows to {output_file.name}")
            else:
                # If no source file index, save all data to single file
                save_data = bin_data.drop(columns=['_bin_label'], errors='ignore')
                output_file = bin_simphotons / "sim_data_0.csv"
                save_data.to_csv(output_file, index=False)
        
        # Clean up temporary columns
        self.data.drop(columns=['_bin_label', '_source_file_idx'], errors='ignore', inplace=True)
        
        # Store groupby info in the Lens object
        self._groupby_column = column
        self._groupby_dir = groupby_dir
        self._groupby_labels = labels
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"\n✓ Groupby complete. Created {len(labels)} groups in: {groupby_dir}")
        
        return self


    def plot(self, opm: "OpticalModel" = None, kind: str = "layout",
                                scale: float = None, 
                                is_dark: bool = False, **kwargs) -> None:
        """
        Plot the lens layout or aberration diagrams.

        Args:
            opm (OpticalModel, optional): Optical model to plot. Defaults to self.opm.
            kind (str): Type of plot ('layout', 'ray', 'opd', 'spot'). Defaults to 'layout'.
            scale (float):  Scale factor for the plot. If None, uses Fit.User_Scale or Fit.All_Same.
            is_dark (bool): Use dark theme for plots. Defaults to False.
            **kwargs: Additional keyword arguments for the figure.
                - dpi (int, optional): Figure resolution. Defaults to 120.
                - figsize (tuple, optional): Figure size as (width, height). Defaults to (8, 2) for layout, (8, 4) for others.
                - frameon (bool, optional): Whether to draw the frame (for layout only). Defaults to False.
                - Other keyword arguments are passed to the plot function.

        Returns:
            None

        Raises:
            ValueError: If opm is None or kind is unsupported.
        """
        opm = opm if opm is not None else self.opm
        if opm is None:
            raise ValueError("No optical model available to plot (self.opm is None).")

        # Set default figsize based on plot kind
        figsize = kwargs.pop("figsize", (8, 2) if kind == "layout" else (8, 4))
        dpi = kwargs.pop("dpi", 120)
        frameon = kwargs.pop("frameon", False)
        # scale = kwargs.pop("scale", 10)
        scale_type = Fit.User_Scale if scale else Fit.All_Same

        # Ensure model is updated and vignetting is applied
        # opm.seq_model.do_apertures = False
        # opm.update_model()
        # apply_paraxial_vignetting(opm)

        if kind == "layout":
            plt.figure(
                FigureClass=InteractiveLayout,
                opt_model=opm,
                frameon=frameon,
                dpi=dpi,
                figsize=figsize,
                do_draw_rays=True,
                do_paraxial_layout=False
            ).plot(**kwargs)
        elif kind == "ray":
            plt.figure(
                FigureClass=RayFanFigure,
                opt_model=opm,
                data_type="Ray",
                scale_type=scale_type,
                is_dark=is_dark,
                dpi=dpi,
                figsize=figsize
            ).plot(**kwargs)
        elif kind == "opd":
            plt.figure(
                FigureClass=RayFanFigure,
                opt_model=opm,
                data_type="OPD",
                scale_type=scale_type,
                is_dark=is_dark,
                dpi=dpi,
                figsize=figsize
            ).plot(**kwargs)
        elif kind == "spot":
            # Remove manual ray tracing - let SpotDiagramFigure handle it
            plt.figure(
                FigureClass=SpotDiagramFigure,
                opt_model=opm,
                scale_type=scale_type,
                user_scale_value=scale,
                is_dark=is_dark,
                frameon=frameon,
                dpi=dpi,
                figsize=figsize
            ).plot(**kwargs)
        else:
            raise ValueError(f"Unsupported plot kind: {kind}, supported kinds are ['layout', 'ray', 'opd', 'spot']")