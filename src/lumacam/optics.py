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
                kind: str = "nikkor_58mm", focus: float = None, zmx_file: str = None,
                focus_gaps: List[Tuple[int, float]] = None, dist_from_obj: float = None,
                gap_between_lenses: float = 15.0, dist_to_screen: float = 20.0, fnumber: float = 8.0,
                FOV: float = None, magnification: float = None,
                verbosity: VerbosityLevel = VerbosityLevel.BASIC):
        """
        Initialize a Lens object with optical model and data management.

        Args:
            archive (str, optional): Directory path for saving results.
            data (pd.DataFrame, optional): Optical photon data table.
            kind (str, optional): Lens type ('nikkor_58mm', 'microscope', 'zmx_file'). Defaults to 'nikkor_58mm'.
            focus (float, optional): Initial focus adjustment in mm relative to default settings.
            zmx_file (str, optional): Path to .zmx file for custom lens (required when kind='zmx_file').
            focus_gaps (List[Tuple[int, float]], optional): List of (gap_index, scaling_factor) for focus adjustment.
            dist_from_obj (float, optional): Distance from object to first lens in mm. Defaults to 35.0.
            gap_between_lenses (float, optional): Gap between lenses in mm. Defaults to 15.0.
            dist_to_screen (float, optional): Distance from last lens to screen in mm. Defaults to 20.0.
            fnumber (float, optional): F-number of the optical system. Defaults to 8.0.
            FOV (float, optional): Field of view in mm. Defaults to None. for 'nikor_58mm', FOV=120mm and for 'microscope', FOV=10mm, for 'zmx_file', FOV=60mm.
            magnification (float, optional): Manually define magnification. Defaults to None.
            verbosity (VerbosityLevel, optional): Verbosity level for logging. Defaults to VerbosityLevel.BASIC.
        Raises:
            ValueError: If invalid lens kind, missing zmx_file for 'zmx_file', or invalid parameters.
        """
        self.kind = kind
        self.focus = focus
        self.zmx_file = zmx_file
        self.focus_gaps = focus_gaps
        self.FOV = FOV

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
            self.dist_from_obj = dist_from_obj
            self.gap_between_lenses = gap_between_lenses
            self.dist_to_screen = dist_to_screen
            self.fnumber = fnumber
            self.default_focus_gaps = focus_gaps or []
            if self.FOV is None:
                self.FOV = 120.0
        else:
            raise ValueError(f"Unknown lens kind: {kind}, supported lenses are ['nikkor_58mm', 'microscope', 'zmx_file']")

        # Validate inputs
        if kind == "zmx_file" and zmx_file is None:
            raise ValueError("zmx_file must be provided when kind='zmx_file'")
        if kind == "zmx_file" and focus_gaps is None:
            print("Warning: focus_gaps not provided for zmx_file; zfine will have no effect unless specified")

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
            if focus is not None:
                self.opm = self.refocus(zfine=focus, save=False)
        elif self.kind == "microscope":
            self.opm0 = self.microscope_nikor_80_200mm_canon_50mm(focus=focus or 0.0, save=False)
            self.opm = deepcopy(self.opm0)
            if focus is not None:
                self.opm = self.refocus(zfine=focus, save=False)
        elif self.kind == "zmx_file":
            self.opm0 = self.load_zmx_lens(zmx_file, focus=focus, save=False)
            self.opm = deepcopy(self.opm0)
            if focus is not None:
                self.opm = self.refocus(zfine=focus, save=False)
        else:
            raise ValueError(f"Unknown lens kind: {self.kind}")

        # get the multiplication value for converting from mm to pixels
        if magnification is not None:
            self.reduction_ratio = magnification
        else:
            self.reduction_ratio = self.get_first_order_parameters().loc["Reduction Ratio","Value"]

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
        sm.do_apertures = True
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
        osp.pupil = PupilSpec(osp, key=['image', 'f/#'], value=fnumber)
        osp.field_of_view = FieldSpec(osp, key=['object', 'height'], flds=[0., 1])  # Set field of view
        osp.spectral_region = WvlSpec([(486.1327, 0.5), (587.5618, 1.0), (656.2725, 0.5)], ref_wl=1)
        sm.do_apertures = True
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

        # Load the .zmx file first
        sm.gaps[0].thi = dist_from_obj
        osp.field_of_view = FieldSpec(osp, key=['object', 'height'], flds=[0., 60])
        osp.spectral_region = WvlSpec([(486.1327, 0.5), (587.5618, 1.0), (656.2725, 0.5)], ref_wl=1)
        
        # Temporarily set a pupil spec
        osp.pupil = PupilSpec(osp, key=['object', 'epd'], value=10.0)
        sm.do_apertures = True
        
        opm.add_from_file(zmx_path)
        opm.update_model()

        # Correct specific thicknesses
        if len(sm.gaps) <= 30:
            raise ValueError(f"Insufficient gaps in .zmx file: {len(sm.gaps)} found, expected at least 31")

        if abs(sm.gaps[22].thi - 2.68) > 1e-6:
            sm.gaps[22].thi = 2.68

        if abs(sm.gaps[30].thi - 1.0) > 1e-6:
            sm.gaps[30].thi = 1.0

        opm.update_model()
        
        # Find the stop surface (should be surface 14 based on .zmx file)
        stop_idx = None
        for i, ifc in enumerate(sm.ifcs):
            if hasattr(ifc, 'interact_mode') and ifc.interact_mode == 'stop':
                stop_idx = i
                break
        
        if stop_idx is None:
            # If not found by interact_mode, find it by checking which surface is the stop
            stop_idx = sm.stop_surface
        
        print(f"  Stop surface found at index: {stop_idx}")
        
        # Calculate EPD and stop diameter from f-number
        pm = opm.parax_model
        efl = pm.opt_inv / pm.pr_ray[0][mc.slp] if hasattr(pm, 'pr_ray') else 58.0
        epd = efl / fnumber  # Entrance pupil diameter
        
        # The stop diameter should be set to match the EPD
        # (For this lens, entrance pupil is close to the stop surface)
        stop_diameter = epd
        
        # Set the stop surface aperture
        sm.ifcs[stop_idx].surface_od = stop_diameter / 2  # surface_od is radius, not diameter
        
        # Set the pupil using EPD
        osp.pupil = PupilSpec(osp, key=['object', 'epd'], value=epd)
        
        opm.update_model()
        apply_paraxial_vignetting(opm)

        print(f"Nikkor 58mm initialized with f/{fnumber:.2f} (EPD = {epd:.2f} mm, Stop diameter = {stop_diameter:.2f} mm, EFL ≈ {efl:.2f} mm)")

        if save:
            output_path = self.archive / "Nikkor_58mm.roa"
            opm.save_model(str(output_path))

        return opm


    def refocus(self, opm: "OpticalModel" = None, zscan: float = 0, zfine: float = 0, fnumber: float = None, save: bool = False) -> OpticalModel:
        """
        Refocus the lens by adjusting gaps relative to default settings.
        """
        opm = deepcopy(self.opm0) if opm is None else deepcopy(opm)
        sm = opm.seq_model
        osp = opm.optical_spec
        pm = opm.parax_model
        
        # Enable apertures immediately
        sm.do_apertures = True
        
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
            # ... existing microscope code ...
            pass
            
        elif self.kind == "zmx_file":
            # ... existing zmx_file code ...
            pass
            
        else:
            raise ValueError(f"Unsupported lens kind: {self.kind}")
        
        # Update f-number if provided
        if fnumber is not None:
            # Update model first
            sm.do_apertures = True
            opm.update_model()
            
            # Calculate new EPD
            efl = pm.opt_inv / pm.pr_ray[0][mc.slp] if hasattr(pm, 'pr_ray') else 58.0
            epd = efl / fnumber
            
            # Find and update stop surface
            stop_idx = sm.stop_surface
            stop_diameter = epd
            
            # CRITICAL: Update the physical stop aperture diameter
            sm.ifcs[stop_idx].surface_od = stop_diameter / 2  # surface_od is radius
            
            # Set new pupil spec
            osp.pupil = PupilSpec(osp, key=['object', 'epd'], value=epd)
            
            print(f"Changed aperture: f/{fnumber:.2f} (EPD = {epd:.2f} mm, Stop diameter = {stop_diameter:.2f} mm)")
        
        # Final update
        sm.do_apertures = True
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

    def trace_rays(self, opm=None, opm_file=None, zscan=0, zfine=0, fnumber=None,
                deadtime=None, blob=0.0, blob_variance=0.0, decay_time=100, 
                join=False, print_stats=False, n_processes=None, chunk_size=1000, 
                progress_bar=True, timeout=3600, return_df=False, split_method="auto",
                seed: int = None,
                verbosity=VerbosityLevel.BASIC
                ) -> Optional[pd.DataFrame]:
        """
        Trace rays from simulation data files and save processed results, optionally applying pixel saturation and blob effect.
        
        If groupby() was called prior to this method, automatically performs grouped tracing for each group.
        Otherwise, performs standard single-archive tracing.

        This method processes ray data from CSV files in the 'SimPhotons' directory using either
        a provided optical model, a saved optical model file, or by creating a refocused version
        of the default optical model. If deadtime or blob is provided, it applies the saturate_photons
        method with output_format="photons" and saves the results directly to the 'TracedPhotons'
        directory, including photon_count, time_diff, nz, pz, and pulse_id columns. It then calls _write_tpx3
        to generate TPX3 files, using pulse_time_ns from the input data (converted to ps).
        Otherwise, it saves raw traced results to 'TracedPhotons' with nz, pz, and pulse_id columns.

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
        zfine : float, default 0
            Focus adjustment in mm relative to default gap thicknesses. Only used if 
            neither opm nor opm_file is provided.
        fnumber : float, optional
            New f-number for the lens. Applied to refocused model if neither opm nor
            opm_file is provided.
        deadtime : float, optional
            Deadtime in nanoseconds for pixel saturation. If provided, applies
            saturate_photons with output_format="photons" and saves results to
            'TracedPhotons', followed by _write_tpx3.
        blob : float, default 0.0
            Interaction radius in pixels for photon hits. If > 0, each photon hit affects
            all pixels within this radius. If provided, applies saturate_photons and
            triggers _write_tpx3.
        blob_variance : float, default 0.0
            Radius value subtracted from blob radius (in pixels). Only used if blob > 0.
            Each photon's actual blob radius is drawn uniformly from [blob - blob_variance, blob].
            Example: blob=5, blob_variance=2.5 → radius uniformly distributed in [2.5, 5.0] pixels.
        decay_time : float, default 100.0
            Exponential decay time constant in nanoseconds for blob activation timing.
            Single delay drawn per photon, applied to all pixels in its blob.
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
            TPX3 file splitting strategy (only used when deadtime or blob is provided):
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
        ValueError: If both opm and opm_file are provided, or if parameters are
                    invalid (e.g., negative deadtime or blob).
        FileNotFoundError: If opm_file does not exist or if no valid simulation
                            data files are found.
        RuntimeError: If tracing fails for a file.
        """
        # Check if groupby was called - if so, delegate to grouped tracing
        if hasattr(self, '_groupby_dir') and hasattr(self, '_groupby_labels'):
            return self._trace_rays_grouped(
                opm=opm,
                opm_file=opm_file,
                zscan=zscan,
                zfine=zfine,
                fnumber=fnumber,
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
                        deadtime=None, blob=0.0, blob_variance=0.0, decay_time=100, seed: int = None,
                        join=False, print_stats=False, n_processes=None, chunk_size=1000, 
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
                if split_method == "event" and 'pulse_id' not in df.columns:
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

                # Apply saturation if deadtime or blob is provided
                if deadtime is not None or blob > 0:
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
                        verbosity=verbosity
                    )
                    
                    if saturated_df is None or saturated_df.empty:
                        if verbosity > VerbosityLevel.QUIET:
                            print(f"  Saturation produced no results for {csv_file.name}")
                        # Still need to save the result_df with in_tpx3=False for all rows
                        # Don't continue here - proceed to save the CSV with no rows marked for TPX3
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
                else:
                    # No saturation applied - mark all rows as in_tpx3=True
                    result_df['in_tpx3'] = True
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"  No saturation applied - all {len(result_df)} rows marked for TPX3")

                # Filter columns to keep only desired ones
                desired_columns = ['pixel_x', 'pixel_y', 'toa2', 'photon_count', 'time_diff', 
                                   'id', 'neutron_id', 'pulse_id', 'pulse_time_ns', 'in_tpx3']
                
                # Keep only columns that exist in the dataframe
                columns_to_keep = [col for col in desired_columns if col in result_df.columns]
                result_df = result_df[columns_to_keep]
                
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"  Filtered to columns: {columns_to_keep}")

                # Save results to file
                output_file = traced_photons_dir / f"traced_{csv_file.name}"
                result_df.to_csv(output_file, index=False)

                # Call _write_tpx3 if deadtime or blob is provided
                if deadtime is not None or blob > 0:
                    # Extract file index from filename (e.g., sim_data_0.csv -> 0)
                    file_index = int(csv_file.stem.split('_')[-1])
                    
                    # Only pass rows marked as in_tpx3 (ensure column exists)
                    if 'in_tpx3' in result_df.columns:
                        tpx3_data = result_df[result_df['in_tpx3']].copy()
                    else:
                        # Fallback: if column doesn't exist, use all rows
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
                        clean=(file_idx == 0),  # Clean directory only for the first file
                        file_index=file_index
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
                            deadtime=None, blob=0.0, blob_variance=0.0, decay_time=100, seed: int = None,
                            join=False, print_stats=False, n_processes=None, chunk_size=1000, 
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



    def saturate_photons(self, data: pd.DataFrame = None, deadtime: float = 600.0, blob: float = 0.0, 
                        blob_variance: float = 0.0, output_format: str = "photons", min_tot: float = 20.0, 
                        decay_time: float = 100.0, seed: int = None,
                        verbosity: VerbosityLevel = VerbosityLevel.BASIC
                        ) -> Union[pd.DataFrame, None]:
        """
        Process traced photons to simulate an image intensifier coupled to an event camera.

        Physical model (UPDATED):
        1. Photon hits image intensifier at position (pixel_x, pixel_y)
        2. Intensifier creates a circular blob on the camera with variable radius
        3. ALL pixels within the blob are activated SIMULTANEOUSLY at time = photon_toa + exponential_delay
        (single exponential delay drawn per photon, applies to entire blob)
        4. Each pixel has independent deadtime (default 600ns)
        5. During deadtime, additional photon blobs can update the pixel's TOT (time to last photon)
        6. Pixels cannot be re-activated (new TOA) until deadtime expires
        7. TOA = activation time for that pixel (first photon blob to activate it)
        8. TOT = time from first activation to last photon blob hit within deadtime

        Parameters:
        -----------
        data : pd.DataFrame, optional
            DataFrame containing photon data to process. If None, loads from 'TracedPhotons' directory.
            Must have columns: pixel_x, pixel_y, toa2, id, neutron_id, pulse_id, pulse_time_ns
        deadtime : float, default 600.0
            Deadtime in nanoseconds for pixel saturation. During this window after activation,
            additional photons update TOT but don't create new activation.
        blob : float, default 0.0
            Maximum blob radius in pixel units (can be float). Each photon from the intensifier triggers 
            all camera pixels within a circular region.
            Example: blob=0 → 1 pixel per photon, blob=1 → ~9 pixels, blob=2 → ~25 pixels.
        blob_variance : float, default 0.0
            Radius value subtracted from blob radius (in pixels). Only used if blob > 0.
            Each photon's actual blob radius is drawn uniformly from [blob - blob_variance, blob].
            Example: blob=5, blob_variance=2.5 → radius uniformly distributed in [2.5, 5.0] pixels.
        output_format : str, default "photons"
            Output format: "photons" for photon-averaged output with nz, pz columns.
            Note: Use trace_rays() -> saturate_photons() -> _write_tpx3() pipeline for TPX3 files.
        min_tot : float, default 20.0
            Minimum Time-Over-Threshold in nanoseconds.
        decay_time : float, default 100.0
            Exponential decay time constant in nanoseconds for blob activation timing.
            Single delay drawn per photon, applied to all pixels in its blob.
        seed : int, optional
            Random seed for reproducibility. If None, uses random state. If specified, allows
            exact reconstruction of results across multiple runs.
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
        """
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

            # Apply the physical blob model with TOT tracking
            result_rows = []
            pixel_state = {}  # Key: (px_i, py_i), Value: {'first_toa': float, 'last_toa': float, 'photon_count': int, 'idx': int}
            
            if verbosity >= VerbosityLevel.DETAILED:
                variance_msg = f", variance ±{blob_variance} pixels" if blob_variance > 0 else ""
                print(f"  Applying physical blob model: max radius {blob} pixels{variance_msg}, deadtime {deadtime}ns, decay {decay_time}ns")

            for idx in range(len(df)):
                cx = px_float[idx]
                cy = py_float[idx]
                photon_toa = toa[idx]
                
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
                
                if len(covered_x) == 0:
                    continue
                
                # Process all pixels in blob
                for px_i, py_i in zip(covered_x, covered_y):
                    pixel_key = (int(px_i), int(py_i))
                    
                    # Check if pixel is currently active (in deadtime)
                    if pixel_key in pixel_state:
                        pixel_info = pixel_state[pixel_key]
                        time_since_first = activation_time - pixel_info['first_toa']
                        
                        if deadtime is not None and time_since_first <= deadtime:
                            # Pixel still in deadtime - update last_toa and increment count
                            pixel_info['last_toa'] = activation_time
                            pixel_info['photon_count'] += 1
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
                        'idx': idx  # Store index of first photon that activated this pixel
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
            column (str): Column name to group by (e.g., 'nz', 'neutronEnergy', 'pulse_id')
            low (float, optional): Lower bound for binning
            high (float, optional): Upper bound for binning
            step (float, optional): Step size for bins
            bins (List[float], optional): Custom bin edges (alternative to low/high/step)
            labels (List[str], optional): Custom labels for bins
            verbosity (VerbosityLevel): Controls output level
        
        Returns:
            Lens: Self for method chaining
        
        Raises:
            ValueError: If column doesn't exist or invalid parameters
        """
        if self.data.empty:
            raise ValueError("No simulation data available. Load data first.")
        
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in data. Available columns: {list(self.data.columns)}")
        
        # Create bins
        if bins is None:
            if low is None or high is None or step is None:
                raise ValueError("Must provide either 'bins' or all of 'low', 'high', 'step'")
            bins = np.arange(low, high + step, step)
        
        if verbosity > VerbosityLevel.BASIC:
            print(f"Grouping by column '{column}' with {len(bins)-1} bins")
        
        # Create groupby folder structure
        groupby_dir = self.archive / column
        groupby_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata about the groupby operation
        metadata = {
            "column": column,
            "bins": bins.tolist() if isinstance(bins, np.ndarray) else bins,
            "labels": labels,
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
        
        # Add bin labels
        if labels is None:
            labels = [f"{bins[i]:.3f}" for i in range(len(bins)-1)]
        
        # Bin the data
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
            opm (OpticalModel, optional): Optical model to plot. Defaults to self.opm0.
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
        opm = opm if opm is not None else self.opm0
        if opm is None:
            raise ValueError("No optical model available to plot (self.opm0 is None).")

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


    def test_fnumber_effect(self, fnumbers=[0.95, 2.8, 5.6, 8.0, 11.0, 16.0], 
                            num_rays=1000, field_height=0.0):
        """
        Test if f-number changes affect ray tracing by checking entrance pupil size.
        
        Args:
            fnumbers: List of f-numbers to test
            num_rays: Number of rays to trace per f-number
            field_height: Field height for the rays (0 = on-axis)
        
        Returns:
            pd.DataFrame with test results
        """
        import matplotlib.pyplot as plt
        
        results = []
        
        print("="*60)
        print("F-NUMBER DIAGNOSTIC TEST")
        print("="*60)
        
        for fnum in fnumbers:
            print(f"\nTesting f/{fnum}...")
            
            # Create a refocused model with this f-number
            test_opm = self.refocus(fnumber=fnum, save=False)
            
            # Get optical specs
            osp = test_opm.optical_spec
            sm = test_opm.seq_model
            pm = test_opm.parax_model
            
            # Check if aperture is actually set
            print(f"  Pupil spec: {osp.pupil.key} = {osp.pupil.value}")
            print(f"  do_apertures: {sm.do_apertures}")
            
            # Get entrance pupil info
            try:
                enp_radius = pm.enp_radius if hasattr(pm, 'enp_radius') else None
                enp_dist = pm.enp_dist if hasattr(pm, 'enp_dist') else None
                
                if enp_radius is not None:
                    print(f"  Entrance pupil radius: {enp_radius:.3f} mm")
                    print(f"  Entrance pupil distance: {enp_dist:.3f} mm")
                else:
                    print(f"  WARNING: Could not get entrance pupil parameters!")
            except Exception as e:
                print(f"  ERROR getting pupil info: {e}")
                enp_radius = None
            
            # Trace a grid of rays to see vignetting
            ray_data = []
            vig_count = 0
            
            # Create rays at different pupil positions
            pupil_coords = np.linspace(-1, 1, int(np.sqrt(num_rays)))
            
            for px in pupil_coords:
                for py in pupil_coords:
                    if px**2 + py**2 > 1:  # Outside unit circle
                        continue
                        
                    try:
                        # Trace ray from field point through pupil
                        ray_pkg = test_opm.optical_spec.lookup_fld_wvl_focus(0)
                        fld, wvl, foc = ray_pkg
                        
                        # Get ray at pupil coordinates (px, py)
                        ray, op_delta, wvl = trace_base(test_opm, [px, py], fld, wvl)
                        
                        if ray is not None and len(ray) > 0:
                            # Check if ray reached the image
                            if len(ray) == len(sm.ifcs):
                                ray_data.append({
                                    'px': px,
                                    'py': py,
                                    'x_final': ray[-1][0][0],
                                    'y_final': ray[-1][0][1],
                                    'vignetted': False
                                })
                            else:
                                vig_count += 1
                                ray_data.append({
                                    'px': px,
                                    'py': py,
                                    'x_final': np.nan,
                                    'y_final': np.nan,
                                    'vignetted': True
                                })
                    except Exception as e:
                        vig_count += 1
                        
            print(f"  Traced {len(ray_data)} rays, {vig_count} vignetted ({100*vig_count/len(ray_data):.1f}%)")
            
            # Calculate RMS spot size for non-vignetted rays
            valid_rays = [r for r in ray_data if not r['vignetted']]
            if valid_rays:
                x_vals = [r['x_final'] for r in valid_rays]
                y_vals = [r['y_final'] for r in valid_rays]
                rms_spot = np.sqrt(np.mean(np.array(x_vals)**2 + np.array(y_vals)**2))
                print(f"  RMS spot size: {rms_spot:.4f} mm")
            else:
                rms_spot = np.nan
                print(f"  WARNING: All rays vignetted!")
            
            results.append({
                'fnumber': fnum,
                'enp_radius': enp_radius,
                'enp_dist': enp_dist,
                'rays_traced': len(ray_data),
                'rays_vignetted': vig_count,
                'vignetting_pct': 100*vig_count/len(ray_data) if ray_data else 0,
                'rms_spot_mm': rms_spot,
                'ray_data': ray_data
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(results_df[['fnumber', 'enp_radius', 'vignetting_pct', 'rms_spot_mm']].to_string(index=False))
        
        # Plot results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, result in enumerate(results):
            if idx >= 6:
                break
                
            ax = axes[idx]
            ray_data = result['ray_data']
            
            # Plot pupil coordinates colored by vignetting
            vignetted = [r for r in ray_data if r['vignetted']]
            not_vignetted = [r for r in ray_data if not r['vignetted']]
            
            if vignetted:
                ax.scatter([r['px'] for r in vignetted], 
                        [r['py'] for r in vignetted],
                        c='red', s=20, alpha=0.5, label='Vignetted')
            if not_vignetted:
                ax.scatter([r['px'] for r in not_vignetted], 
                        [r['py'] for r in not_vignetted],
                        c='blue', s=20, alpha=0.5, label='Passed')
            
            # Draw unit circle
            circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
            ax.add_patch(circle)
            
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
            try:
                ax.set_title(f'f/{result["fnumber"]}\nEnP radius: {result["enp_radius"]:.2f} mm\nVignetting: {result["vignetting_pct"]:.1f}%')
            except:
                pass
            ax.set_xlabel('Pupil X')
            ax.set_ylabel('Pupil Y')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.archive / 'fnumber_test.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {self.archive / 'fnumber_test.png'}")
        plt.show()
        
        # Additional diagnostic: Check if entrance pupil radius scales correctly
        print("\n" + "="*60)
        print("EXPECTED vs ACTUAL SCALING")
        print("="*60)
        if results_df['enp_radius'].notna().all():
            base_fnum = results_df.iloc[0]['fnumber']
            base_radius = results_df.iloc[0]['enp_radius']
            
            print(f"{'f-number':<10} {'Expected Radius':<20} {'Actual Radius':<20} {'Ratio':<10}")
            print("-"*60)
            for _, row in results_df.iterrows():
                expected_radius = base_radius * base_fnum / row['fnumber']
                actual_radius = row['enp_radius']
                ratio = actual_radius / expected_radius if expected_radius > 0 else 0
                print(f"{row['fnumber']:<10.2f} {expected_radius:<20.3f} {actual_radius:<20.3f} {ratio:<10.3f}")
            
            # The ratio should be close to 1.0 if f-number is working correctly
            # If ratio is constant but not 1.0, the scaling is working but absolute values might be off
        
        return results_df

    def test_focus_sensitivity(self, fnumbers=[0.95, 8.0, 16.0], 
                            zfocus_range=np.linspace(-2, 2, 21),
                            num_rays=100):
        """
        Test how f-number affects focus sensitivity (depth of field).
        
        Args:
            fnumbers: List of f-numbers to test
            zfocus_range: Array of focus positions to scan (mm)
            num_rays: Number of rays to trace per configuration
        
        Returns:
            pd.DataFrame with focus sensitivity results
        """
        import matplotlib.pyplot as plt
        
        print("="*60)
        print("FOCUS SENSITIVITY (DEPTH OF FIELD) TEST")
        print("="*60)
        
        all_results = []
        
        for fnum in fnumbers:
            print(f"\nTesting f/{fnum}...")
            focus_results = []
            
            for zfocus in tqdm(zfocus_range, desc=f"f/{fnum} focus scan"):
                # Refocus the lens
                test_opm = self.refocus(zfine=zfocus, fnumber=fnum, save=False)
                
                # Trace rays through the lens
                ray_data = []
                sm = test_opm.seq_model
                
                # Create rays at different pupil positions
                pupil_coords = np.linspace(-0.9, 0.9, int(np.sqrt(num_rays)))
                
                for px in pupil_coords:
                    for py in pupil_coords:
                        if px**2 + py**2 > 0.81:  # Stay within 0.9 radius
                            continue
                        
                        try:
                            ray_pkg = test_opm.optical_spec.lookup_fld_wvl_focus(0)
                            fld, wvl, foc = ray_pkg
                            ray, op_delta, wvl_val = trace_base(test_opm, [px, py], fld, wvl)
                            
                            if ray is not None and len(ray) == len(sm.ifcs):
                                x_final = ray[-1][0][0]
                                y_final = ray[-1][0][1]
                                ray_data.append({
                                    'x': x_final,
                                    'y': y_final,
                                    'r': np.sqrt(x_final**2 + y_final**2)
                                })
                        except:
                            pass
                
                if ray_data:
                    # Calculate spot statistics
                    x_vals = np.array([r['x'] for r in ray_data])
                    y_vals = np.array([r['y'] for r in ray_data])
                    
                    rms_spot = np.sqrt(np.mean(x_vals**2 + y_vals**2))
                    max_spot = np.sqrt(np.max(x_vals**2 + y_vals**2))
                    
                    focus_results.append({
                        'fnumber': fnum,
                        'zfocus': zfocus,
                        'rms_spot_mm': rms_spot,
                        'max_spot_mm': max_spot,
                        'num_rays': len(ray_data)
                    })
            
            all_results.extend(focus_results)
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Plot focus curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: RMS spot size vs focus
        ax1 = axes[0]
        for fnum in fnumbers:
            data = results_df[results_df['fnumber'] == fnum]
            ax1.plot(data['zfocus'], data['rms_spot_mm'], 
                    marker='o', label=f'f/{fnum}', linewidth=2)
        
        ax1.set_xlabel('Focus Position (mm)', fontsize=12)
        ax1.set_ylabel('RMS Spot Size (mm)', fontsize=12)
        ax1.set_title('Focus Sensitivity (Depth of Field)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Normalized to show DoF comparison
        ax2 = axes[1]
        for fnum in fnumbers:
            data = results_df[results_df['fnumber'] == fnum].copy()
            # Normalize by minimum spot size
            min_spot = data['rms_spot_mm'].min()
            data['normalized_spot'] = data['rms_spot_mm'] / min_spot
            ax2.plot(data['zfocus'], data['normalized_spot'], 
                    marker='o', label=f'f/{fnum}', linewidth=2)
        
        ax2.set_xlabel('Focus Position (mm)', fontsize=12)
        ax2.set_ylabel('Normalized Spot Size', fontsize=12)
        ax2.set_title('Depth of Field Comparison (Normalized)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='1.5x threshold')
        
        plt.tight_layout()
        plt.savefig(self.archive / 'focus_sensitivity_test.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {self.archive / 'focus_sensitivity_test.png'}")
        plt.show()
        
        # Calculate depth of field for each f-number
        print("\n" + "="*60)
        print("DEPTH OF FIELD ANALYSIS")
        print("="*60)
        print(f"{'f-number':<10} {'Min RMS (mm)':<15} {'DoF @ 1.5x (mm)':<20} {'DoF @ 2x (mm)':<20}")
        print("-"*60)
        
        for fnum in fnumbers:
            data = results_df[results_df['fnumber'] == fnum].copy()
            min_spot = data['rms_spot_mm'].min()
            min_focus = data.loc[data['rms_spot_mm'].idxmin(), 'zfocus']
            
            # Find depth of field (range where spot < 1.5x minimum)
            threshold_15 = min_spot * 1.5
            threshold_20 = min_spot * 2.0
            
            in_dof_15 = data[data['rms_spot_mm'] <= threshold_15]
            in_dof_20 = data[data['rms_spot_mm'] <= threshold_20]
            
            if len(in_dof_15) > 0:
                dof_15 = in_dof_15['zfocus'].max() - in_dof_15['zfocus'].min()
            else:
                dof_15 = 0
                
            if len(in_dof_20) > 0:
                dof_20 = in_dof_20['zfocus'].max() - in_dof_20['zfocus'].min()
            else:
                dof_20 = 0
            
            print(f"{fnum:<10.2f} {min_spot:<15.4f} {dof_15:<20.3f} {dof_20:<20.3f}")
        
        print("\nExpected behavior:")
        print("  - Larger f-number (f/16) → WIDER depth of field → flatter curve")
        print("  - Smaller f-number (f/0.95) → NARROWER depth of field → sharper curve")
        
        return results_df

    def test_focus_sensitivity_real(self, fnumbers=[0.95, 8.0, 16.0], 
                                    zfocus_range=np.linspace(-2, 2, 21),
                                    object_point=[0, 0, 0],
                                    num_rays=100):
        """
        Test focus sensitivity by tracing real rays from object space.
        This properly tests depth of field with actual ray angles.
        
        Args:
            fnumbers: List of f-numbers to test
            zfocus_range: Array of focus positions to scan (mm)
            object_point: Object point coordinates [x, y, z] in mm
            num_rays: Number of rays to trace per configuration
        
        Returns:
            pd.DataFrame with focus sensitivity results
        """
        import matplotlib.pyplot as plt
        from rayoptics.raytr.raytrace import trace_raw
        
        print("="*60)
        print("FOCUS SENSITIVITY TEST (Real Ray Angles)")
        print("="*60)
        
        all_results = []
        
        for fnum in fnumbers:
            print(f"\nTesting f/{fnum}...")
            
            # Get the entrance pupil size for this f-number
            test_opm = self.refocus(fnumber=fnum, save=False)
            
            # Try to get actual entrance pupil radius
            sm = test_opm.seq_model
            osp = test_opm.optical_spec
            
            # Calculate entrance pupil size from first order optics
            # For a lens, EPD (entrance pupil diameter) ≈ focal_length / f_number
            try:
                fod = self.get_first_order_parameters(test_opm)
                efl = fod.loc['Effective Focal Length (mm)', 'Value']
                epd = efl / fnum  # Entrance pupil diameter
                epr = epd / 2     # Entrance pupil radius
                print(f"  Effective focal length: {efl:.2f} mm")
                print(f"  Entrance pupil diameter: {epd:.2f} mm (radius: {epr:.2f} mm)")
            except:
                print(f"  WARNING: Could not calculate entrance pupil size")
                epr = 29.5 / fnum  # Fallback approximation for 58mm lens
            
            focus_results = []
            
            for zfocus in tqdm(zfocus_range, desc=f"f/{fnum} focus scan"):
                # Refocus the lens
                test_opm = self.refocus(zfine=zfocus, fnumber=fnum, save=False)
                sm = test_opm.seq_model
                
                # Object is at distance dist_from_obj from first surface
                obj_z = -self.dist_from_obj
                
                # Trace rays from object point to entrance pupil
                ray_data = []
                
                # Create grid of aim points at entrance pupil
                # For simplicity, assume entrance pupil is near first lens surface
                pupil_z = 0  # At first surface
                n_rays_per_side = int(np.sqrt(num_rays))
                
                for i in range(n_rays_per_side):
                    for j in range(n_rays_per_side):
                        # Sample entrance pupil uniformly
                        u = (i + 0.5) / n_rays_per_side  # 0 to 1
                        v = (j + 0.5) / n_rays_per_side  # 0 to 1
                        
                        # Convert to radius and angle
                        r = epr * np.sqrt(u)  # Uniform sampling in area
                        theta = 2 * np.pi * v
                        
                        pupil_x = r * np.cos(theta)
                        pupil_y = r * np.sin(theta)
                        
                        # Ray starts at object
                        ray_start = np.array([object_point[0], object_point[1], obj_z])
                        
                        # Ray aims at this point on entrance pupil
                        aim_point = np.array([pupil_x, pupil_y, pupil_z])
                        
                        # Calculate direction
                        direction = aim_point - ray_start
                        direction = direction / np.linalg.norm(direction)
                        
                        try:
                            # Trace the ray
                            wvl = test_opm.optical_spec.spectral_region.reference_wvl
                            ray_result = trace_raw(test_opm, [ray_start, direction], wvl)
                            
                            if ray_result is not None:
                                ray, op_delta = ray_result
                                
                                # Check if ray reached the image surface
                                if len(ray) == len(sm.ifcs):
                                    x_final = ray[-1][0][0]
                                    y_final = ray[-1][0][1]
                                    ray_data.append({
                                        'x': x_final,
                                        'y': y_final,
                                        'r': np.sqrt(x_final**2 + y_final**2)
                                    })
                        except:
                            pass
                
                if ray_data:
                    # Calculate spot statistics
                    x_vals = np.array([r['x'] for r in ray_data])
                    y_vals = np.array([r['y'] for r in ray_data])
                    
                    rms_spot = np.sqrt(np.mean(x_vals**2 + y_vals**2))
                    max_spot = np.sqrt(np.max(x_vals**2 + y_vals**2))
                    
                    focus_results.append({
                        'fnumber': fnum,
                        'zfocus': zfocus,
                        'rms_spot_mm': rms_spot,
                        'max_spot_mm': max_spot,
                        'num_rays': len(ray_data),
                        'epr': epr
                    })
            
            all_results.extend(focus_results)
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Plot focus curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: RMS spot size vs focus
        ax1 = axes[0]
        for fnum in fnumbers:
            data = results_df[results_df['fnumber'] == fnum]
            ax1.plot(data['zfocus'], data['rms_spot_mm'], 
                    marker='o', label=f'f/{fnum}', linewidth=2, markersize=4)
        
        ax1.set_xlabel('Focus Position (mm)', fontsize=12)
        ax1.set_ylabel('RMS Spot Size (mm)', fontsize=12)
        ax1.set_title('Focus Sensitivity (Depth of Field)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Normalized to show DoF comparison
        ax2 = axes[1]
        for fnum in fnumbers:
            data = results_df[results_df['fnumber'] == fnum].copy()
            # Normalize by minimum spot size
            min_spot = data['rms_spot_mm'].min()
            data['normalized_spot'] = data['rms_spot_mm'] / min_spot
            ax2.plot(data['zfocus'], data['normalized_spot'], 
                    marker='o', label=f'f/{fnum}', linewidth=2, markersize=4)
        
        ax2.set_xlabel('Focus Position (mm)', fontsize=12)
        ax2.set_ylabel('Normalized Spot Size', fontsize=12)
        ax2.set_title('Depth of Field Comparison (Normalized)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='1.5x threshold')
        
        plt.tight_layout()
        plt.savefig(self.archive / 'focus_sensitivity_real_test.png', dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {self.archive / 'focus_sensitivity_real_test.png'}")
        plt.show()
        
        # Calculate depth of field for each f-number
        print("\n" + "="*60)
        print("DEPTH OF FIELD ANALYSIS")
        print("="*60)
        print(f"{'f-number':<10} {'EPR (mm)':<12} {'Min RMS (mm)':<15} {'DoF @ 1.5x (mm)':<20} {'DoF @ 2x (mm)':<20}")
        print("-"*80)
        
        for fnum in fnumbers:
            data = results_df[results_df['fnumber'] == fnum].copy()
            min_spot = data['rms_spot_mm'].min()
            epr = data['epr'].iloc[0]
            
            # Find depth of field (range where spot < threshold × minimum)
            threshold_15 = min_spot * 1.5
            threshold_20 = min_spot * 2.0
            
            in_dof_15 = data[data['rms_spot_mm'] <= threshold_15]
            in_dof_20 = data[data['rms_spot_mm'] <= threshold_20]
            
            if len(in_dof_15) > 0:
                dof_15 = in_dof_15['zfocus'].max() - in_dof_15['zfocus'].min()
            else:
                dof_15 = 0
                
            if len(in_dof_20) > 0:
                dof_20 = in_dof_20['zfocus'].max() - in_dof_20['zfocus'].min()
            else:
                dof_20 = 0
            
            print(f"{fnum:<10.2f} {epr:<12.2f} {min_spot:<15.4f} {dof_15:<20.3f} {dof_20:<20.3f}")
        
        print("\nExpected behavior:")
        print("  - Larger f-number (f/16) → smaller EPR → WIDER depth of field")
        print("  - Smaller f-number (f/0.95) → larger EPR → NARROWER depth of field")
        print("  - DoF ratio f/16 : f/0.95 should be ~16.84x")
        
        return results_df

    def test_focus_sensitivity_from_object(self, fnumbers=[0.95, 8.0, 16.0], 
                                        zfocus_range=np.linspace(-2, 2, 21),
                                        num_rays=100):
        """
        Test focus sensitivity by tracing rays from object point through entrance pupil.
        This is the correct way to see depth of field effects.
        """
        import matplotlib.pyplot as plt
        from rayoptics.raytr.raytrace import trace
        
        print("="*60)
        print("FOCUS SENSITIVITY TEST (From Object Space)")
        print("="*60)
        
        all_results = []
        
        for fnum in fnumbers:
            print(f"\nTesting f/{fnum}...")
            
            # Create model with this f-number
            test_opm = self.refocus(fnumber=fnum, save=False)
            sm = test_opm.seq_model
            osp = test_opm.optical_spec
            
            # Get the EPD value
            epd = osp.pupil.value
            epr = epd / 2
            
            print(f"  Entrance pupil diameter (EPD): {epd:.2f} mm")
            print(f"  Object distance: {sm.gaps[0].thi:.2f} mm")
            
            focus_results = []
            
            for zfocus in tqdm(zfocus_range, desc=f"f/{fnum} focus scan", disable=True):
                # Refocus
                test_opm = self.refocus(zfine=zfocus, fnumber=fnum, save=False)
                sm = test_opm.seq_model
                osp = test_opm.optical_spec
                
                # Object point at (0, 0) on-axis
                obj_pt = np.array([0., 0., -sm.gaps[0].thi])
                
                # Get wavelength
                wvl = osp.spectral_region.reference_wvl
                
                # Trace rays from object through entrance pupil
                ray_data = []
                n_rays_per_side = int(np.sqrt(num_rays))
                
                for i in range(n_rays_per_side):
                    for j in range(n_rays_per_side):
                        # Sample entrance pupil (assume at z=0, first lens surface)
                        # Physical coordinates in mm
                        u = -epr + 2 * epr * (i + 0.5) / n_rays_per_side
                        v = -epr + 2 * epr * (j + 0.5) / n_rays_per_side
                        
                        # Skip if outside circular pupil
                        if u**2 + v**2 > epr**2:
                            continue
                        
                        # Entrance pupil point
                        pupil_pt = np.array([u, v, 0.])
                        
                        # Ray direction from object to pupil
                        ray_dir = pupil_pt - obj_pt
                        ray_dir = ray_dir / np.linalg.norm(ray_dir)
                        
                        try:
                            # Trace from object point with direction
                            # Initial ray at object surface
                            ray0 = np.array([obj_pt, ray_dir])
                            
                            # Trace through the system
                            ray_result = trace(test_opm, ray0, wvl)
                            
                            if ray_result is not None:
                                # Get final ray position at image
                                final_ray = ray_result[-1]
                                x_final = final_ray[0][0]
                                y_final = final_ray[0][1]
                                
                                ray_data.append({
                                    'x': x_final,
                                    'y': y_final,
                                    'r': np.sqrt(x_final**2 + y_final**2)
                                })
                        except:
                            pass
                
                if len(ray_data) > 0:
                    x_vals = np.array([r['x'] for r in ray_data])
                    y_vals = np.array([r['y'] for r in ray_data])
                    
                    rms_spot = np.sqrt(np.mean(x_vals**2 + y_vals**2))
                    
                    focus_results.append({
                        'fnumber': fnum,
                        'zfocus': zfocus,
                        'rms_spot_mm': rms_spot,
                        'num_rays': len(ray_data),
                        'epr': epr
                    })
            
            print(f"  Traced {len(focus_results)} focus positions with {focus_results[0]['num_rays'] if focus_results else 0} rays each")
            all_results.extend(focus_results)
        
        if not all_results:
            print("\nERROR: No rays traced!")
            return None
        
        results_df = pd.DataFrame(all_results)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Absolute
        ax1 = axes[0]
        for fnum in fnumbers:
            data = results_df[results_df['fnumber'] == fnum]
            if len(data) > 0:
                ax1.plot(data['zfocus'], data['rms_spot_mm'], 
                        marker='o', label=f'f/{fnum}', linewidth=2, markersize=4)
        ax1.set_xlabel('Focus Position (mm)')
        ax1.set_ylabel('RMS Spot Size (mm)')
        ax1.set_title('Focus Sensitivity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Normalized
        ax2 = axes[1]
        colors = ['blue', 'green', 'red']
        for idx, fnum in enumerate(fnumbers):
            data = results_df[results_df['fnumber'] == fnum].copy()
            if len(data) > 0:
                min_spot = data['rms_spot_mm'].min()
                data['normalized'] = data['rms_spot_mm'] / min_spot
                ax2.plot(data['zfocus'], data['normalized'], 
                        marker='o', label=f'f/{fnum}', linewidth=2, markersize=4,
                        color=colors[idx % len(colors)])
        
        ax2.set_xlabel('Focus Position (mm)')
        ax2.set_ylabel('Normalized Spot Size')
        ax2.set_title('Depth of Field (Normalized)')
        ax2.axhline(y=1.5, color='red', linestyle='--', alpha=0.3, label='1.5x')
        ax2.axhline(y=2.0, color='orange', linestyle='--', alpha=0.3, label='2.0x')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.9, 3.0])
        
        plt.tight_layout()
        plt.savefig(self.archive / 'focus_sensitivity_from_object.png', dpi=150)
        print(f"\nPlot saved to: {self.archive / 'focus_sensitivity_from_object.png'}")
        plt.show()
        
        # DoF analysis
        print("\n" + "="*60)
        print("DEPTH OF FIELD ANALYSIS")
        print("="*60)
        print(f"{'f-number':<10} {'EPR (mm)':<12} {'Min RMS':<12} {'Best Focus':<12} {'DoF@1.5x':<12} {'DoF@2x':<12}")
        print("-"*80)
        
        for fnum in fnumbers:
            data = results_df[results_df['fnumber'] == fnum].copy()
            if len(data) == 0:
                print(f"{fnum:<10.2f} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
                continue
            
            min_spot = data['rms_spot_mm'].min()
            best_focus = data.loc[data['rms_spot_mm'].idxmin(), 'zfocus']
            epr = data['epr'].iloc[0]
            
            in_dof_15 = data[data['rms_spot_mm'] <= min_spot * 1.5]
            in_dof_20 = data[data['rms_spot_mm'] <= min_spot * 2.0]
            
            dof_15 = (in_dof_15['zfocus'].max() - in_dof_15['zfocus'].min()) if len(in_dof_15) > 0 else 0
            dof_20 = (in_dof_20['zfocus'].max() - in_dof_20['zfocus'].min()) if len(in_dof_20) > 0 else 0
            
            print(f"{fnum:<10.2f} {epr:<12.2f} {min_spot:<12.4f} {best_focus:<12.2f} {dof_15:<12.3f} {dof_20:<12.3f}")
        
        # Calculate DoF ratios
        print("\n" + "="*60)
        print("DoF RATIO ANALYSIS")
        print("="*60)
        
        dof_values = {}
        for fnum in fnumbers:
            data = results_df[results_df['fnumber'] == fnum].copy()
            if len(data) > 0:
                min_spot = data['rms_spot_mm'].min()
                in_dof_20 = data[data['rms_spot_mm'] <= min_spot * 2.0]
                dof_20 = (in_dof_20['zfocus'].max() - in_dof_20['zfocus'].min()) if len(in_dof_20) > 0 else 0
                dof_values[fnum] = dof_20
        
        if len(dof_values) >= 2:
            base_fnum = min(dof_values.keys())
            base_dof = dof_values[base_fnum]
            print(f"\nBase: f/{base_fnum} DoF = {base_dof:.3f} mm")
            for fnum, dof in dof_values.items():
                if fnum != base_fnum:
                    ratio = dof / base_dof if base_dof > 0 else 0
                    expected_ratio = fnum / base_fnum
                    print(f"f/{fnum}: DoF = {dof:.3f} mm, ratio = {ratio:.2f}x (expected ~{expected_ratio:.2f}x)")
        
        return results_df