from rayoptics.environment import OpticalModel, PupilSpec, FieldSpec, WvlSpec, InteractiveLayout
from rayoptics.environment import RayFanFigure, SpotDiagramFigure, Fit
from rayoptics.gui import roafile
from rayoptics.elem.elements import Element
from rayoptics.raytr.trace import apply_paraxial_vignetting, trace_base
import matplotlib.pyplot as plt
from typing import Union, List, Tuple
from pathlib import Path
from multiprocessing import Pool
from functools import partial   
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from rayoptics.raytr import analyses
from lmfit import Parameters, minimize, MinimizerException
import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
import glob
import importlib.resources
from enum import IntEnum
from io import StringIO
from contextlib import redirect_stdout

class VerbosityLevel(IntEnum):
    """Verbosity levels for simulation output."""
    QUIET = 0    # Show nothing except progress bar
    BASIC = 1    # Show progress bar and basic info
    DETAILED = 2 # Show everything

class Lens:
    """
    Lens defining object with integrated data management.
    """
    def __init__(self, archive: str = None, data: "pd.DataFrame" = None,
                 kind: str = "nikkor_58mm", focus: float = None, zmx_file: str = None,
                 focus_gaps: List[Tuple[int, float]] = None, dist_from_obj: float = None,
                 gap_between_lenses: float = 15.0, dist_to_screen: float = 20.0, fnumber: float = 8.0):
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

        Raises:
            ValueError: If invalid lens kind, missing zmx_file for 'zmx_file', or invalid parameters.
        """
        self.kind = kind
        self.focus = focus
        self.zmx_file = zmx_file
        self.focus_gaps = focus_gaps

        # Set default parameters based on lens kind
        if kind == "nikkor_58mm":
            self.dist_from_obj = dist_from_obj if dist_from_obj else 461.535  # Match imported model
            self.gap_between_lenses = 0.0
            self.dist_to_screen = 0.0
            self.fnumber = fnumber if fnumber != 8.0 else 0.98
            self.default_focus_gaps = [(22, 2.68)]  # Default thickness for gap 22
        elif kind == "microscope":
            self.dist_from_obj = dist_from_obj if dist_from_obj else 41.0  # Default distance for microscope
            self.gap_between_lenses = gap_between_lenses
            self.dist_to_screen = dist_to_screen
            self.fnumber = fnumber
            self.default_focus_gaps = [(24, None), (31, None)]  # Will be set after loading
        elif kind == "zmx_file":
            self.dist_from_obj = dist_from_obj
            self.gap_between_lenses = gap_between_lenses
            self.dist_to_screen = dist_to_screen
            self.fnumber = fnumber
            self.default_focus_gaps = focus_gaps or []
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
            save (bool, optional): Save the optical model to a file.

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
        
        opm.rebuild_from_seq()

        # Store default gap thicknesses for microscope
        self.default_focus_gaps = [(24, sm.gaps[24].thi), (31, sm.gaps[31].thi)]
        opm = self.refocus(opm=opm, zfine=focus, save=False)
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

        # Debug: Print number of gaps
        # print(f"Loaded {len(sm.gaps)} gaps from .zmx file")

        # Correct specific thicknesses
        if len(sm.gaps) <= 30:
            raise ValueError(f"Insufficient gaps in .zmx file: {len(sm.gaps)} found, expected at least 31")

        # Correct surface 22 thickness (from 21.2900 mm to 2.68000 mm)
        if abs(sm.gaps[22].thi - 2.68) > 1e-6:
            # print(f"Correcting surface 22 thickness from {sm.gaps[22].thi:.6f} to 2.68000 mm")
            sm.gaps[22].thi = 2.68

        # Correct surface 30 thickness (from 0.00000 mm to 1.00000 mm)
        if abs(sm.gaps[30].thi - 1.0) > 1e-6:
            # print(f"Correcting surface 30 thickness from {sm.gaps[30].thi:.6f} to 1.00000 mm")
            sm.gaps[30].thi = 1.0

        # Ensure stop is at surface 14
        # sm.set_stop(surface=14)

        # Apply paraxial vignetting and update model
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

    def refocus(self, opm: "OpticalModel" = None, zscan: float = 0, zfine: float = 0, fnumber: float = None, save: bool = False) -> OpticalModel:
        """
        Refocus the lens by adjusting gaps relative to default settings.

        Args:
            opm (OpticalModel, optional): Optical model to refocus. Defaults to self.opm0.
            zscan (float): Distance to move the lens assembly in mm relative to default object distance. Defaults to 0.
            zfine (float): Focus adjustment in mm relative to default gap thicknesses (for microscope, gap 24 increases, gap 31 decreases). Defaults to 0.
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
                # print(f"Adjusting nikkor_58mm focus: gap {gap_index} from {sm.gaps[gap_index].thi:.6f} to {new_thi:.6f} mm (zfine={zfine})")
                sm.gaps[gap_index].thi = new_thi
            sm.gaps[0].thi = self.dist_from_obj + zscan
            # print(f"Adjusting nikkor_58mm object distance: from {sm.gaps[0].thi - zscan:.6f} to {sm.gaps[0].thi:.6f} mm (zscan={zscan})")
            
        elif self.kind == "microscope":
            if len(self.default_focus_gaps) != 2:
                raise ValueError("Default focus gaps not set correctly for microscope")
            if zfine != 0:
                # Gap 24: Increase by zfine
                gap_index_24, default_thi_24 = self.default_focus_gaps[0]
                if gap_index_24 >= len(sm.gaps):
                    raise ValueError(f"Invalid gap index {gap_index_24} for microscope lens")
                if default_thi_24 is None:
                    raise ValueError(f"Default thickness not set for gap {gap_index_24}")
                new_thi_24 = default_thi_24 + zfine
                # print(f"Adjusting microscope focus: gap {gap_index_24} from {sm.gaps[gap_index_24].thi:.6f} to {new_thi_24:.6f} mm (zfine=+{zfine})")
                sm.gaps[gap_index_24].thi = new_thi_24

                # Gap 31: Decrease by zfine
                gap_index_31, default_thi_31 = self.default_focus_gaps[1]
                if gap_index_31 >= len(sm.gaps):
                    raise ValueError(f"Invalid gap index {gap_index_31} for microscope lens")
                if default_thi_31 is None:
                    raise ValueError(f"Default thickness not set for gap {gap_index_31}")
                new_thi_31 = default_thi_31 - zfine
                # print(f"Adjusting microscope focus: gap {gap_index_31} from {sm.gaps[gap_index_31].thi:.6f} to {new_thi_31:.6f} mm (zfine=-{zfine})")
                sm.gaps[gap_index_31].thi = new_thi_31
            sm.gaps[0].thi = self.dist_from_obj + zscan
            # print(f"Adjusting microscope object distance: from {sm.gaps[0].thi - zscan:.6f} to {sm.gaps[0].thi:.6f} mm (zscan={zscan})")
            
        elif self.kind == "zmx_file":
            if zfine != 0 and self.focus_gaps is not None:
                for gap_index, scaling_factor in self.focus_gaps:
                    if gap_index >= len(sm.gaps):
                        raise ValueError(f"Invalid gap index {gap_index} for zmx_file lens")
                    default_thi = sm.gaps[gap_index].thi
                    new_thi = default_thi + zfine * scaling_factor
                    # print(f"Adjusting zmx_file focus: gap {gap_index} from {sm.gaps[gap_index].thi:.6f} to {new_thi:.6f} mm (zfine={zfine}, scale={scaling_factor})")
                    sm.gaps[gap_index].thi = new_thi
            sm.gaps[0].thi = self.dist_from_obj + zscan
            # print(f"Adjusting zmx_file object distance: from {sm.gaps[0].thi - zscan:.6f} to {sm.gaps[0].thi:.6f} mm (zscan={zscan})")
            
        else:
            raise ValueError(f"Unsupported lens kind: {self.kind}")
        
        if fnumber is not None:
            osp.pupil = PupilSpec(osp, key=['image', 'f/#'], value=fnumber)
        
        sm.do_apertures = False
        opm.update_model()
        # apply_paraxial_vignetting(opm)
        
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
            List of ray tuples: (id, position, direction, wavelength)
        chunk_size : int
            Number of rays per chunk
            
        Returns:
        --------
        list
            List of chunks, where each chunk is a list of ray tuples
        """
        return [rays[i:i+chunk_size] for i in range(0, len(rays), chunk_size)]


    def trace_rays(self, opm=None, join=False, print_stats=False, n_processes=None,
                   chunk_size=1000, progress_bar=True, timeout=3600, return_df=False,
                   verbosity=VerbosityLevel.BASIC):
        """
        Trace rays from simulation data files and save processed results.

        This method:
        1. Locates all non-empty 'sim_data_*.csv' files in 'SimPhotons' directory under self.archive
        2. Processes ray data in parallel chunks using the specified optical model
        3. Saves traced results to 'TracedPhotons' directory
        4. Optionally returns combined results as a DataFrame

        Parameters:
        -----------
        opm : OpticalModel, optional
            Custom optical model to use instead of self.opm0
        join : bool, default False
            If True, concatenates original data with traced results
        print_stats : bool, default False
            If True, prints tracing statistics
        n_processes : int, optional
            Number of processes for parallel execution (None uses CPU count)
        chunk_size : int, default 1000
            Number of rays per processing chunk
        progress_bar : bool, default True
            If True, displays a progress bar during processing
        timeout : int, default 3600
            Maximum time in seconds for processing each file
        return_df : bool, default False
            If True, returns a combined DataFrame of all processed files
        verbosity : VerbosityLevel, default VerbosityLevel.BASIC
            Controls the level of output detail:
            - QUIET: Only progress bar
            - BASIC: Progress bar + basic info
            - DETAILED: All available information

        Returns:
        --------
        pd.DataFrame or None
            Combined DataFrame of all processed results if return_df=True, otherwise None

        Raises:
        -------
        Exception
            If parallel processing or file operations fail
        """
        sim_photons_dir = self.archive / "SimPhotons"
        traced_photons_dir = self.archive / "TracedPhotons"
        traced_photons_dir.mkdir(parents=True, exist_ok=True)

        # Find all non-empty sim_data_*.csv files
        csv_files = sorted(sim_photons_dir.glob("sim_data_*.csv"))
        valid_files = [f for f in csv_files if f.stat().st_size > 100]

        if not valid_files:
            if verbosity >= VerbosityLevel.BASIC:
                print("No valid simulation data files found in 'SimPhotons' directory.")
            return None

        all_results = []

        # Progress bar for file processing
        file_iter = tqdm(valid_files, desc="Processing files", disable=not progress_bar or verbosity == VerbosityLevel.QUIET)
        
        for csv_file in file_iter:
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Processing file: {csv_file.name}")

            # Load data
            df = pd.read_csv(csv_file)
            if df.empty:
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Skipping empty file: {csv_file.name}")
                continue

            # Verify data integrity
            df['_row_index'] = np.arange(len(df))

            # Get wavelengths for the optical model
            wvl = df["wavelength"].value_counts().to_frame().reset_index()
            wvl["count"] = 1
            wvl_values = wvl.values

            # Convert DataFrame to ray format
            rays = [
                (np.array([row.x, row.y, row.z], dtype=np.float64),
                 np.array([row.dx, row.dy, row.dz], dtype=np.float64),
                 np.array([row.wavelength], dtype=np.float64))
                for row in df.itertuples()
            ]

            # Split rays into chunks
            chunks = []
            index_chunks = []
            for i in range(0, len(rays), chunk_size):
                end = min(i + chunk_size, len(rays))
                chunks.append(rays[i:end])
                index_chunks.append(df['_row_index'].iloc[i:end].tolist())
            
            rays = None  # Clear memory

            # Process chunks in parallel
            process_chunk = partial(
                self._process_ray_chunk,
                lens_kind=self.kind,
                zmx_file=self.zmx_file,
                focus_gaps=self.focus_gaps,
                dist_from_obj=self.dist_from_obj,
                gap_between_lenses=self.gap_between_lenses,
                dist_to_screen=self.dist_to_screen,
                fnumber=self.fnumber,
                wvl_values=wvl_values,
                opm=opm
            )
            try:
                with Pool(processes=n_processes) as pool:
                    results_with_indices = []
                    for chunk_idx, (chunk_result, indices) in enumerate(
                        tqdm(
                            zip(pool.imap(process_chunk, chunks), index_chunks),
                            total=len(chunks),
                            desc=f"Tracing rays ({csv_file.name})",
                            disable=not progress_bar or verbosity == VerbosityLevel.QUIET
                        )
                    ):
                        if chunk_result is None:
                            chunk_result = [None] * len(indices)
                        elif len(chunk_result) != len(indices):
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"Warning: Chunk {chunk_idx} returned {len(chunk_result)} results but expected {len(indices)}")
                            if len(chunk_result) < len(indices):
                                chunk_result = chunk_result + [None] * (len(indices) - len(chunk_result))
                            else:
                                chunk_result = chunk_result[:len(indices)]
                        
                        results_with_indices.extend(zip(chunk_result, indices))
                    
                    pool.close()
                    pool.join()
                    
            except Exception as e:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Error in parallel processing for {csv_file.name}: {str(e)}")
                raise

            # Sort results by original row index
            results_with_indices.sort(key=lambda x: x[1])
            
            # Create result DataFrame
            processed_results = []
            if len(results_with_indices) != len(df):
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Warning: Mismatch in result count. Expected {len(df)}, got {len(results_with_indices)}")
                if len(results_with_indices) < len(df):
                    missing_indices = set(range(len(df))) - set(idx for _, idx in results_with_indices)
                    for idx in missing_indices:
                        results_with_indices.append((None, idx))
                    results_with_indices.sort(key=lambda x: x[1])
                else:
                    results_with_indices = results_with_indices[:len(df)]
            
            for entry, row_idx in results_with_indices:
                if entry is None:
                    processed_results.append({
                        "_row_index": row_idx,
                        "x2": np.nan, "y2": np.nan, "z2": np.nan
                    })
                else:
                    try:
                        ray, path_length, wvl = entry
                        position = ray[0]
                        processed_results.append({
                            "_row_index": row_idx,
                            "x2": position[0], "y2": position[1], "z2": position[2]
                        })
                    except Exception as e:
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"Error processing result entry: {str(e)}")
                        processed_results.append({
                            "_row_index": row_idx,
                            "x2": np.nan, "y2": np.nan, "z2": np.nan
                        })

            result_df = pd.DataFrame(processed_results)
            result_df = result_df.sort_values(by="_row_index").reset_index(drop=True)

            if join:
                result = pd.merge(df, result_df.drop(columns=["_row_index"]), 
                                 left_on="_row_index", right_index=True, how="left")
            else:
                result = result_df.drop(columns=["_row_index"])
                id_cols = ["id", "neutron_id"]
                for col in id_cols:
                    if col in df.columns:
                        result[col] = df[col].values
                if "toa" in df.columns:
                    result["toa2"] = df["toa"].values

            if "_row_index" in result.columns:
                result = result.drop(columns=["_row_index"])

            if print_stats and verbosity >= VerbosityLevel.BASIC:
                total = len(df)
                traced = result.dropna(subset=["x2"]).shape[0]
                percentage = (traced / total) * 100
                print(f"File: {csv_file.name} - Original events: {total}, "
                      f"Traced events: {traced}, Percentage: {percentage:.1f}%")

            output_file = traced_photons_dir / f"traced_{csv_file.name}"
            result.to_csv(output_file, index=False)
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Saved traced results to {output_file}")

            if return_df:
                all_results.append(result)

        if return_df and all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Returning combined DataFrame with {len(combined_df)} rows")
            return combined_df

        return None

    def _process_ray_chunk(self, chunk, lens_kind, zmx_file, focus_gaps, dist_from_obj,
                           gap_between_lenses, dist_to_screen, fnumber, wvl_values, opm=None):
        """Process a chunk of rays at once by reconstructing the optical model."""
        try:
            # Reconstruct the optical model in the worker process
            if opm is not None:
                opt_model = deepcopy(opm)
            else:
                if lens_kind == "nikkor_58mm":
                    opt_model = self.nikkor_58mm(
                        dist_from_obj=dist_from_obj,
                        fnumber=fnumber,
                        save=False
                    )
                elif lens_kind == "microscope":
                    opt_model = self.microscope_nikor_80_200mm_canon_50mm(
                        focus=self.focus,
                        dist_from_obj=dist_from_obj,
                        gap_between_lenses=gap_between_lenses,
                        dist_to_screen=dist_to_screen,
                        fnumber=fnumber,
                        save=False
                    )
                elif lens_kind == "zmx_file":
                    opt_model = self.load_zmx_lens(
                        zmx_file=zmx_file,
                        focus=self.focus,
                        dist_from_obj=dist_from_obj,
                        gap_between_lenses=gap_between_lenses,
                        dist_to_screen=dist_to_screen,
                        fnumber=fnumber,
                        save=False
                    )
                else:
                    raise ValueError(f"Unsupported lens kind: {lens_kind}")

            # Set the spectral region
            opt_model.optical_spec.spectral_region = WvlSpec(wvl_values, ref_wl=1)

            # Trace the rays
            return analyses.trace_list_of_rays(
                opt_model,
                chunk,
                output_filter="last",
                rayerr_filter="summary"
            )
        except Exception as e:
            # Log the error and return empty results
            # print(f"Error processing chunk: {str(e)}")
            return [None] * len(chunk)

    def zscan(self, zfocus_range: Union[np.ndarray, list, float] = 0.,
            zfine_range: Union[np.ndarray, list, float] = 0.,
            data: pd.DataFrame = None, opm: "OpticalModel" = None,
            n_processes: int = None, chunk_size: int = 1000,
            archive: str = None, verbose: VerbosityLevel = VerbosityLevel.QUIET) -> pd.Series:
        """
        Perform a Z-scan to determine the optimal focus by evaluating ray tracing results.

        Parameters:
        -----------
        zfocus_range : Union[np.ndarray, list, float], default 0.
            Range of z-focus positions to scan (scalar or iterable)
        zfine_range : Union[np.ndarray, list, float], default 0.
            Range of z-fine positions to scan (scalar or iterable)
        data : pd.DataFrame, optional
            Input DataFrame with ray data; overrides archive/class data if provided
        opm : OpticalModel, optional
            Custom optical model; uses self.opm0 if None
        n_processes : int, optional
            Number of processes for parallel ray tracing (None uses CPU count)
        chunk_size : int, default 1000
            Number of rays per processing chunk
        archive : str, optional
            Path to archive directory containing 'SimPhotons' with simulation data files
        verbose : VerbosityLevel, default VerbosityLevel.BASIC
            Controls output detail: QUIET (0), BASIC (1), DETAILED (2)

        Returns:
        --------
        pd.Series
            Series mapping each scanned z-value to the combined standard deviation of x2 and y2
        """
        # Load data from archive if provided
        if archive is not None:
            archive_path = Path(archive)
            sim_photons_dir = archive_path / "SimPhotons"
            if not sim_photons_dir.exists():
                raise ValueError(f"SimPhotons directory not found in {archive_path}")

            csv_files = sorted(sim_photons_dir.glob("sim_data_*.csv"))
            valid_dfs = []
            for file in tqdm(csv_files, desc="Loading simulation data", disable=verbose == VerbosityLevel.QUIET):
                try:
                    if file.stat().st_size > 100:
                        df = pd.read_csv(file)
                        if not df.empty:
                            valid_dfs.append(df)
                except Exception as e:
                    if verbose >= VerbosityLevel.DETAILED:
                        print(f"⚠️ Skipping {file.name} due to error: {e}")
                    continue

            if valid_dfs:
                data = pd.concat(valid_dfs, ignore_index=True)
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Loaded {len(valid_dfs)} valid simulation data files with {len(data)} rows.")
            else:
                raise ValueError(f"No valid simulation data files found in {sim_photons_dir}")

        # Fallback to provided or class data
        if data is None:
            data = getattr(self, 'data', None)
            if data is None:
                raise ValueError("No data provided and no class data available.")
            if verbose >= VerbosityLevel.DETAILED:
                print(f"Using class data with {len(data)} rows.")

        opm = deepcopy(opm) if opm else deepcopy(self.opm0)

        # Determine scan type
        zfocus_is_scalar = isinstance(zfocus_range, (float, int))
        zfine_is_scalar = isinstance(zfine_range, (float, int))

        if not zfocus_is_scalar and not zfine_is_scalar:
            raise ValueError("Either zfocus_range or zfine_range must be a scalar, not both iterables.")

        if zfocus_is_scalar:
            scan_range = np.array(zfine_range if isinstance(zfine_range, (np.ndarray, list)) else [zfine_range])
            fixed_value = zfocus_range
            scan_type = "zfine"
        else:
            scan_range = np.array(zfocus_range if isinstance(zfocus_range, (np.ndarray, list)) else [zfocus_range])
            fixed_value = zfine_range
            scan_type = "zfocus"

        results = {}
        min_std = float('inf')
        best_focus = None

        # Single progress bar for zscan
        pbar = tqdm(scan_range, desc=f"Z-scan ({scan_type})", disable=verbose == VerbosityLevel.QUIET)

        for value in pbar:
            if verbose >= VerbosityLevel.DETAILED:
                print(f"\n{'='*50}")
                print(f"Processing {scan_type} = {value:.2f}")

            try:
                new_opm = self.refocus(opm, zfocus=value if scan_type == "zfocus" else fixed_value,
                                    zfine=value if scan_type == "zfine" else fixed_value)
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Refocused OPM with {scan_type}={value:.2f}, fixed_value={fixed_value:.2f}")
            except Exception as e:
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Error refocusing for {scan_type} = {value:.2f}: {str(e)}")
                results[value] = float('inf')
                continue

            # Trace rays with progress bar off for BASIC verbosity
            traced_df = self.trace_rays(opm=new_opm, join=False, print_stats=False,
                                    n_processes=n_processes, chunk_size=chunk_size,
                                    progress_bar=False,  # Disable nested bars
                                    return_df=True, verbosity=verbose)

            if traced_df is None or traced_df.empty:
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Warning: No valid traced data for {scan_type} = {value:.2f}")
                results[value] = float('inf')
                continue

            if 'x2' not in traced_df.columns or 'y2' not in traced_df.columns:
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Warning: Missing x2 or y2 columns for {scan_type} = {value:.2f}")
                results[value] = float('inf')
                continue

            try:
                valid_x2 = traced_df['x2'].dropna()
                valid_y2 = traced_df['y2'].dropna()
                if len(valid_x2) == 0 or len(valid_y2) == 0:
                    if verbose >= VerbosityLevel.DETAILED:
                        print(f"No valid data points for std calculation at {scan_type} = {value:.2f}")
                    results[value] = float('inf')
                    continue

                std_value = np.sqrt(valid_x2.std()**2 + valid_y2.std()**2)
                results[value] = std_value

                if verbose >= VerbosityLevel.DETAILED:
                    print(f"x2 std: {valid_x2.std():.3f}, y2 std: {valid_y2.std():.3f}, combined std: {std_value:.3f}")

                if std_value < min_std:
                    min_std = std_value
                    best_focus = value

                if verbose >= VerbosityLevel.BASIC:
                    pbar.set_description(
                        f"Z-scan ({scan_type}) [current: {value:.2f}, std: {std_value:.3f}, best: {min_std:.3f} @ {best_focus:.2f}]"
                    )

            except Exception as e:
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Error analyzing traced data for {scan_type} = {value:.2f}: {str(e)}")
                if verbose >= VerbosityLevel.DETAILED:
                    import traceback
                    traceback.print_exc()
                results[value] = float('inf')
                continue

        pbar.close()

        if verbose >= VerbosityLevel.DETAILED:
            print("\nFinal results:")
            for k, v in sorted(results.items()):
                print(f"{scan_type} = {k:.2f}: std = {v:.3f}")

        if min_std != float('inf') and best_focus is not None:
            if verbose >= VerbosityLevel.BASIC:
                print(f"Z-scan completed. Best {scan_type}: {best_focus:.2f} with std: {min_std:.3f}")
        else:
            if verbose >= VerbosityLevel.BASIC:
                print("Z-scan completed but no valid results were found.")

        return pd.Series(results)




    def zscan_optimize(self, initial_zfocus: float = 0., initial_zfine: float = 0.,
                    initial_fnumber: float = None,
                    optimize_param: str = "zfocus", zfocus_min: float = None, zfocus_max: float = None,
                    zfine_min: float = None, zfine_max: float = None,
                    fnumber_min: float = None, fnumber_max: float = None,
                    data: pd.DataFrame = None, opm: "OpticalModel" = None,
                    n_processes: int = None, chunk_size: int = 1000, archive: str = None,
                    verbose: VerbosityLevel = VerbosityLevel.BASIC) -> dict:
        """
        Optimize z-focus, z-fine positions, and/or f-number using lmfit minimization to minimize ray position spread
        while maximizing the number of traced photons.

        This method:
        1. Loads simulation data from archive, provided DataFrame, or class data
        2. Optimizes zfocus, zfine, fnumber, or combinations sequentially using lmfit
        3. Balances minimizing std of x2/y2 with maximizing traced photons
        4. Returns the best parameters and minimum objective value achieved

        Parameters:
        -----------
        initial_zfocus : float, default 0.
            Initial guess for z-focus position
        initial_zfine : float, default 0.
            Initial guess for z-fine position
        initial_fnumber : float, optional
            Initial guess for f-number (None uses the default lens f-number)
        optimize_param : str, default "zfocus"
            Parameter to optimize: "zfocus", "zfine", "fnumber", or combinations like "both", 
            "zfocus+fnumber", "zfine+fnumber", "all"
        zfocus_min : float, optional
            Minimum allowable z-focus value
        zfocus_max : float, optional
            Maximum allowable z-focus value
        zfine_min : float, optional
            Minimum allowable z-fine value
        zfine_max : float, optional
            Maximum allowable z-fine value
        fnumber_min : float, optional
            Minimum allowable f-number value
        fnumber_max : float, optional
            Maximum allowable f-number value
        data : pd.DataFrame, optional
            Input ray data; overrides archive/class data if provided
        opm : OpticalModel, optional
            Custom optical model; uses self.opm0 if None
        n_processes : int, optional
            Number of processes for parallel ray tracing (None uses CPU count)
        chunk_size : int, default 1000
            Number of rays per processing chunk
        archive : str, optional
            Path to archive directory with 'SimPhotons' simulation data files
        verbose : VerbosityLevel, default VerbosityLevel.BASIC
            Controls output detail: QUIET (0), BASIC (1), DETAILED (2)

        Returns:
        --------
        dict
            Optimization results with keys:
            - best_zfocus: Optimal z-focus position (if optimized)
            - best_zfine: Optimal z-fine position (if optimized)
            - best_fnumber: Optimal f-number (if optimized)
            - min_std: Minimum standard deviation achieved
            - traced_fraction: Fraction of photons traced at optimal position
            - result: lmfit MinimizeResult object from the last optimization

        Raises:
        -------
        ValueError
            If optimize_param is invalid or no valid data is found
        """
        # Load data from archive if provided
        if archive is not None:
            archive_path = Path(archive)
            sim_photons_dir = archive_path / "SimPhotons"
            if not sim_photons_dir.exists():
                raise ValueError(f"SimPhotons directory not found in {archive_path}")

            csv_files = sorted(sim_photons_dir.glob("sim_data_*.csv"))
            valid_dfs = []
            for file in tqdm(csv_files, desc="Loading simulation data", disable=verbose == VerbosityLevel.QUIET):
                try:
                    if file.stat().st_size > 100:
                        df = pd.read_csv(file)
                        if not df.empty:
                            valid_dfs.append(df)
                except Exception as e:
                    if verbose >= VerbosityLevel.DETAILED:
                        print(f"⚠️ Skipping {file.name} due to error: {e}")
                    continue

            if valid_dfs:
                data = pd.concat(valid_dfs, ignore_index=True)
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Loaded {len(valid_dfs)} valid simulation data files with {len(data)} rows.")
            else:
                raise ValueError(f"No valid simulation data files found in {sim_photons_dir}")

        # Fallback to provided or class data
        if data is None:
            data = getattr(self, 'data', None)
            if data is None:
                raise ValueError("No data provided and no class data available.")
            if verbose >= VerbosityLevel.DETAILED:
                print(f"Using class data with {len(data)} rows.")

        # Clean data by dropping rows with NaN in essential columns
        essential_columns = ['x', 'y', 'z', 'dx', 'dy', 'dz', 'wavelength']
        if not all(col in data.columns for col in essential_columns):
            raise ValueError(f"Data missing required columns: {essential_columns}")
        data = data.dropna(subset=essential_columns)
        if data.empty:
            raise ValueError("Data is empty after removing NaN from essential columns.")
        total_photons = len(data)
        if verbose >= VerbosityLevel.DETAILED:
            print(f"Cleaned data to {total_photons} rows after removing NaN from essential columns.")
        if verbose >= VerbosityLevel.DETAILED:
            print(f"Data NaN summary:\n{data.isna().sum()}")

        opm = deepcopy(opm) if opm else deepcopy(self.opm0)
        
        # Get default f-number if not provided
        if initial_fnumber is None:
            # Extract the default f-number from the optical model
            try:
                initial_fnumber = opm.optical_spec.pupil.value
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Using default f-number from optical model: {initial_fnumber}")
            except:
                # Default to 0.95 for the Nikkor lens if we can't extract it
                initial_fnumber = 0.95
                if verbose >= VerbosityLevel.DETAILED:
                    print(f"Could not extract f-number from optical model, using default: {initial_fnumber}")

        # Validate optimize_param
        valid_params = ["zfocus", "zfine", "fnumber", "both", "zfocus+fnumber", "zfine+fnumber", "all"]
        if optimize_param not in valid_params:
            raise ValueError(f"optimize_param must be one of {valid_params}")

        # Determine which parameters to optimize
        optimize_zfocus = optimize_param in ["zfocus", "both", "zfocus+fnumber", "all"]
        optimize_zfine = optimize_param in ["zfine", "both", "zfine+fnumber", "all"]
        optimize_fnumber = optimize_param in ["fnumber", "zfocus+fnumber", "zfine+fnumber", "all"]

        # Objective function factory with support for f-number
        def create_objective(param_name: str, fixed_params: dict):
            best_result = {
                'z': fixed_params.get('zfocus', initial_zfocus) if param_name == 'zfocus' else 
                    fixed_params.get('zfine', initial_zfine) if param_name == 'zfine' else
                    fixed_params.get('fnumber', initial_fnumber),
                'std': float('inf'), 
                'traced_fraction': 0.0
            }
            iteration_count = [0]

            def objective(params):
                z = params['z'].value
                
                # Create a dictionary of parameters for refocus method
                refocus_kwargs = fixed_params.copy()
                refocus_kwargs[param_name] = z
                
                try:
                    current_opm = self.refocus(opm, **refocus_kwargs)
                except Exception as e:
                    if verbose >= VerbosityLevel.DETAILED:
                        print(f"Iteration {iteration_count[0]}: {param_name}={z:.3f}, refocus failed: {str(e)}")
                    iteration_count[0] += 1
                    return float('inf')

                df = self.trace_rays(opm=current_opm, join=False, print_stats=False,
                                n_processes=n_processes, chunk_size=chunk_size,
                                progress_bar=False, return_df=True, verbosity=verbose)

                if df is None or df.empty:
                    if verbose >= VerbosityLevel.DETAILED:
                        print(f"Iteration {iteration_count[0]}: {param_name}={z:.3f}, trace_rays returned None or empty")
                    iteration_count[0] += 1
                    return float('inf')

                if 'x2' not in df.columns or 'y2' not in df.columns:
                    if verbose >= VerbosityLevel.DETAILED:
                        print(f"Iteration {iteration_count[0]}: {param_name}={z:.3f}, missing x2/y2 columns")
                    iteration_count[0] += 1
                    return float('inf')

                valid_df = df.dropna(subset=['x2', 'y2'])
                traced_count = len(valid_df)
                traced_fraction = traced_count / total_photons if total_photons > 0 else 0.0

                if traced_count < 2:
                    if verbose >= VerbosityLevel.DETAILED:
                        print(f"Iteration {iteration_count[0]}: {param_name}={z:.3f}, insufficient valid data (traced: {traced_count}/{total_photons})")
                    iteration_count[0] += 1
                    return 1e6 + 1000.0 * (1.0 - traced_fraction)  # High base value to avoid NaN issues

                std_x2 = valid_df['x2'].std()
                std_y2 = valid_df['y2'].std()
                std_value = np.sqrt(std_x2**2 + std_y2**2) if not (pd.isna(std_x2) or pd.isna(std_y2)) else float('inf')

                # Objective: minimize std, heavily penalize low traced fraction
                penalty = 1000.0 * (1.0 - traced_fraction)
                objective_value = std_value + penalty if std_value != float('inf') else 1e6 + penalty

                # Different reporting for fnumber optimization
                if verbose >= VerbosityLevel.DETAILED:
                    param_str = f"f/{z:.2f}" if param_name == 'fnumber' else f"{param_name}={z:.3f}"
                    print(f"Iteration {iteration_count[0]}: {param_str}, std={std_value:.3f}, "
                        f"traced={traced_count}/{total_photons} ({traced_fraction:.2%}), penalty={penalty:.3f}, objective={objective_value:.3f}")

                if objective_value < (best_result['std'] + 1000.0 * (1.0 - best_result['traced_fraction'])):
                    best_result['z'] = z
                    best_result['std'] = std_value
                    best_result['traced_fraction'] = traced_fraction

                iteration_count[0] += 1
                return objective_value

            return objective, best_result

        results = {}
        current_zfocus = initial_zfocus
        current_zfine = initial_zfine
        current_fnumber = initial_fnumber

        # Optimize zfocus
        if optimize_zfocus:
            if verbose >= VerbosityLevel.BASIC:
                print(f"Starting optimization for zfocus with initial value {current_zfocus:.3f}")
            
            fixed_params = {'zfine': current_zfine, 'fnumber': current_fnumber}
            objective_func, best_result = create_objective('zfocus', fixed_params)

            params = Parameters()
            params.add('z', value=current_zfocus, min=zfocus_min, max=zfocus_max)

            try:
                result = minimize(objective_func, params, method='nelder')  # Nelder-Mead method
                best_zfocus = float(result.params['z'].value)
                min_std = best_result['std']
                traced_fraction = best_result['traced_fraction']

                results['best_zfocus'] = best_zfocus
                results['min_std'] = min_std
                results['traced_fraction'] = traced_fraction
                results['result'] = result

                if verbose >= VerbosityLevel.BASIC:
                    print(f"Optimized zfocus: {best_zfocus:.3f}, min std: {min_std:.3f}, traced fraction: {traced_fraction:.2%}")

                current_zfocus = best_zfocus

            except MinimizerException as e:
                if verbose >= VerbosityLevel.BASIC:
                    print(f"Optimization of zfocus failed: {str(e)}")
                results['best_zfocus'] = current_zfocus
                results['min_std'] = float('inf')
                results['traced_fraction'] = 0.0
                results['result'] = None

        # Optimize zfine
        if optimize_zfine:
            if verbose >= VerbosityLevel.BASIC:
                print(f"Starting optimization for zfine with initial value {current_zfine:.3f}")
            
            fixed_params = {'zfocus': current_zfocus, 'fnumber': current_fnumber}
            objective_func, best_result = create_objective('zfine', fixed_params)

            params = Parameters()
            params.add('z', value=current_zfine, min=zfine_min, max=zfine_max)

            try:
                result = minimize(objective_func, params, method='nelder')
                best_zfine = float(result.params['z'].value)
                min_std = best_result['std']
                traced_fraction = best_result['traced_fraction']

                results['best_zfine'] = best_zfine
                results['min_std'] = min_std
                results['traced_fraction'] = traced_fraction
                results['result'] = result

                if verbose >= VerbosityLevel.BASIC:
                    print(f"Optimized zfine: {best_zfine:.3f}, min std: {min_std:.3f}, traced fraction: {traced_fraction:.2%}")

                current_zfine = best_zfine

            except MinimizerException as e:
                if verbose >= VerbosityLevel.BASIC:
                    print(f"Optimization of zfine failed: {str(e)}")
                results['best_zfine'] = current_zfine
                results['min_std'] = float('inf')
                results['traced_fraction'] = 0.0
                results['result'] = None

        # Optimize fnumber
        if optimize_fnumber:
            if verbose >= VerbosityLevel.BASIC:
                print(f"Starting optimization for f-number with initial value f/{current_fnumber:.2f}")
            
            fixed_params = {'zfocus': current_zfocus, 'zfine': current_zfine}
            objective_func, best_result = create_objective('fnumber', fixed_params)

            params = Parameters()
            params.add('z', value=current_fnumber, min=fnumber_min, max=fnumber_max)

            try:
                result = minimize(objective_func, params, method='nelder')
                best_fnumber = float(result.params['z'].value)
                min_std = best_result['std']
                traced_fraction = best_result['traced_fraction']

                results['best_fnumber'] = best_fnumber
                results['min_std'] = min_std
                results['traced_fraction'] = traced_fraction
                results['result'] = result

                if verbose >= VerbosityLevel.BASIC:
                    print(f"Optimized f-number: f/{best_fnumber:.2f}, min std: {min_std:.3f}, traced fraction: {traced_fraction:.2%}")

            except MinimizerException as e:
                if verbose >= VerbosityLevel.BASIC:
                    print(f"Optimization of f-number failed: {str(e)}")
                results['best_fnumber'] = current_fnumber
                results['min_std'] = float('inf')
                results['traced_fraction'] = 0.0
                results['result'] = None

        return results


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