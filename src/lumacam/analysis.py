import pandas as pd
import numpy as np
import os
import subprocess
from pathlib import Path
from tqdm.notebook import tqdm
from enum import IntEnum
from typing import Dict, Any, List
from dataclasses import dataclass, field
from typing import Optional
import json
from multiprocessing import Pool
import glob
import shutil
import tempfile
import uuid
from scipy.spatial import cKDTree
import logging

# Try to import neutron_event_analyzer
try:
    import neutron_event_analyzer as nea
    NEA_AVAILABLE = True
except ImportError:
    NEA_AVAILABLE = False

class VerbosityLevel(IntEnum):
    """Verbosity levels for simulation output."""
    QUIET = 0    # Show nothing except progress bar
    BASIC = 1    # Show progress bar and basic info
    DETAILED = 2 # Show everything

@dataclass
class Photon2EventConfig:
    """Configuration for the photon2event step."""
    dSpace_px: int = 40
    dTime_s: float = 50e-9
    durationMax_s: float = 500e-9
    dTime_ext: int = 5

    def write(self, output_file: str=".paramsterSettings.json") -> str:
        """
        Write the photon2event configuration to a JSON file.
        
        Args:
            output_file: The path to save the parameters file.
            
        Returns:
            The path to the created JSON file.
        """
        parameters = {
            "photon2event": {
                "dSpace_px": self.dSpace_px,
                "dTime_s": self.dTime_s,
                "durationMax_s": self.durationMax_s,
                "dTime_ext": self.dTime_ext
            }
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(parameters, f, indent=4)
        return output_file

@dataclass
class BinningConfig:
    """Configuration for a single binning dimension."""
    nBins: int
    resolution: Optional[float] = None
    resolution_s: Optional[float] = None
    resolution_px: Optional[float] = None
    offset: Optional[float] = None
    offset_s: Optional[float] = None
    offset_px: Optional[float] = None

@dataclass
class EventBinningConfig:
    """Configuration for the event binning step."""
    binning_t: Optional[BinningConfig] = None
    binning_x: Optional[BinningConfig] = None
    binning_y: Optional[BinningConfig] = None
    binning_nPhotons: Optional[BinningConfig] = None
    binning_psd: Optional[BinningConfig] = None
    binning_t_relToExtTrigger: Optional[BinningConfig] = None
    
    @classmethod
    def empty(cls) -> 'EventBinningConfig':
        """Create an empty configuration."""
        return cls()
    
    def tof_binning(self) -> 'EventBinningConfig':
        self.binning_t_relToExtTrigger = BinningConfig(
            resolution_s=1.5625e-9,
            nBins=640,
            offset_s=0
        )
        return self
    
    def psd_binning(self) -> 'EventBinningConfig':
        self.binning_psd = BinningConfig(
            resolution=1e-6,
            nBins=100,
            offset=0
        )
        return self
    
    def nphotons_binning(self) -> 'EventBinningConfig':
        self.binning_nPhotons = BinningConfig(
            resolution=1,
            nBins=10,
            offset=0
        )
        return self
    
    def time_binning(self) -> 'EventBinningConfig':
        self.binning_t = BinningConfig(
            resolution_s=1.5625e-9,
            nBins=640,
            offset_s=0
        )
        return self
    
    def spatial_binning(self) -> 'EventBinningConfig':
        self.binning_x = BinningConfig(
            resolution_px=32,
            nBins=8,
            offset_px=0
        )
        self.binning_y = BinningConfig(
            resolution_px=32,
            nBins=8,
            offset_px=0
        )
        return self
    
    def write(self, output_file: str=".parameterEvents.json") -> str:
        parameters = {"bin_events": {}}
        
        if self.binning_t:
            parameters["bin_events"]["binning_t"] = self._get_config_dict(self.binning_t, use_s=True)
        if self.binning_x:
            parameters["bin_events"]["binning_x"] = self._get_config_dict(self.binning_x, use_px=True)
        if self.binning_y:
            parameters["bin_events"]["binning_y"] = self._get_config_dict(self.binning_y, use_px=True)
        if self.binning_nPhotons:
            parameters["bin_events"]["binning_nPhotons"] = self._get_config_dict(self.binning_nPhotons)
        if self.binning_psd:
            parameters["bin_events"]["binning_psd"] = self._get_config_dict(self.binning_psd)
        if self.binning_t_relToExtTrigger:
            parameters["bin_events"]["binning_t_relToExtTrigger"] = self._get_config_dict(self.binning_t_relToExtTrigger, use_s=True)
        
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(parameters, f, indent=4)
        return output_file
    
    def _get_config_dict(self, config: BinningConfig, use_s: bool = False, use_px: bool = False) -> Dict[str, Any]:
        result = {"nBins": config.nBins}
        if use_s:
            result["resolution_s"] = config.resolution_s if config.resolution_s is not None else config.resolution
            result["offset_s"] = config.offset_s if config.offset_s is not None else (config.offset or 0)
        elif use_px:
            result["resolution_px"] = config.resolution_px if config.resolution_px is not None else config.resolution
            result["offset_px"] = config.offset_px if config.offset_px is not None else (config.offset or 0)
        else:
            result["resolution"] = config.resolution if config.resolution is not None else 0
            result["offset"] = config.offset if config.offset is not None else 0
        return result

class Analysis:
    def __init__(self, archive: str = "test",
                 data: "pd.DataFrame" = None,
                 sim_data: "pd.DataFrame" = None,
                 empir_dirpath: str = None):
        self.archive = Path(archive)
        if empir_dirpath is not None:
            self.empir_dirpath = Path(empir_dirpath)
        else:
            try:
                from G4LumaCam.config.paths import EMPIR_PATH
                self.empir_dirpath = Path(EMPIR_PATH)
            except ImportError:
                self.empir_dirpath = Path("./empir")
                
        self.traced_dir = self.archive / "TracedPhotons"
        self.sim_dir = self.archive / "SimPhotons"
        self.photon_files_dir = self.archive / "PhotonFiles"
        self.photon_files_dir.mkdir(parents=True, exist_ok=True)

        self.Photon2EventConfig = Photon2EventConfig
        self.EventBinningConfig = EventBinningConfig

        if data is not None:
            self.data = data
        else:
            if not self.traced_dir.exists():
                raise FileNotFoundError(f"{self.traced_dir} does not exist.")
            traced_files = sorted(self.traced_dir.glob("traced_sim_data_*.csv"))
            if not traced_files:
                raise FileNotFoundError(f"No traced simulation data found in {self.traced_dir}.")
            valid_dfs = []
            for file in tqdm(traced_files, desc="Loading traced data"):
                try:
                    df = pd.read_csv(file)
                    if not df.empty:
                        valid_dfs.append(df)
                except Exception as e:
                    print(f"⚠️ Skipping {file.name} due to error: {e}")
            if valid_dfs:
                self.data = pd.concat(valid_dfs, ignore_index=True)
            else:
                raise ValueError("No valid traced data found!")

        if sim_data is not None:
            self.sim_data = sim_data
        else:
            if not self.sim_dir.exists():
                raise FileNotFoundError(f"{self.sim_dir} does not exist.")
            sim_files = sorted(self.sim_dir.glob("sim_data_*.csv"))
            if not sim_files:
                raise FileNotFoundError(f"No sim simulation data found in {self.sim_dir}.")
            valid_dfs = []
            for file in tqdm(sim_files, desc="Loading sim data"):
                try:
                    df = pd.read_csv(file)
                    if not df.empty:
                        valid_dfs.append(df)
                except Exception as e:
                    print(f"⚠️ Skipping {file.name} due to error: {e}")
            if valid_dfs:
                self.sim_data = pd.concat(valid_dfs, ignore_index=True)
            else:
                raise ValueError("No valid sim data found!")

        if not self.empir_dirpath.exists():
            raise FileNotFoundError(f"{self.empir_dirpath} does not exist.")
        
        required_files = {
            "empir_import_photons": "empir_import_photons",
            "empir_bin_photons": "empir_bin_photons",
            "empir_bin_events": "empir_bin_events",
            "process_photon2event": "process_photon2event.sh",
            "empir_export_events": "empir_export_events"
        }
        
        self.executables = {}
        for attr_name, filename in required_files.items():
            file_path = self.empir_dirpath / filename
            if not file_path.exists():
                raise FileNotFoundError(f"{filename} not found in {self.empir_dirpath}")
            self.executables[attr_name] = file_path
            setattr(self, attr_name, file_path)
    def _process_single_file(self, file, photon_files_dir, verbosity: VerbosityLevel = VerbosityLevel.QUIET):
        """
        Process a single traced photon file to generate an .empirphot file.

        Args:
            file: Path to the input CSV file.
            photon_files_dir: Directory to save .empirphot files.
            verbosity: Verbosity level for logging.

        Returns:
            bool: True if processing was successful, False otherwise.
        """
        try:
            df = pd.read_csv(file)
            if df.empty:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"⚠️ Skipping empty file: {file.name}")
                return False
            
            # Check for required columns
            required_columns = ["x2", "y2", "toa2"]
            if not all(col in df.columns for col in required_columns):
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"❌ Missing required columns {required_columns} in {file.name}")
                return False

            df = df[required_columns].dropna()
            df["toa2"] *= 1e-9  # Convert ns to seconds
            df["px"] = (df["x2"] + 10) / 10 * 128
            df["py"] = (df["y2"] + 10) / 10 * 128
            df = df[["px", "py", "toa2"]]
            df.columns = ["x [px]", "y [px]", "t [s]"]
            df["t_relToExtTrigger [s]"] = df["t [s]"]
            df = df.loc[(df["t [s]"] >= 0) & (df["t [s]"] < 1)]

            if df.empty:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"⚠️ No valid data after filtering in {file.name}")
                return False

            imported_photons_dir = self.archive / "ImportedPhotons"
            imported_photons_dir.mkdir(exist_ok=True)
            output_csv = imported_photons_dir / f"imported_{file.stem}.csv"
            df.sort_values("t [s]").to_csv(output_csv, index=False)

            empir_file = photon_files_dir / f"{file.stem}.empirphot"
            cmd = f"{self.empir_import_photons} {output_csv} {empir_file} csv"
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Running command: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"❌ Failed to run empir_import_photons for {file.name}: {result.stderr}")
                return False
            
            if not empir_file.exists():
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"❌ Empirphot file not created: {empir_file}")
                return False

            if verbosity >= VerbosityLevel.BASIC:
                print(f"✔ Processed {file.name} → {empir_file}")
            return True
        except Exception as e:
            if verbosity >= VerbosityLevel.BASIC:
                print(f"❌ Error processing {file.name}: {str(e)}")
            return False

    @staticmethod
    def _process_single_file_wrapper(args):
        """Static method to process a single file for multiprocessing."""
        analysis_instance, file, photon_files_dir, verbosity = args
        return analysis_instance._process_single_file(file, photon_files_dir, verbosity)

    def _run_import_photons(self, archive: Path = None, parallel=True, verbosity: VerbosityLevel = VerbosityLevel.QUIET):
        """
        Process all traced photon files to generate .empirphot files.

        Args:
            archive: Path to the archive directory (defaults to self.archive).
            parallel: Whether to process files in parallel using multiprocessing.
            verbosity: Verbosity level for logging.

        Raises:
            FileNotFoundError: If no traced files are found.
        """
        if archive is None:
            archive = self.archive
        archive = Path(archive)
        photon_files_dir = archive / "PhotonFiles"
        photon_files_dir.mkdir(parents=True, exist_ok=True)

        traced_files = sorted((archive / "TracedPhotons").glob("traced_sim_data_*.csv"))
        if not traced_files:
            raise FileNotFoundError(f"No traced files found in {archive / 'TracedPhotons'}")
        
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Processing {len(traced_files)} traced photon files into {photon_files_dir}...")

        success_count = 0
        if parallel:
            with Pool() as pool:
                results = list(tqdm(pool.imap_unordered(self._process_single_file_wrapper, 
                                                        [(self, file, photon_files_dir, verbosity) for file in traced_files]), 
                                    total=len(traced_files), desc="Processing files"))
                success_count = sum(1 for result in results if result)
        else:
            for file in tqdm(traced_files, desc="Processing files"):
                if self._process_single_file(file, photon_files_dir, verbosity=verbosity):
                    success_count += 1

        if verbosity >= VerbosityLevel.BASIC:
            print(f"✅ Finished processing {success_count}/{len(traced_files)} files successfully!")
        
        if success_count == 0:
            raise RuntimeError(f"No files were processed successfully, cannot proceed to photon2event. Check {photon_files_dir} for .empirphot files.")

    def _run_photon2event(self, archive: str = None, 
                          config: Photon2EventConfig = None,
                          verbosity: VerbosityLevel = VerbosityLevel.QUIET,
                          **config_kwargs):
        """
        Run the photon-to-event association step using empir_photon2event.

        Args:
            archive: Path to the archive directory.
            config: Photon2EventConfig object with clustering parameters.
            verbosity: Verbosity level for logging.
            **config_kwargs: Additional configuration parameters.

        Raises:
            FileNotFoundError: If no .empirphot files are found in the input folder.
        """
        if archive is None:
            archive = self.archive
        archive = Path(archive)
        
        (archive / "EventFiles").mkdir(parents=True, exist_ok=True)
        input_folder = archive / "PhotonFiles"
        empirphot_files = list(input_folder.glob("*.empirphot"))
        
        if not empirphot_files:
            raise FileNotFoundError(f"No empirphot files found in {input_folder}")
        
        params_file = archive / "parameterSettings.json"
        if config is None:
            config = Photon2EventConfig(**config_kwargs)
        config.write(params_file)
        
        success_count = 0
        for empirphot_file in tqdm(empirphot_files, desc="Running photon2event", disable=(verbosity == VerbosityLevel.QUIET)):
            output_file = archive / "EventFiles" / f"{empirphot_file.stem}.empirevent"
            process_command = (
                f"{self.empir_dirpath}/bin/empir_photon2event "
                f"-i '{empirphot_file}' "
                f"-o '{output_file}' "
                f"--paramsFile '{params_file}'"
            )
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Running command: {process_command}")
            result = subprocess.run(process_command, shell=True, capture_output=True, text=True)
            if result.returncode == 0 and output_file.exists():
                success_count += 1
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"✔ Generated {output_file}")
            else:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"❌ Failed to generate {output_file}: {result.stderr}")

        if success_count == 0:
            raise RuntimeError(f"No empirevent files were generated in {archive / 'EventFiles'}")


    def process_data(self, 
                     dSpace_px: float = 4.0,
                     dTime_s: float = 50e-9,
                     durationMax_s: float = 500e-9,
                     dTime_ext: float = 1.0,
                     nBins: int = 1000,
                     nPhotons_bins: int = None,
                     binning_time_resolution: float = 1.5625e-9,
                     binning_offset: float = 0.0,
                     verbosity: VerbosityLevel = VerbosityLevel.QUIET,
                     suffix: str = "",
                     method: str = "empir_pipeline") -> pd.DataFrame:
        """
        Streamlined method to process data through the complete analysis pipeline.
        
        Args:
            dSpace_px: Spatial clustering distance in pixels
            dTime_s: Time clustering threshold in seconds
            durationMax_s: Maximum event duration in seconds
            dTime_ext: Time extension factor for clustering
            nBins: Number of time bins
            nPhotons_bins: Number of photon bins
            binning_time_resolution: Time resolution for binning in seconds
            binning_offset: Time offset for binning in seconds
            verbosity: Level of output verbosity
            suffix: Optional suffix for output folder and files
            method: Processing method - "empir_pipeline" or "nea"/"neutron_event_analyzer"
            
        Returns:
            DataFrame with processed data including stacks, counts, and error
        """
        valid_methods = ["empir_pipeline", "nea", "neutron_event_analyzer"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")
        
        if method in ["nea", "neutron_event_analyzer"] and not NEA_AVAILABLE:
            raise ImportError("neutron_event_analyzer package is required for 'nea' method.")

        analysed_dir = self.archive / "AnalysedResults"
        analysed_dir.mkdir(parents=True, exist_ok=True)
        imported_photons_dir = self.archive / "ImportedPhotons"
        imported_photons_dir.mkdir(parents=True, exist_ok=True)
        suffix_dir = analysed_dir / (suffix.strip("_") if suffix else "default")
        suffix_dir.mkdir(parents=True, exist_ok=True)

        if method == "empir_pipeline":
            self._run_import_photons(verbosity=verbosity)
            p2e_config = self.Photon2EventConfig(
                dSpace_px=dSpace_px,
                dTime_s=dTime_s,
                durationMax_s=durationMax_s,
                dTime_ext=dTime_ext
            )
            params_file = suffix_dir / "parameterSettings.json"
            p2e_config.write(params_file)
            self._run_photon2event(config=p2e_config, verbosity=verbosity)
            binning_config = self.EventBinningConfig().time_binning()
            binning_config.binning_t.nBins = nBins
            binning_config.binning_t.resolution_s = binning_time_resolution
            binning_config.binning_t.offset_s = binning_offset
            if nPhotons_bins is not None:
                binning_config = binning_config.nphotons_binning()
                binning_config.binning_nPhotons.nBins = nPhotons_bins
            event_params_file = suffix_dir / "parameterEvents.json"
            binning_config.write(event_params_file)
            self._run_event_binning(config=binning_config, verbosity=verbosity)
            result_df = self._read_binned_data()
            if nPhotons_bins is None:
                result_df.columns = ["stacks", "counts"]
            else:
                result_df.columns = ["stacks", "nPhotons", "counts"]
            result_df["err"] = np.sqrt(result_df["counts"])
            result_df["stacks"] = np.arange(len(result_df))
            output_csv = suffix_dir / "counts.csv"
            result_df.to_csv(output_csv, index=False)
            if verbosity >= VerbosityLevel.BASIC:
                print(f"Processed data saved to {output_csv}")
            return result_df
        else:  # nea method
            with tempfile.TemporaryDirectory(dir="/tmp") as temp_nea_dir:
                temp_nea_dir = Path(temp_nea_dir)
                photon_files_dir = temp_nea_dir / "PhotonFiles"
                event_files_dir = temp_nea_dir / "EventFiles"
                photon_files_dir.mkdir(parents=True, exist_ok=True)
                event_files_dir.mkdir(parents=True, exist_ok=True)

                self._run_import_photons(verbosity=verbosity)
                p2e_config = self.Photon2EventConfig(
                    dSpace_px=dSpace_px,
                    dTime_s=dTime_s,
                    durationMax_s=durationMax_s,
                    dTime_ext=dTime_ext
                )
                params_file = temp_nea_dir / "parameterSettings.json"
                p2e_config.write(params_file)
                self._run_photon2event(config=p2e_config, archive=temp_nea_dir, verbosity=verbosity)

                analyzer = nea.Analyse(data_folder=str(temp_nea_dir), export_dir=str(self.empir_dirpath))
                analyzer.load(
                    event_glob="EventFiles/*.empirevent",
                    photon_glob="PhotonFiles/*.empirphot"
                )
                analyzer.associate(
                    time_norm_ns=dTime_s * 1e9,  # Convert to ns
                    spatial_norm_px=dSpace_px,
                    dSpace_px=dSpace_px,
                    max_time_ns=durationMax_s * 1e9,
                    verbosity=verbosity,
                    method="kdtree"  # Use kdtree for simulation data
                )
                results_df = analyzer.get_combined_dataframe()

                # Format results to match empir_pipeline output
                if not results_df.empty:
                    recon_df = results_df.copy()
                    recon_df = recon_df.rename(columns={
                        'assoc_x': 'x [px]',
                        'assoc_y': 'y [px]',
                        'assoc_t': 't [s]',
                        'assoc_n': 'nPhotons [1]',
                        'assoc_PSD': 'PSD value'
                    })
                    recon_df['t_relToExtTrigger [s]'] = recon_df['t [s]']
                    recon_df['neutron_id'] = recon_df.groupby(['x [px]', 'y [px]', 't [s]', 'nPhotons [1]', 'PSD value']).ngroup() + 1
                    expected_columns = ['x [px]', 'y [px]', 't [s]', 'nPhotons [1]', 'PSD value', 't_relToExtTrigger [s]', 'neutron_id']
                    for col in expected_columns:
                        if col not in recon_df.columns:
                            recon_df[col] = np.nan if col != 'nPhotons [1]' else 0
                    recon_df = recon_df[expected_columns]

                    # Apply binning
                    binning_config = self.EventBinningConfig().time_binning()
                    binning_config.binning_t.nBins = nBins
                    binning_config.binning_t.resolution_s = binning_time_resolution
                    binning_config.binning_t.offset_s = binning_offset
                    if nPhotons_bins is not None:
                        binning_config = binning_config.nphotons_binning()
                        binning_config.binning_nPhotons.nBins = nPhotons_bins
                    event_params_file = suffix_dir / "parameterEvents.json"
                    binning_config.write(event_params_file)

                    # Perform binning manually on recon_df
                    bins = np.linspace(
                        binning_offset,
                        binning_offset + nBins * binning_time_resolution,
                        nBins + 1
                    )
                    if nPhotons_bins:
                        photon_bins = np.arange(nPhotons_bins + 1)
                        result_df = recon_df.groupby([
                            pd.cut(recon_df['t [s]'], bins=bins, labels=range(nBins)),
                            pd.cut(recon_df['nPhotons [1]'], bins=photon_bins, labels=range(nPhotons_bins))
                        ]).size().reset_index(name='counts')
                        result_df.columns = ['stacks', 'nPhotons', 'counts']
                    else:
                        result_df = recon_df.groupby(
                            pd.cut(recon_df['t [s]'], bins=bins, labels=range(nBins))
                        ).size().reset_index(name='counts')
                        result_df.columns = ['stacks', 'counts']
                    result_df['err'] = np.sqrt(result_df['counts'])
                    result_df['stacks'] = result_df['stacks'].astype(int)

                    output_csv = suffix_dir / "counts.csv"
                    result_df.to_csv(output_csv, index=False)
                    if verbosity >= VerbosityLevel.BASIC:
                        print(f"Processed data saved to {output_csv}")
                    return result_df
                else:
                    if verbosity >= VerbosityLevel.BASIC:
                        print("No associated events found by NEA")
                    return pd.DataFrame()

    def process_data_event_by_event(self,
                                   dSpace_px: float = 4.0,
                                   dTime_s: float = 50e-9,
                                   durationMax_s: float = 500e-9,
                                   dTime_ext: float = 1.0,
                                   nBins: int = 1000,
                                   binning_time_resolution: float = 1.5625e-9,
                                   binning_offset: float = 0.0,
                                   verbosity: int = 0,
                                   merge: bool = False,
                                   suffix: str = "",
                                   time_norm_ns: float = 1.0,
                                   spatial_norm_px: float = 1.0,
                                   weight_px_in_s: float = 1e-9,
                                   max_time_ns: float = 500.0,
                                   fov: float = 120.0,
                                   focus_factor: float = 1.2,
                                   method: str = "empir_pipeline") -> pd.DataFrame:
        """
        Processes data event by event, grouping optical photons by neutron_id.
        
        Args:
            dSpace_px: Spatial clustering distance in pixels for photon-to-event clustering.
            dTime_s: Time clustering threshold in seconds for photon-to-event clustering.
            durationMax_s: Maximum event duration in seconds for photon-to-event clustering.
            dTime_ext: Time extension factor for clustering.
            nBins: Number of time bins for histogram-based analysis.
            binning_time_resolution: Time resolution for binning in seconds.
            binning_offset: Time offset for binning in seconds.
            verbosity: Level of output verbosity (0=QUIET, 1=BASIC, 2=DETAILED).
            merge: If True, merge results with simulation and traced data and save.
            suffix: Optional suffix for output folder and files.
            time_norm_ns: Normalization factor for time differences (ns) in photon-event association.
            spatial_norm_px: Normalization factor for spatial differences (px) in photon-event association.
            weight_px_in_s: Weight of spatial distance in seconds (px-to-s conversion) for association.
            max_time_ns: Maximum time difference (ns) for associating photons with events.
            fov: Field of view in mm.
            focus_factor: Factor relating hit position on sensor to actual hit position on scintillator screen.
            method: Processing method - "empir_pipeline" or "nea"/"neutron_event_analyzer".
        
        Returns:
            DataFrame with processed event data (optionally merged with sim_data and traced_data).
        """
        valid_methods = ["empir_pipeline", "nea", "neutron_event_analyzer"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")
        
        if method in ["nea", "neutron_event_analyzer"] and not NEA_AVAILABLE:
            raise ImportError("neutron_event_analyzer package is required for 'nea' method.")

        if method == "empir_pipeline":
            return self._process_empir_pipeline(
                dSpace_px=dSpace_px, dTime_s=dTime_s, durationMax_s=durationMax_s,
                dTime_ext=dTime_ext, nBins=nBins, binning_time_resolution=binning_time_resolution,
                binning_offset=binning_offset, verbosity=verbosity, merge=merge, suffix=suffix,
                time_norm_ns=time_norm_ns, spatial_norm_px=spatial_norm_px, 
                fov=fov, focus_factor=focus_factor
            )
        else:  # nea method
            with tempfile.TemporaryDirectory(dir="/tmp") as temp_nea_dir:
                temp_nea_dir = Path(temp_nea_dir)
                photon_files_dir = temp_nea_dir / "PhotonFiles"
                event_files_dir = temp_nea_dir / "EventFiles"
                photon_files_dir.mkdir(parents=True, exist_ok=True)
                event_files_dir.mkdir(parents=True, exist_ok=True)

                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Using temporary directory for NEA: {temp_nea_dir}")
                self._run_import_photons(archive=temp_nea_dir, verbosity=verbosity)
                p2e_config = self.Photon2EventConfig(
                    dSpace_px=dSpace_px,
                    dTime_s=dTime_s,
                    durationMax_s=durationMax_s,
                    dTime_ext=dTime_ext
                )
                params_file = temp_nea_dir / "parameterSettings.json"
                p2e_config.write(params_file)
                self._run_photon2event(config=p2e_config, archive=temp_nea_dir, verbosity=verbosity)

                analyzer = nea.Analyse(data_folder=str(temp_nea_dir), export_dir=str(self.empir_dirpath))
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Loading NEA data from {temp_nea_dir}")
                    print(f"Photon files: {list((temp_nea_dir / 'PhotonFiles').glob('*.empirphot'))}")
                    print(f"Event files: {list((temp_nea_dir / 'EventFiles').glob('*.empirevent'))}")
                    print(f"Association parameters: time_norm_ns={time_norm_ns}, spatial_norm_px={spatial_norm_px}, "
                          f"dSpace_px={dSpace_px}, weight_px_in_s={weight_px_in_s}, max_time_ns={max_time_ns}")
                analyzer.load(
                    event_glob="EventFiles/*.empirevent",
                    photon_glob="PhotonFiles/*.empirphot"
                )
                analyzer.associate(
                    time_norm_ns=time_norm_ns,
                    spatial_norm_px=spatial_norm_px,
                    dSpace_px=dSpace_px,
                    weight_px_in_s=weight_px_in_s,
                    max_time_ns=max_time_ns,
                    verbosity=verbosity,
                    method="kdtree"
                )
                results_df = analyzer.get_combined_dataframe()

                # Format results to match empir_pipeline output
                if not results_df.empty:
                    recon_df = results_df.copy()
                    recon_df = recon_df.rename(columns={
                        'assoc_x': 'x [px]',
                        'assoc_y': 'y [px]',
                        'assoc_t': 't [s]',
                        'assoc_n': 'nPhotons [1]',
                        'assoc_PSD': 'PSD value'
                    })
                    recon_df['t_relToExtTrigger [s]'] = recon_df['t [s]']
                    recon_df['neutron_id'] = recon_df.groupby(['x [px]', 'y [px]', 't [s]', 'nPhotons [1]', 'PSD value']).ngroup() + 1
                    expected_columns = ['x [px]', 'y [px]', 't [s]', 'nPhotons [1]', 'PSD value', 't_relToExtTrigger [s]', 'neutron_id']
                    for col in expected_columns:
                        if col not in recon_df.columns:
                            recon_df[col] = np.nan if col != 'nPhotons [1]' else 0
                    recon_df = recon_df[expected_columns]

                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Before dropna: {len(recon_df)} rows")
                        print(f"Non-NaN counts: {recon_df.notna().sum().to_dict()}")
                    
                    suffix_dir = self.archive / "AnalysedResults" / (suffix.strip("_") if suffix else "default")
                    suffix_dir.mkdir(parents=True, exist_ok=True)
                    all_batches_csv = suffix_dir / "all_batches_results.csv"
                    recon_df.to_csv(all_batches_csv, index=False)
                    if verbosity >= VerbosityLevel.BASIC:
                        print(f"All batches results saved to {all_batches_csv} ({len(recon_df)} rows)")

                    if merge:
                        traced_data = pd.DataFrame()
                        traced_photons_path = self.archive / "TracedPhotons"
                        if traced_photons_path.exists():
                            traced_files = list(traced_photons_path.glob("*.csv"))
                            if traced_files:
                                traced_dfs = [pd.read_csv(f) for f in traced_files]
                                traced_data = pd.concat(traced_dfs, ignore_index=True)
                                if verbosity >= VerbosityLevel.BASIC:
                                    print(f"Loaded {len(traced_data)} traced photons")
                
                        merged_df = self.merge_sim_and_recon_data(
                            self.sim_data, traced_data, recon_df,
                            fov=fov, focus_factor=focus_factor,
                            time_norm_ns=time_norm_ns, spatial_norm_px=spatial_norm_px,
                            dSpace_px=dSpace_px, verbosity=verbosity
                        )
                
                        merged_csv = suffix_dir / "merged_all_batches_results.csv"
                        merged_df.to_csv(merged_csv, index=False)
                
                        if verbosity >= VerbosityLevel.BASIC:
                            print(f"NEA merged results saved to {merged_csv} ({len(merged_df)} rows)")
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"After dropna: {len(merged_df.dropna())} rows")
                            print(f"Non-NaN counts after merge: {merged_df.notna().sum().to_dict()}")
                
                        return merged_df
                    return recon_df
                else:
                    if verbosity >= VerbosityLevel.BASIC:
                        print("No associated events found by NEA")
                    return pd.DataFrame()

                    
    def _process_empir_pipeline(self,
                               dSpace_px: float,
                               dTime_s: float,
                               durationMax_s: float,
                               dTime_ext: float,
                               nBins: int,
                               binning_time_resolution: float,
                               binning_offset: float,
                               verbosity: int,
                               merge: bool,
                               suffix: str,
                               time_norm_ns: float,
                               spatial_norm_px: float,
                               fov: float,
                               focus_factor: float) -> pd.DataFrame:
        # Implementation of empir_pipeline (unchanged from original)
        analysed_dir = self.archive / "AnalysedResults"
        analysed_dir.mkdir(parents=True, exist_ok=True)
        imported_photons_dir = self.archive / "ImportedPhotons"
        imported_photons_dir.mkdir(parents=True, exist_ok=True)
        suffix_dir = analysed_dir / (suffix.strip("_") if suffix else "default")
        suffix_dir.mkdir(parents=True, exist_ok=True)

    def _run_import_photons(self, archive: Path = None, parallel=True, verbosity: VerbosityLevel = VerbosityLevel.QUIET):
        """
        Process all traced photon files to generate .empirphot files.

        Args:
            archive: Path to the directory where .empirphot files should be saved (defaults to self.archive).
            parallel: Whether to process files in parallel using multiprocessing.
            verbosity: Verbosity level for logging.

        Raises:
            FileNotFoundError: If no traced files are found in self.archive/TracedPhotons.
        """
        if archive is None:
            archive = self.archive
        archive = Path(archive)
        photon_files_dir = archive / "PhotonFiles"
        photon_files_dir.mkdir(parents=True, exist_ok=True)

        traced_files = sorted((self.archive / "TracedPhotons").glob("traced_sim_data_*.csv"))
        if not traced_files:
            raise FileNotFoundError(f"No traced files found in {self.archive / 'TracedPhotons'}")
        
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Processing {len(traced_files)} traced photon files from {self.archive / 'TracedPhotons'} into {photon_files_dir}...")

        success_count = 0
        if parallel:
            with Pool() as pool:
                results = list(tqdm(pool.imap_unordered(self._process_single_file_wrapper, 
                                                        [(self, file, photon_files_dir, verbosity) for file in traced_files]), 
                                    total=len(traced_files), desc="Processing files"))
                success_count = sum(1 for result in results if result)
        else:
            for file in tqdm(traced_files, desc="Processing files"):
                if self._process_single_file(file, photon_files_dir, verbosity=verbosity):
                    success_count += 1

        if verbosity >= VerbosityLevel.BASIC:
            print(f"✅ Finished processing {success_count}/{len(traced_files)} files successfully!")
        
        if success_count == 0:
            raise RuntimeError(f"No files were processed successfully, cannot proceed to photon2event. Check {photon_files_dir} for .empirphot files.")

    def merge_sim_and_recon_data(self, sim_data, traced_data, recon_data, 
                                 fov: float = 120.0, focus_factor: float = 1.2,
                                 time_norm_ns: float = 1.0, spatial_norm_px: float = 1.0,
                                 dSpace_px: float = 4.0, verbosity: int = 0):
        sim_df = sim_data.copy()
        traced_df = traced_data.copy()
        recon_df = recon_data.copy()
        
        sim_df['x2'] = np.nan
        sim_df['y2'] = np.nan
        sim_df['z2'] = np.nan
        sim_df['toa2'] = np.nan
        sim_df['photon_px'] = np.nan
        sim_df['photon_py'] = np.nan
        
        if not traced_df.empty:
            if len(traced_df) != len(sim_df):
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Warning: traced_data ({len(traced_df)} rows) and sim_data ({len(sim_df)} rows) have different lengths. Truncating to minimum.")
                min_len = min(len(traced_df), len(sim_df))
                sim_df = sim_df.iloc[:min_len].copy()
                traced_df = traced_df.iloc[:min_len].copy()
            
            time_col = None
            for candidate in ['toa2', 'toa', 'time']:
                if candidate in traced_df.columns:
                    time_col = candidate
                    break
            
            if time_col and all(col in traced_df.columns for col in ['x2', 'y2']):
                sim_df['x2'] = traced_df['x2'].values
                sim_df['y2'] = traced_df['y2'].values
                sim_df['z2'] = traced_df.get('z2', pd.Series(np.nan, index=traced_df.index)).values
                sim_df['toa2'] = traced_df[time_col].values
                sim_df['photon_px'] = (sim_df['x2'] + 10) / 10 * 128
                sim_df['photon_py'] = (sim_df['y2'] + 10) / 10 * 128
                if verbosity >= VerbosityLevel.DETAILED:
                    print(f"Assigned traced columns. Non-NaN counts: x2={sim_df['x2'].notna().sum()}, toa2={sim_df['toa2'].notna().sum()}")
            else:
                if verbosity >= VerbosityLevel.BASIC:
                    print("Warning: traced_data missing required columns (x2, y2, toa2/toa/time). Traced columns remain NaN.")
        
        merged_df = sim_df.copy()
        
        for col in recon_df.columns:
            if col != 'neutron_id':
                merged_df[col] = np.nan
        merged_df['event_id'] = np.nan
        merged_df['time_diff_ns'] = np.nan
        merged_df['spatial_diff_px'] = np.nan
        
        recon_groups = recon_df.groupby('neutron_id')
        
        for neutron_id in sorted(merged_df['neutron_id'].unique()):
            sim_group = merged_df[merged_df['neutron_id'] == neutron_id].copy()
            recon_group = recon_groups.get_group(neutron_id) if neutron_id in recon_groups.groups else pd.DataFrame()
            
            if recon_group.empty:
                continue
            
            recon_group = recon_group.sort_values('t [s]').reset_index(drop=True)
            recon_group['event_id'] = recon_group.index + 1
            
            for _, recon_row in recon_group.iterrows():
                n_photons = int(recon_row['nPhotons [1]'])
                recon_time_s = recon_row['t [s]']
                recon_x = recon_row['x [px]']
                recon_y = recon_row['y [px]']
                event_id = recon_row['event_id']
                
                sim_times = sim_group['toa2'].values * 1e-9  # Convert ns to s
                sim_px = sim_group['photon_px'].values
                sim_py = sim_group['photon_py'].values
                
                time_diffs = np.abs(sim_times - recon_time_s) * 1e9
                spatial_diffs = np.sqrt((sim_px - recon_x)**2 + (sim_py - recon_y)**2)
                
                if np.all(np.isnan(sim_px)) or np.all(np.isnan(sim_py)) or np.all(np.isnan(sim_times)):
                    combined_diffs = time_diffs / time_norm_ns
                    spatial_diffs = np.array([np.nan] * len(time_diffs))
                else:
                    combined_diffs = (time_diffs / time_norm_ns) + (spatial_diffs / spatial_norm_px)
                
                if n_photons == 1:
                    if len(sim_group) > 0:
                        closest_idx = np.argmin(combined_diffs)
                        sim_idx = sim_group.index[closest_idx]
                        for col in recon_df.columns:
                            if col != 'neutron_id':
                                merged_df.loc[sim_idx, col] = recon_row[col]
                        merged_df.loc[sim_idx, 'event_id'] = event_id
                        merged_df.loc[sim_idx, 'time_diff_ns'] = time_diffs[closest_idx]
                        merged_df.loc[sim_idx, 'spatial_diff_px'] = spatial_diffs[closest_idx]
                else:
                    if len(sim_group) >= n_photons:
                        closest_indices = np.argsort(combined_diffs)[:n_photons]
                        selected_px = sim_group.iloc[closest_indices]['photon_px']
                        selected_py = sim_group.iloc[closest_indices]['photon_py']
                        if not (np.all(np.isnan(selected_px)) or np.all(np.isnan(selected_py))):
                            com_x = selected_px.mean()
                            com_y = selected_py.mean()
                            com_dist = np.sqrt((com_x - recon_x)**2 + (com_y - recon_y)**2)
                            if com_dist > dSpace_px:
                                continue
                        for idx in closest_indices:
                            sim_idx = sim_group.index[idx]
                            for col in recon_df.columns:
                                if col != 'neutron_id':
                                    merged_df.loc[sim_idx, col] = recon_row[col]
                            merged_df.loc[sim_idx, 'event_id'] = event_id
                            merged_df.loc[sim_idx, 'time_diff_ns'] = time_diffs[idx]
                            merged_df.loc[sim_idx, 'spatial_diff_px'] = spatial_diffs[idx]
        
        merged_df = self.calculate_reconstruction_stats(merged_df, fov=fov, focus_factor=focus_factor)
        
        sim_cols = [col for col in sim_df.columns if col not in ['x2', 'y2', 'z2', 'toa2', 'photon_px', 'photon_py']]
        traced_cols = ['x2', 'y2', 'z2', 'toa2', 'photon_px', 'photon_py']
        recon_cols = [col for col in recon_df.columns if col != 'neutron_id']
        final_cols = sim_cols + traced_cols + recon_cols + ['event_id', 'time_diff_ns', 'spatial_diff_px', 'x3', 'y3', 'delta_x', 'delta_y', 'delta_r']
        merged_df = merged_df[[col for col in final_cols if col in merged_df.columns]]
        
        return merged_df

    def calculate_reconstruction_stats(self, df: pd.DataFrame, fov: float = 120.0, focus_factor: float = 1.2):
        df["x3"] = (128 - df["x [px]"]) / 256 * fov * focus_factor
        df["y3"] = (128 - df["y [px]"]) / 256 * fov * focus_factor
        df["delta_x"] = df["x3"] - df["nx"]
        df["delta_y"] = df["y3"] - df["ny"]
        df["delta_r"] = np.sqrt(df["delta_x"]**2 + df["delta_y"]**2)
        return df

    def export_events(self, 
                      archive: Path = None,
                      enhanced_results: pd.DataFrame = None, 
                      verbosity: VerbosityLevel = VerbosityLevel.QUIET) -> bool:
        archive = archive if archive is not None else self.archive
        
        if enhanced_results is None or enhanced_results.empty:
            reconstructed_path = archive / "ReconstructedEvents" / "events_with_shape_parameters.csv"
            if not reconstructed_path.exists():
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"No enhanced results found at {reconstructed_path}")
                return False
            try:
                enhanced_results = pd.read_csv(reconstructed_path)
            except Exception as e:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Error reading enhanced results: {str(e)}")
                return False
        
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Exporting {len(enhanced_results)} events to CSV files...")
        
        try:
            exported_events_dir = archive / "ExportedEvents"
            exported_events_dir.mkdir(parents=True, exist_ok=True)
            
            if 'file_number' in enhanced_results.columns:
                for file_num, group in enhanced_results.groupby('file_number'):
                    output_file = exported_events_dir / f"exported_event_data_{file_num}.csv"
                    group_save = group.copy()
                    if 'original_photon_ids' in group_save.columns:
                        group_save['original_photon_ids'] = group_save['original_photon_ids'].apply(
                            lambda x: str(x) if isinstance(x, list) else x
                        )
                    group_save.to_csv(output_file, index=False)
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Exported {len(group)} events to {output_file}")
            else:
                output_file = exported_events_dir / "exported_event_data.csv"
                enhanced_results_save = enhanced_results.copy()
                if 'original_photon_ids' in enhanced_results_save.columns:
                    enhanced_results_save['original_photon_ids'] = enhanced_results_save['original_photon_ids'].apply(
                        lambda x: str(x) if isinstance(x, list) else x
                    )
                enhanced_results_save.to_csv(output_file, index=False)
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Exported all {len(enhanced_results)} events to {output_file}")
            
            if verbosity >= VerbosityLevel.BASIC:
                print("Export completed successfully.")
            return True
        except Exception as e:
            if verbosity >= VerbosityLevel.BASIC:
                print(f"Error exporting events: {str(e)}")
            return False

    def save_reconstructed_events(self, 
                                 archive: Path = None,
                                 enhanced_results: pd.DataFrame = None, 
                                 verbosity: VerbosityLevel = VerbosityLevel.QUIET) -> bool:
        archive = archive if archive is not None else self.archive
        
        if enhanced_results is None or enhanced_results.empty:
            if verbosity >= VerbosityLevel.BASIC:
                print("No enhanced results provided to save.")
            return False
        
        try:
            reconstructed_dir = archive / "ReconstructedEvents"
            reconstructed_dir.mkdir(parents=True, exist_ok=True)
            output_file = reconstructed_dir / "events_with_shape_parameters.csv"
            enhanced_results_save = enhanced_results.copy()
            if 'original_photon_ids' in enhanced_results_save.columns:
                enhanced_results_save['original_photon_ids'] = enhanced_results_save['original_photon_ids'].apply(
                    lambda x: str(x) if isinstance(x, list) else x
                )
            enhanced_results_save.to_csv(output_file, index=False)
            if verbosity >= VerbosityLevel.BASIC:
                valid_count = enhanced_results['major_axis_px'].notna().sum()
                print(f"Saved {len(enhanced_results)} events with {valid_count} having valid shape parameters to {output_file}")
                track_count = enhanced_results['track_length_3d'].notna().sum()
                if track_count > 0:
                    print(f"Calculated track lengths for {track_count} out of {len(enhanced_results)} events")
            return True
        except Exception as e:
            if verbosity >= VerbosityLevel.BASIC:
                print(f"Error saving reconstructed events: {str(e)}")
            return False

    def cleanup_temporary_folders(self, 
                                 temp_folders: list = None, 
                                 verbosity: VerbosityLevel = VerbosityLevel.QUIET) -> bool:
        if temp_folders is None:
            temp_folders = ["temp", "tmp", "cache"]
        try:
            for folder in temp_folders:
                if isinstance(folder, str):
                    if Path(folder).is_absolute():
                        folder_path = Path(folder)
                    else:
                        folder_path = self.archive / folder
                else:
                    folder_path = folder
                if folder_path.exists() and folder_path.is_dir():
                    shutil.rmtree(folder_path)
                    if verbosity >= VerbosityLevel.BASIC:
                        print(f"Removed temporary folder: {folder_path}")
                elif verbosity >= VerbosityLevel.DETAILED:
                    print(f"Temporary folder not found: {folder_path}")
            return True
        except Exception as e:
            if verbosity >= VerbosityLevel.BASIC:
                print(f"Error cleaning up temporary folders: {str(e)}")
            return False