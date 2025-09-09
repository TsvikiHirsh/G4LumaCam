import pandas as pd
import numpy as np
import os
import subprocess
from pathlib import Path
from tqdm.notebook import tqdm
from enum import IntEnum
from dataclasses import dataclass, field
import json
import os
from typing import Dict, Any, Optional, Union
from multiprocessing import Pool
import glob
import shutil
import subprocess


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
        # Structure the parameters as required
        parameters = {
            "photon2event": {
                "dSpace_px": self.dSpace_px,
                "dTime_s": self.dTime_s,
                "durationMax_s": self.durationMax_s,
                "dTime_ext": self.dTime_ext
            }
        }
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Write the JSON file
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
        """
        Add time-of-flight binning to the configuration.
        
        Returns:
            Updated EventBinningConfig with binning_t_relToExtTrigger defined
        """
        self.binning_t_relToExtTrigger = BinningConfig(
            resolution_s=1.5625e-9,
            nBins=640,
            offset_s=0
        )
        return self
    
    def psd_binning(self) -> 'EventBinningConfig':
        """
        Add pulse shape discrimination binning to the configuration.
        
        Returns:
            Updated EventBinningConfig with binning_psd defined
        """
        self.binning_psd = BinningConfig(
            resolution=1e-6,
            nBins=100,
            offset=0
        )
        return self
    
    def nphotons_binning(self) -> 'EventBinningConfig':
        """
        Add number of photons binning to the configuration.
        
        Returns:
            Updated EventBinningConfig with binning_nPhotons defined
        """
        self.binning_nPhotons = BinningConfig(
            resolution=1,
            nBins=10,
            offset=0
        )
        return self
    
    def time_binning(self) -> 'EventBinningConfig':
        """
        Add time binning to the configuration.
        
        Returns:
            Updated EventBinningConfig with binning_t defined
        """
        self.binning_t = BinningConfig(
            resolution_s=1.5625e-9,
            nBins=640,
            offset_s=0
        )
        return self
    
    def spatial_binning(self) -> 'EventBinningConfig':
        """
        Add spatial (x and y) binning to the configuration.
        
        Returns:
            Updated EventBinningConfig with binning_x and binning_y defined
        """
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
        """
        Write the event binning configuration to a JSON file.
        
        Args:
            output_file: The path to save the parameters file.
        
        Returns:
            The path to the created JSON file.
        """
        # Structure the parameters as required
        parameters = {"bin_events": {}}
        
        # Add only the configurations that are defined
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
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Write the JSON file
        with open(output_file, 'w') as f:
            json.dump(parameters, f, indent=4)
        
        return output_file
    
    def _get_config_dict(self, config: BinningConfig, use_s: bool = False, use_px: bool = False) -> Dict[str, Any]:
        """Convert a BinningConfig to a dictionary with appropriate keys."""
        if config is None:
            return {}
            
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
        """
        Analysis class for the LumaCam data.

        Inputs:
            archive: str - The directory containing traced photon data.
            data: pd.DataFrame, Optional - The data to be analysed directly.
            sim_data: pd.DataFrame, Optional - The simulation data to be analysed directly.
            empir_dirpath: str, Optional - The path to the empir directory. If None, will use
                           EMPIR_PATH from config if available, otherwise defaults to "./empir".
        
        The class will:
        1. Use the provided DataFrame if `data` is given.
        2. Otherwise, load all traced photon files in `archive/TracedPhotons/`.
        """
        self.archive = Path(archive)
        
        # Determine EMPIR directory path
        if empir_dirpath is not None:
            # Use explicitly provided path
            self.empir_dirpath = Path(empir_dirpath)
        else:
            # Try to load from config
            try:
                from G4LumaCam.config.paths import EMPIR_PATH
                self.empir_dirpath = Path(EMPIR_PATH)
            except ImportError:
                # Fall back to default
                self.empir_dirpath = Path("./empir")
                
        self.traced_dir = self.archive / "TracedPhotons"
        self.sim_dir = self.archive / "SimPhotons"
        self.photon_files_dir = self.archive / "PhotonFiles"
        self.photon_files_dir.mkdir(parents=True, exist_ok=True)

        self.Photon2EventConfig = Photon2EventConfig
        self.EventBinningConfig = EventBinningConfig

        # Handle data input
        if data is not None:
            self.data = data
        else:
            # Load all traced photon files
            if not self.traced_dir.exists():
                raise FileNotFoundError(f"{self.traced_dir} does not exist.")
            
            traced_files = sorted(self.traced_dir.glob("traced_sim_data_*.csv"))
            if not traced_files:
                raise FileNotFoundError(f"No traced simulation data found in {self.traced_dir}.")

            # print(f"Loading {len(traced_files)} traced photon files...")

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

        # Handle sim data input
        if sim_data is not None:
            self.sim_data = sim_data
        else:
            # Load all sim photon files
            if not self.sim_dir.exists():
                raise FileNotFoundError(f"{self.sim_dir} does not exist.")
            
            sim_files = sorted(self.sim_dir.glob("sim_data_*.csv"))
            if not sim_files:
                raise FileNotFoundError(f"No sim simulation data found in {self.sim_dir}.")

            # print(f"Loading {len(sim_files)} sim photon files...")

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


        # Validate empir directory and executables
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

    def _process_single_file(self, file, verbosity: VerbosityLevel = VerbosityLevel.QUIET):
        """
        Processes a single traced photon file and runs the empir_import_photons script.
        """
        try:
            df = pd.read_csv(file)
            if df.empty:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"⚠️ Skipping empty file: {file.name}")
                return
            
            df = df[["x2", "y2", "toa2"]].dropna()
            df["toa2"] *= 1e-9
            df["px"] = (df["x2"] + 10) / 10 * 128  # convert between px and mm
            df["py"] = (df["y2"] + 10) / 10 * 128  # convert between px and mm
            df = df[["px", "py", "toa2"]]
            df.columns = ["x [px]", "y [px]", "t [s]"]
            df["t_relToExtTrigger [s]"] = df["t [s]"]
            df = df.loc[(df["t [s]"] >= 0) & (df["t [s]"] < 1)]

            # Create ImportedPhotons directory if it doesn't exist
            imported_photons_dir = self.archive / "ImportedPhotons"
            imported_photons_dir.mkdir(exist_ok=True)

            # Save to ImportedPhotons directory
            output_csv = imported_photons_dir / f"imported_{file.stem}.csv"
            df.sort_values("t [s]").to_csv(output_csv, index=False)

            # Output empirphot file
            empir_file = self.photon_files_dir / f"{file.stem}.empirphot"
            os.system(f"{self.empir_import_photons} {output_csv} {empir_file} csv")
            if verbosity >= VerbosityLevel.BASIC:
                print(f"✔ Processed {file.name} → {empir_file.name}")

        except Exception as e:
            if verbosity >= VerbosityLevel.BASIC: 
                print(f"❌ Error processing {file.name}: {e}")

    def _run_import_photons(self, parallel=True,verbosity: VerbosityLevel = VerbosityLevel.QUIET):
        """
        Runs empir_import_photons for all traced photon files.
        """
        traced_files = sorted(self.traced_dir.glob("traced_sim_data_*.csv"))
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Processing {len(traced_files)} traced photon files...")

        if parallel:
            with Pool() as pool:
                list(tqdm(pool.imap_unordered(self._process_single_file, traced_files), 
                          total=len(traced_files), desc="Processing files"))
        else:
            for file in tqdm(traced_files, desc="Processing files"):
                self._process_single_file(file,verbosity=verbosity)
        if verbosity >= VerbosityLevel.BASIC:
            print("✅ Finished processing all files!")


    def _run_photon2event(self, archive: str = None, 
                                config: Photon2EventConfig = None,
                                verbosity: VerbosityLevel = VerbosityLevel.BASIC,
                                **config_kwargs):
        """
        Run the photon2event script with the given configuration, processing all .empirphot files.
        """
        if archive is None:
            archive = self.archive
        archive = Path(archive)
        
        # Ensure eventFiles directory exists
        (archive / "EventFiles").mkdir(parents=True, exist_ok=True)
        
        # Get all empirphot files
        input_folder = archive / "PhotonFiles"
        empirphot_files = list(input_folder.glob("*.empirphot"))
        
        if not empirphot_files:
            raise FileNotFoundError(f"No empirphot files found in {input_folder}")
        
        # Create config file
        params_file = archive / "parameterSettings.json"
        if config is None:
            config = Photon2EventConfig(**config_kwargs)
        config.write(params_file)
        
        # Process each empirphot file
        for empirphot_file in empirphot_files:
            output_file = archive / "EventFiles" / f"{empirphot_file.stem}.empirevent"
            process_command = (
                f"{self.empir_dirpath}/bin/empir_photon2event "
                f"-i '{empirphot_file}' "
                f"-o '{output_file}' "
                f"--paramsFile '{params_file}'"
            )
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Running command: {process_command}")
                subprocess.run(process_command, shell=True)
            else:
                with tqdm(total=100, desc=f"Processing {empirphot_file.name}", disable=(verbosity == VerbosityLevel.QUIET)) as pbar:
                    subprocess.run(process_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    for _ in range(10):
                        pbar.update(10)

    def _run_event_binning(self, archive: str = None, 
                                config: EventBinningConfig = None,
                                verbosity: VerbosityLevel = VerbosityLevel.QUIET,
                                **config_kwargs):
        """
        Run the binning script with the given configuration, processing all .empirevent files.
        """
        if archive is None:
            archive = self.archive
        archive = Path(archive)
        
        # Ensure eventFiles directory exists
        input_folder = archive / "EventFiles"
        if not input_folder.exists():
            raise FileNotFoundError(f"No eventFiles folder found at {input_folder}")
        
        # Create config file
        params_file = archive / "parameterEvents.json"
        if config is None:
            config = EventBinningConfig(**config_kwargs)
        config.write(params_file)
        
        # Run binning on all empirevent files
        output_file = archive / "binned.empirevent"
        process_command = (
            f"{self.empir_bin_events} "
            f"-I '{input_folder}' "
            f"-o '{output_file}' "
            f"--paramsFile '{params_file}' "
            f"--fileFormat csv"
        )
        if verbosity >= VerbosityLevel.DETAILED:
            print(f"Running command: {process_command}")
            subprocess.run(process_command, shell=True)
        else:
            with tqdm(total=100, desc="Running binning", disable=(verbosity == VerbosityLevel.QUIET)) as pbar:
                subprocess.run(process_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for _ in range(10):
                    pbar.update(10)

    def _read_binned_data(self, archive: str = None) -> pd.DataFrame:
        """
        Read the binned data from the output file.

        Inputs:
            archive: str, The name of the directory to save the data

        Returns:
            The binned data as a DataFrame
        """
        if archive is None:
            archive = self.archive
        archive = Path(archive)

        binned_data = pd.read_csv(archive/"binned.empirevent",header=0)
        return binned_data

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
                    suffix: str = "") -> pd.DataFrame:
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
            suffix: Optional suffix for output folder and files (e.g., "_test" creates "AnalysedResults/test")
            
        Returns:
            DataFrame with processed data including stacks, counts, and error
        """
        # Create base directories
        analysed_dir = self.archive / "AnalysedResults"
        analysed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create ImportedPhotons directory
        imported_photons_dir = self.archive / "ImportedPhotons"
        imported_photons_dir.mkdir(parents=True, exist_ok=True)
        
        # Create suffixed subfolder
        suffix_dir = analysed_dir / (suffix.strip("_") if suffix else "default")
        suffix_dir.mkdir(parents=True, exist_ok=True)
        
        # Run the import photons step
        self._run_import_photons(verbosity=verbosity)
        
        # Setup photon2event config
        p2e_config = self.Photon2EventConfig()
        p2e_config.dSpace_px = dSpace_px
        p2e_config.dTime_s = dTime_s
        p2e_config.durationMax_s = durationMax_s
        p2e_config.dTime_ext = dTime_ext
        params_file = suffix_dir / "parameterSettings.json"  # Save in suffixed folder
        p2e_config.write(params_file)
        
        # Run photon2event
        self._run_photon2event(config=p2e_config, verbosity=verbosity)
        
        # Setup event binning config
        binning_config = self.EventBinningConfig().time_binning()
        binning_config.binning_t.nBins = nBins
        binning_config.binning_t.resolution_s = binning_time_resolution
        binning_config.binning_t.offset_s = binning_offset
        if nPhotons_bins is not None:
            binning_config = binning_config.nphotons_binning()
            binning_config.binning_nPhotons.nBins = nPhotons_bins
        event_params_file = suffix_dir / "parameterEvents.json"  # Save in suffixed folder
        binning_config.write(event_params_file)
        
        # Run event binning
        self._run_event_binning(config=binning_config, verbosity=verbosity)
        
        # Read and process binned data
        result_df = self._read_binned_data()
        if nPhotons_bins is None:
            result_df.columns = ["stacks", "counts"]
        else:
            result_df.columns = ["stacks", "nPhotons", "counts"]
        result_df["err"] = np.sqrt(result_df["counts"])
        result_df["stacks"] = np.arange(len(result_df))
        
        # Save results in suffixed folder
        output_csv = suffix_dir / "counts.csv"
        result_df.to_csv(output_csv, index=False)
        
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Processed data saved to {output_csv}")
        
        return result_df

        
    def process_data_advanced(self,
                            photon2event_config: dict = None,
                            event_binning_config: dict = None,
                            binning_type: str = "time",
                            verbosity: VerbosityLevel = VerbosityLevel.QUIET) -> pd.DataFrame:
        """
        Advanced version of process_data allowing full configuration control.
        """
        self._run_import_photons()
        
        if photon2event_config is None:
            photon2event_config = {
                "dSpace_px": 4.0,
                "dTime_s": 50e-9,
                "durationMax_s": 500e-9,
                "dTime_ext": 1.0
            }
        p2e_config = self.Photon2EventConfig(**photon2event_config)
        self._run_photon2event(config=p2e_config, verbosity=verbosity)
        
        if event_binning_config is None:
            event_config = self.EventBinningConfig()
            binning_methods = {
                "time": event_config.time_binning,
                "tof": event_config.tof_binning,
                "psd": event_config.psd_binning,
                "nphotons": event_config.nphotons_binning,
                "spatial": event_config.spatial_binning
            }
            if binning_type not in binning_methods:
                raise ValueError(f"Invalid binning_type. Must be one of: {list(binning_methods.keys())}")
            event_config = binning_methods[binning_type]()
        else:
            event_config = self.EventBinningConfig(**event_binning_config)
        
        self._run_event_binning(config=event_config, verbosity=verbosity)
        
        result_df = self._read_binned_data()
        
        if binning_type in ["time", "tof"]:
            result_df.columns = ["stacks", "counts"]
            result_df["err"] = np.sqrt(result_df["counts"])
            result_df["stacks"] = np.arange(len(result_df))
            
            os.makedirs(self.archive, exist_ok=True)
            result_df.to_csv(f"{self.archive}/counts.csv", index=False)
        
        return result_df



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
                                    fov: float = 120.0,
                                    focus_factor: float = 1.2,
                                    method: str = "neutron_event_analyzer") -> pd.DataFrame:
        """
        Processes data event by event, grouping optical photons by neutron_id.
        
        This method supports two different processing approaches:
        1. "empir_pipeline" (default): Uses empir tools directly for photon2event conversion
        2. "nea" or "neutron_event_analyzer": Uses the neutron_event_analyzer package
        
        Args:
            dSpace_px: Spatial clustering distance in pixels
            dTime_s: Time clustering threshold in seconds
            durationMax_s: Maximum event duration in seconds
            dTime_ext: Time extension factor for clustering
            nBins: Number of time bins
            binning_time_resolution: Time resolution for binning in seconds
            binning_offset: Time offset for binning in seconds
            verbosity: Level of output verbosity
            merge: If True, merge results with simulation and traced data and save
            suffix: Optional suffix for output folder and files
            time_norm_ns: Normalization factor for time differences (ns) in matching
            spatial_norm_px: Normalization factor for spatial differences (px) in matching
            fov: Field of view in mm
            focus_factor: Factor that relates the hit position on the sensor to the actual hit position on the scintillator screen
            method: Processing method - "empir_pipeline" (default) or "nea"/"neutron_event_analyzer"
        
        Returns:
            DataFrame with processed event data (optionally merged with sim_data and traced_data)
        """
        
        # Validate method argument
        valid_methods = ["empir_pipeline", "nea", "neutron_event_analyzer"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")
        
        # Route to appropriate processing method
        if method == "empir_pipeline":
            return self._process_empir_pipeline(
                dSpace_px=dSpace_px, dTime_s=dTime_s, durationMax_s=durationMax_s,
                dTime_ext=dTime_ext, nBins=nBins, binning_time_resolution=binning_time_resolution,
                binning_offset=binning_offset, verbosity=verbosity, merge=merge, suffix=suffix,
                time_norm_ns=time_norm_ns, spatial_norm_px=spatial_norm_px, 
                fov=fov, focus_factor=focus_factor
            )
        else:  # nea or neutron_event_analyzer
            return self._process_nea_method(
                dSpace_px=dSpace_px, verbosity=verbosity, merge=merge, suffix=suffix,
                time_norm_ns=time_norm_ns, spatial_norm_px=spatial_norm_px,
                fov=fov, focus_factor=focus_factor
            )

    def _process_empir_pipeline(self,
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
                            fov: float = 120.0,
                            focus_factor: float = 1.2) -> pd.DataFrame:
        """
        Original empir pipeline implementation
        """
        # Create base directories
        analysed_dir = self.archive / "AnalysedResults"
        analysed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create ImportedPhotons directory
        imported_photons_dir = self.archive / "ImportedPhotons"
        imported_photons_dir.mkdir(parents=True, exist_ok=True)
        
        # Create suffixed subfolder
        suffix_dir = analysed_dir / (suffix.strip("_") if suffix else "default")
        suffix_dir.mkdir(parents=True, exist_ok=True)
        
        # Temporary directory for event processing
        temp_dir = self.archive / "temp_events"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Directory for batch results
        result_dir = self.archive / "EventResults"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine number of batches
        sim_photons_path = self.archive / "SimPhotons"
        traced_photons_path = self.archive / "TracedPhotons"  
        
        if sim_photons_path.exists():
            batch_files = list(sim_photons_path.glob("*.csv"))
            num_batches = len(batch_files)
            if verbosity >= 1:
                print(f"Detected {num_batches} batches from SimPhotons folder")
        elif traced_photons_path.exists():
            batch_files = list(traced_photons_path.glob("*.csv"))
            num_batches = len(batch_files)
            if verbosity >= 1:
                print(f"Detected {num_batches} batches from TracedPhotons folder")
        else:
            num_batches = 1
            if verbosity >= 1:
                print("No batch information found, defaulting to a single batch")
        
        # Combine data with simulation data
        if verbosity >= 1:
            print("Combining data with simulation data to include neutron_id...")
        
        if 'neutron_id' not in self.sim_data.columns:
            raise ValueError("Simulation data must contain a 'neutron_id' column")
        
        if len(self.data) != len(self.sim_data):
            raise ValueError(f"Data length mismatch: self.data has {len(self.data)} rows, self.sim_data has {len(self.sim_data)} rows")
        
        combined_data = self.data.copy()
        combined_data['neutron_id'] = self.sim_data['neutron_id']
        
        has_batch_id = 'batch_id' in self.sim_data.columns
        
        if has_batch_id:
            combined_data['batch_id'] = self.sim_data['batch_id']
            if verbosity >= 1:
                print("Using batch_id from simulation data")
        else:
            unique_neutron_ids = combined_data['neutron_id'].unique()
            events_per_batch = len(unique_neutron_ids) // num_batches + 1
            neutron_to_batch = {nid: min(i // events_per_batch, num_batches - 1) 
                                for i, nid in enumerate(unique_neutron_ids)}
            combined_data['batch_id'] = combined_data['neutron_id'].map(neutron_to_batch)
            if verbosity >= 1:
                print(f"Assigned {len(unique_neutron_ids)} neutron events to {num_batches} batches")
        
        # Group data by neutron_id
        if verbosity >= 1:
            print("Grouping data by neutron_id...")
        
        neutron_groups = combined_data.groupby('neutron_id')
        
        batch_results = {i: [] for i in range(num_batches)}
        
        # Setup photon2event config
        p2e_config = self.Photon2EventConfig()
        p2e_config.dSpace_px = dSpace_px
        p2e_config.dTime_s = dTime_s
        p2e_config.durationMax_s = durationMax_s
        p2e_config.dTime_ext = dTime_ext
        params_file = suffix_dir / "parameterSettings.json"
        p2e_config.write(params_file)
        
        # Setup event binning config
        binning_config = self.EventBinningConfig().time_binning()
        binning_config.binning_t.nBins = nBins
        binning_config.binning_t.resolution_s = binning_time_resolution
        binning_config.binning_t.offset_s = binning_offset
        event_params_file = suffix_dir / "parameterEvents.json"
        binning_config.write(event_params_file)
        
        # Process each neutron event
        for neutron_id, group in tqdm(neutron_groups, desc="Processing neutron events"):
            try:
                if group.empty:
                    continue
                    
                batch_id = group['batch_id'].iloc[0]
                
                df = group[["x2", "y2", "toa2"]].dropna()
                df["toa2"] *= 1e-9  # Convert toa2 from ns to s for processing
                df["px"] = (df["x2"] + 10) / 10 * 128
                df["py"] = (df["y2"] + 10) / 10 * 128
                df = df[["px", "py", "toa2"]]
                df.columns = ["x [px]", "y [px]", "t [s]"]
                df["t_relToExtTrigger [s]"] = df["t [s]"]
                df = df.loc[(df["t [s]"] >= 0) & (df["t [s]"] < 1)]
                
                if df.empty:
                    continue
                
                event_prefix = f"event_{neutron_id}"
                # Save to ImportedPhotons directory
                imported_csv = imported_photons_dir / f"imported_{event_prefix}.csv"
                df.sort_values("t [s]").to_csv(imported_csv, index=False)
                
                # Still use temp_dir for intermediate files
                temp_csv = temp_dir / f"{event_prefix}.csv"
                df.sort_values("t [s]").to_csv(temp_csv, index=False)
                
                empirphot_file = temp_dir / f"{event_prefix}.empirphot"
                cmd = f"{self.empir_import_photons} {temp_csv} {empirphot_file} csv"
                if verbosity >= 2:
                    print(f"Running: {cmd}")
                    subprocess.run(cmd, shell=True)
                else:
                    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                empirevent_file = temp_dir / f"{event_prefix}.empirevent"
                cmd = (
                    f"{self.empir_dirpath}/bin/empir_photon2event "
                    f"-i '{empirphot_file}' "
                    f"-o '{empirevent_file}' "
                    f"--paramsFile '{params_file}'"
                )
                if verbosity >= 2:
                    print(f"Running: {cmd}")
                    subprocess.run(cmd, shell=True)
                else:
                    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                event_result_csv = temp_dir / f"{event_prefix}_result.csv"
                cmd = (
                    f"{self.empir_dirpath}/empir_export_events "
                    f"{empirevent_file} "
                    f"{event_result_csv} "
                    f"csv"
                )
                if verbosity >= 2:
                    print(f"Running: {cmd}")
                    subprocess.run(cmd, shell=True)
                else:
                    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if event_result_csv.exists():
                    try:
                        result_df = pd.read_csv(event_result_csv)
                        result_df.columns = [col.strip() for col in result_df.columns]
                        expected_columns = ['x [px]', 'y [px]', 't [s]', 'nPhotons [1]', 'PSD value',
                                            't_relToExtTrigger [s]']
                        for col in expected_columns:
                            if col not in result_df.columns:
                                raise ValueError(f"Missing expected column '{col}' in result CSV")
                        result_df['neutron_id'] = neutron_id
                        batch_results[batch_id].append(result_df)
                    except pd.errors.EmptyDataError:
                        if verbosity >= 1:
                            print(f"Empty result for neutron_id {neutron_id}")
                
                # Clean up temporary files (but keep imported_csv)
                if verbosity < 2:
                    temp_csv.unlink(missing_ok=True)
                    empirphot_file.unlink(missing_ok=True)
                    empirevent_file.unlink(missing_ok=True)
                    event_result_csv.unlink(missing_ok=True)
                    
            except Exception as e:
                if verbosity >= 1:
                    print(f"Error processing neutron_id {neutron_id}: {str(e)}")
        
        # Combine and save results by batch
        all_results = []
        for batch_id, results in batch_results.items():
            if not results:
                if verbosity >= 1:
                    print(f"No valid results for batch {batch_id}")
                continue
                
            batch_df = pd.concat(results, ignore_index=True)
            all_results.append(batch_df)
            
            batch_csv = result_dir / f"batch_{batch_id}_results.csv"
            batch_df.to_csv(batch_csv, index=False)
            
            if verbosity >= 1:
                print(f"Batch {batch_id} results saved to {batch_csv} ({len(results)} neutron events)")
        
        if not all_results:
            if verbosity >= 1:
                print("No valid results found!")
            return pd.DataFrame()
        
        combined_results = pd.concat(all_results, ignore_index=True)
        
        # Save combined results in suffixed folder
        combined_csv = suffix_dir / "all_batches_results.csv"
        combined_results.to_csv(combined_csv, index=False)
        
        if verbosity >= 1:
            print(f"All results saved to {combined_csv}")
            print(f"Processed {len(combined_results)} neutron events across {len(batch_results)} batches")
        
        # Optional merging with simulation and traced data
        if merge:
            if verbosity >= 1:
                print("Merging processed results with simulation and traced data...")
            
            # Load traced data (unchanged)
            traced_data = pd.DataFrame()
            traced_photons_path = self.archive / "TracedPhotons"
            if traced_photons_path.exists():
                traced_files = list(traced_photons_path.glob("*.csv"))
                if traced_files:
                    traced_dfs = [pd.read_csv(f) for f in traced_files]
                    traced_data = pd.concat(traced_dfs, ignore_index=True)
            
            # Call shared merge method (replaces the nested def)
            merged_df = self.merge_sim_and_recon_data(
                self.sim_data, traced_data, combined_results, 
                fov=fov, focus_factor=focus_factor, time_norm_ns=time_norm_ns, 
                spatial_norm_px=spatial_norm_px, dSpace_px=dSpace_px, verbosity=verbosity
            )
            
            # Save merged results in suffixed folder
            merged_csv = suffix_dir / "merged_all_batches_results.csv"
            merged_df.to_csv(merged_csv, index=False)
            
            if verbosity >= 1:
                print(f"Merged results saved to {merged_csv}")
            
            return merged_df
        
        return combined_results

    def _process_nea_method(self,
                            dSpace_px: float = 4.0,
                            verbosity: int = 0,
                            merge: bool = False,
                            suffix: str = "",
                            time_norm_ns: float = 1.0,
                            spatial_norm_px: float = 1.0,
                            fov: float = 120.0,
                            focus_factor: float = 1.2) -> pd.DataFrame:
        """
        Neutron Event Analyzer (nea) implementation - aligned with empir_pipeline output.
        """
        try:
            import neutron_event_analyzer as nea
        except ImportError:
            raise ImportError(
                "neutron_event_analyzer package is required for this method. "
                "Install with: pip install neutron_event_analyzer[lumacam]"
            )
        
        # Create base directories
        analysed_dir = self.archive / "AnalysedResults"
        analysed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create suffixed subfolder
        suffix_dir = analysed_dir / (suffix.strip("_") if suffix else "default")
        suffix_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary directories for NEA processing (will be cleaned up)
        temp_nea_dir = self.archive / "temp_nea"
        temp_nea_dir.mkdir(parents=True, exist_ok=True)
        
        photon_files_dir = temp_nea_dir / "photonFiles"
        event_files_dir = temp_nea_dir / "eventFiles"
        export_dir = temp_nea_dir / "export"
        photon_files_dir.mkdir(parents=True, exist_ok=True)
        event_files_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if verbosity >= 1:
            print(f"Processing data using neutron_event_analyzer method in {temp_nea_dir}...")
        
        # Combine data with simulation data
        if 'neutron_id' not in self.sim_data.columns:
            raise ValueError("Simulation data must contain a 'neutron_id' column")
        
        if len(self.data) != len(self.sim_data):
            raise ValueError(f"Data length mismatch: self.data has {len(self.data)} rows, self.sim_data has {len(self.sim_data)} rows")
        
        combined_data = self.data.copy()
        combined_data['neutron_id'] = self.sim_data['neutron_id']
        
        # Process neutron events and create empir files
        neutron_groups = combined_data.groupby('neutron_id')
        
        empirphot_files = []
        empirevent_files = []
        
        if verbosity >= 1:
            print(f"Creating empir files for {len(neutron_groups)} neutron events...")
        
        for neutron_id, group in tqdm(neutron_groups, desc="Creating empir files"):
            try:
                if group.empty:
                    if verbosity >= 2:
                        print(f"Skipping empty group for neutron_id {neutron_id}")
                    continue
                
                # Prepare photon data
                df = group[["x2", "y2", "toa2"]].dropna()
                if df.empty:
                    if verbosity >= 2:
                        print(f"Skipping neutron_id {neutron_id}: no valid x2, y2, toa2 data")
                    continue
                    
                df = df.copy()
                df["toa2"] *= 1e-9  # Convert toa2 from ns to s
                df["px"] = (df["x2"] + 10) / 10 * 128
                df["py"] = (df["y2"] + 10) / 10 * 128
                df = df[["px", "py", "toa2"]]
                df.columns = ["x [px]", "y [px]", "t [s]"]
                df["t_relToExtTrigger [s]"] = df["t [s]"]
                df = df.loc[(df["t [s]"] >= 0) & (df["t [s]"] < 1)]
                
                if df.empty:
                    if verbosity >= 2:
                        print(f"Skipping neutron_id {neutron_id}: no data after time filtering")
                    continue
                
                # Save photon data and create empirphot file
                event_prefix = f"event_{neutron_id}"
                temp_csv = temp_nea_dir / f"{event_prefix}.csv"
                df.sort_values("t [s]").to_csv(temp_csv, index=False)
                
                empirphot_file = photon_files_dir / f"{event_prefix}.empirphot"
                cmd = f"{self.empir_import_photons} {temp_csv} {empirphot_file} csv"
                if verbosity >= 2:
                    print(f"Running: {cmd}")
                subprocess.run(cmd, shell=True, check=True)
                
                if empirphot_file.exists() and empirphot_file.stat().st_size > 0:
                    empirphot_files.append(str(empirphot_file))
                    if verbosity >= 2:
                        print(f"Created empirphot: {empirphot_file} ({empirphot_file.stat().st_size} bytes)")
                else:
                    if verbosity >= 2:
                        print(f"Failed to create valid empirphot: {empirphot_file}")
                
                # Create empirevent file
                empirevent_file = event_files_dir / f"{event_prefix}.empirevent"
                p2e_config = self.Photon2EventConfig(dSpace_px=dSpace_px)
                params_file = suffix_dir / "parameterSettings.json"
                p2e_config.write(params_file)
                
                cmd = (
                    f"{self.empir_dirpath}/bin/empir_photon2event "
                    f"-i '{empirphot_file}' "
                    f"-o '{empirevent_file}' "
                    f"--paramsFile '{params_file}'"
                )
                if verbosity >= 2:
                    print(f"Running: {cmd}")
                subprocess.run(cmd, shell=True, check=True)
                
                if empirevent_file.exists() and empirevent_file.stat().st_size > 0:
                    empirevent_files.append(str(empirevent_file))
                    if verbosity >= 2:
                        print(f"Created empirevent: {empirevent_file} ({empirevent_file.stat().st_size} bytes)")
                else:
                    if verbosity >= 2:
                        print(f"Failed to create valid empirevent: {empirevent_file}")
                
                # Clean up temporary CSV
                temp_csv.unlink(missing_ok=True)
                    
            except Exception as e:
                if verbosity >= 1:
                    print(f"Error processing neutron_id {neutron_id}: {str(e)}")
        
        if not empirphot_files or not empirevent_files:
            if verbosity >= 1:
                print(f"No valid empir files created! Photons: {len(empirphot_files)}, Events: {len(empirevent_files)}")
            return pd.DataFrame()
        
        if verbosity >= 1:
            print(f"Created {len(empirphot_files)} photon files and {len(empirevent_files)} event files")
        
        # Validate file pairing
        if verbosity >= 2:
            photon_bases = {Path(f).stem for f in empirphot_files}
            event_bases = {Path(f).stem for f in empirevent_files}
            paired_bases = photon_bases & event_bases
            print(f"Photon file bases (sample): {list(photon_bases)[:5]}...")
            print(f"Event file bases (sample): {list(event_bases)[:5]}...")
            print(f"Paired bases: {len(paired_bases)} (sample: {list(paired_bases)[:5]}...)")
        
        # Initialize NEA
        if verbosity >= 1:
            print(f"Initializing neutron_event_analyzer...")
        
        export_dir_str = str(export_dir)
        analyzer = nea.Analyse(data_folder=str(temp_nea_dir), export_dir=export_dir_str)
        
        # Create glob patterns
        photon_pattern = str(photon_files_dir / "*.empirphot")
        event_pattern = str(event_files_dir / "*.empirevent")
        
        if verbosity >= 2:
            print(f"Photon glob pattern: {photon_pattern}")
            print(f"Event glob pattern: {event_pattern}")
            print(f"Photon files found: {len(glob.glob(photon_pattern))}")
            print(f"Event files found: {len(glob.glob(event_pattern))}")
        
        # Load the data
        try:
            analyzer.load(event_glob=event_pattern, photon_glob=photon_pattern)
            if verbosity >= 2:
                print(f"Loaded data: Events shape {analyzer.events_df.shape if hasattr(analyzer, 'events_df') else 'N/A'}, "
                    f"Photons shape {analyzer.photons_df.shape if hasattr(analyzer, 'photons_df') else 'N/A'}")
        except Exception as e:
            if verbosity >= 1:
                print(f"Error loading data in NEA: {str(e)}")
            return pd.DataFrame()
        
        # Run association
        if verbosity >= 1:
            print("Running association...")
        try:
            analyzer.associate(
                time_norm_ns=time_norm_ns,
                spatial_norm_px=spatial_norm_px,
                dSpace_px=dSpace_px,
                verbosity=verbosity
            )
        except Exception as e:
            if verbosity >= 1:
                print(f"Error in NEA association: {str(e)}")
            return pd.DataFrame()
        
        # Get results
        if hasattr(analyzer, 'associated_df') and not analyzer.associated_df.empty:
            results_df = analyzer.associated_df.dropna().copy()
            
            if verbosity >= 1:
                print(f"NEA association completed with {len(results_df)} associated events")
            
            if verbosity >= 2:
                print(f"Associated_df columns: {list(results_df.columns)}")
                print(f"Sample associated_df:\n{results_df.head().to_string()}")
            
            # Map neutron_id
            if 'file_number' in results_df.columns:
                results_df['neutron_id'] = results_df['file_number'].apply(
                    lambda x: int(str(x).split('_')[-1]) if isinstance(x, str) and 'event_' in str(x) else x
                )
                if verbosity >= 2:
                    print(f"Extracted neutron_ids (sample): {results_df['neutron_id'].unique()[:5]}...")
            else:
                if verbosity >= 1:
                    print("Warning: No file_number column; assigning sequential IDs")
                results_df['neutron_id'] = np.arange(len(results_df))
            
            # Prepare recon_df to match empir format
            nea_column_mapping = {
                'event_x_px': 'x [px]',
                'event_y_px': 'y [px]',
                'event_t_s': 't [s]',
                'n_photons': 'nPhotons [1]',
                # Add mappings for other columns if present in NEA output, e.g.:
                # 'event_psd': 'PSD value',
                # 'event_t_rel_s': 't_relToExtTrigger [s]'
            }
            
            if verbosity >= 2:
                print(f"Applying NEA column mapping: {nea_column_mapping}")
            
            recon_df = results_df.copy()
            for nea_col, expected_col in nea_column_mapping.items():
                if nea_col in recon_df.columns:
                    recon_df = recon_df.rename(columns={nea_col: expected_col})
                    if verbosity >= 2:
                        print(f"Renamed {nea_col} to {expected_col}")
            
            # Add missing columns to match empir_pipeline output
            expected_columns = ['x [px]', 'y [px]', 't [s]', 'nPhotons [1]', 'PSD value', 't_relToExtTrigger [s]', 'neutron_id']
            for col in expected_columns:
                if col not in recon_df.columns:
                    if col == 'PSD value':
                        recon_df[col] = 0.0
                        if verbosity >= 2:
                            print(f"Added 'PSD value' column with default 0.0")
                    elif col == 't_relToExtTrigger [s]':
                        recon_df[col] = np.nan
                        if verbosity >= 2:
                            print(f"Added 't_relToExtTrigger [s]' column with default NaN")
                    elif col != 'neutron_id':
                        recon_df[col] = np.nan
                        if verbosity >= 2:
                            print(f"Added missing column {col} with default NaN")
            
            # Ensure columns are in correct order
            recon_df = recon_df[expected_columns]
            
            if verbosity >= 2:
                print(f"Final recon_df columns: {list(recon_df.columns)}")
                print(f"Sample recon_df:\n{recon_df.head().to_string()}")
            
            # Save recon events
            all_batches_csv = suffix_dir / "all_batches_results.csv"
            recon_df.to_csv(all_batches_csv, index=False)
            
            if verbosity >= 1:
                print(f"All batches results saved to {all_batches_csv} ({len(recon_df)} rows)")
            
            # Optional merging
            if merge:
                if verbosity >= 1:
                    print("Merging NEA results with simulation and traced data...")
                
                traced_data = pd.DataFrame()
                traced_photons_path = self.archive / "TracedPhotons"
                if traced_photons_path.exists():
                    traced_files = list(traced_photons_path.glob("*.csv"))
                    if traced_files:
                        traced_dfs = [pd.read_csv(f) for f in traced_files]
                        traced_data = pd.concat(traced_dfs, ignore_index=True)
                        if verbosity >= 1:
                            print(f"Loaded {len(traced_data)} traced photons")
                
                merged_df = self.merge_sim_and_recon_data(
                    self.sim_data, traced_data, recon_df,
                    fov=fov, focus_factor=focus_factor,
                    time_norm_ns=time_norm_ns, spatial_norm_px=spatial_norm_px,
                    dSpace_px=dSpace_px, verbosity=verbosity
                )
                
                merged_csv = suffix_dir / "merged_all_batches_results.csv"
                merged_df.to_csv(merged_csv, index=False)
                
                if verbosity >= 1:
                    print(f"NEA merged results saved to {merged_csv} ({len(merged_df)} rows)")
                
                return merged_df
            
            return recon_df
        else:
            if verbosity >= 1:
                print("No associated events found by NEA")
            return pd.DataFrame()
            
        # finally:
        # Always clean up temp files
        if temp_nea_dir.exists():
            shutil.rmtree(temp_nea_dir, ignore_errors=True)
            if verbosity >= 2:
                print(f"Cleaned up temporary directory: {temp_nea_dir}")


    def merge_sim_and_recon_data(self, sim_data, traced_data, recon_data, 
                                fov: float = 120.0, focus_factor: float = 1.2,
                                time_norm_ns: float = 1.0, spatial_norm_px: float = 1.0,
                                dSpace_px: float = 4.0, verbosity: int = 0):
        """
        Merge simulation, traced photon, and reconstruction dataframes based on neutron_id
        and row-by-row correspondence between sim_data and traced_data.
        Matches reconstructed events to simulation photons by minimizing a combined
        time and spatial distance metric.
        Adds event_id for each reconstructed event per neutron_id.
        
        Args:
            sim_data: DataFrame with simulation data
            traced_data: DataFrame with traced photon data (row-aligned with sim_data)
            recon_data: DataFrame with reconstructed event data
            fov (float, optional): Field of View in mm, used to scale photon positions
            focus_factor (float, optional): determines the ratio of position recorded on the sensor to the actual hit position on the scintillator screen 
            time_norm_ns: Normalization factor for time differences (ns) in matching
            spatial_norm_px: Normalization factor for spatial differences (px) in matching
            dSpace_px: Spatial clustering distance for center-of-mass check
            verbosity: Verbosity level for logging
        
        Returns:
            Merged DataFrame with simulation, traced, and reconstruction columns
        """
        sim_df = sim_data.copy()
        traced_df = traced_data.copy()
        recon_df = recon_data.copy()
        
        # Initialize traced columns in sim_df
        sim_df['x2'] = np.nan
        sim_df['y2'] = np.nan
        sim_df['z2'] = np.nan
        sim_df['toa2'] = np.nan
        sim_df['photon_px'] = np.nan
        sim_df['photon_py'] = np.nan
        
        # Assign traced data columns (row-by-row correspondence)
        if not traced_df.empty:
            if len(traced_df) != len(sim_df):
                if verbosity >= 1:
                    print(f"Warning: traced_data ({len(traced_df)} rows) and sim_data ({len(sim_df)} rows) have different lengths. Truncating to minimum.")
                min_len = min(len(traced_df), len(sim_df))
                sim_df = sim_df.iloc[:min_len].copy()
                traced_df = traced_df.iloc[:min_len].copy()
            
            # Identify time column
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
                if verbosity >= 2:
                    print(f"Assigned traced columns. Non-NaN counts: x2={sim_df['x2'].notna().sum()}, toa2={sim_df['toa2'].notna().sum()}")
            else:
                if verbosity >= 1:
                    print("Warning: traced_data missing required columns (x2, y2, toa2/toa/time). Traced columns remain NaN.")
        
        # Initialize merged DataFrame with sim_df
        merged_df = sim_df.copy()
        
        # Add reconstruction columns with NaN
        for col in recon_df.columns:
            if col != 'neutron_id':
                merged_df[col] = np.nan
        merged_df['event_id'] = np.nan
        merged_df['time_diff_ns'] = np.nan
        merged_df['spatial_diff_px'] = np.nan
        
        # Group reconstruction data by neutron_id
        recon_groups = recon_df.groupby('neutron_id')
        
        # Match reconstruction events to simulation rows
        for neutron_id in sorted(merged_df['neutron_id'].unique()):
            sim_group = merged_df[merged_df['neutron_id'] == neutron_id].copy()
            recon_group = recon_groups.get_group(neutron_id) if neutron_id in recon_groups.groups else pd.DataFrame()
            
            if recon_group.empty:
                continue
            
            # Assign event_id to reconstruction events
            recon_group = recon_group.sort_values('t [s]').reset_index(drop=True)
            recon_group['event_id'] = recon_group.index + 1
            
            # For each reconstruction event, find the closest simulation photon(s)
            for _, recon_row in recon_group.iterrows():
                n_photons = int(recon_row['nPhotons [1]'])
                recon_time_s = recon_row['t [s]']
                recon_x = recon_row['x [px]']
                recon_y = recon_row['y [px]']
                event_id = recon_row['event_id']
                
                # Compute distances
                sim_times = sim_group['toa2'].values  # in seconds? Wait, toa2 is ns in your data, but code assumes s? Fix: convert to s
                sim_times_ns = sim_times * 1e-9 if 'ns' in str(sim_times.dtype) else sim_times  # Adjust if needed; from head, toa2 is ns
                sim_px = sim_group['photon_px'].values
                sim_py = sim_group['photon_py'].values
                
                time_diffs = np.abs(sim_times_ns - recon_time_s) * 1e9  # Diff in s to ns
                spatial_diffs = np.sqrt((sim_px - recon_x)**2 + (sim_py - recon_y)**2)
                
                if np.all(np.isnan(sim_px)) or np.all(np.isnan(sim_py)) or np.all(np.isnan(sim_times)):
                    combined_diffs = time_diffs / time_norm_ns
                    spatial_diffs = np.array([np.nan] * len(time_diffs))
                else:
                    combined_diffs = (time_diffs / time_norm_ns) + (spatial_diffs / spatial_norm_px)
                
                # Select closest photon(s)
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
                                continue  # Skip if center of mass is too far
                        for idx in closest_indices:
                            sim_idx = sim_group.index[idx]
                            for col in recon_df.columns:
                                if col != 'neutron_id':
                                    merged_df.loc[sim_idx, col] = recon_row[col]
                            merged_df.loc[sim_idx, 'event_id'] = event_id
                            merged_df.loc[sim_idx, 'time_diff_ns'] = time_diffs[idx]
                            merged_df.loc[sim_idx, 'spatial_diff_px'] = spatial_diffs[idx]
        
        merged_df = self.calculate_reconstruction_stats(merged_df, fov=fov, focus_factor=focus_factor)
        
        # Ensure column order (matching your head example)
        sim_cols = [col for col in sim_df.columns if col not in ['x2', 'y2', 'z2', 'toa2', 'photon_px', 'photon_py']]
        traced_cols = ['x2', 'y2', 'z2', 'toa2', 'photon_px', 'photon_py']
        recon_cols = [col for col in recon_df.columns if col != 'neutron_id']
        final_cols = sim_cols + traced_cols + recon_cols + ['event_id', 'time_diff_ns', 'spatial_diff_px', 'x3', 'y3', 'delta_x', 'delta_y', 'delta_r']
        merged_df = merged_df[[col for col in final_cols if col in merged_df.columns]]
        
        return merged_df
                
    def calculate_reconstruction_stats(self,df: pd.DataFrame, fov:float=120, focus_factor:float = 1.2):
        """
        Calculates stats on reconstructed events, Adds columms to the analysis dataframe.

        Input:
            - df (pd.DataFrame): merged_df that contains all the reconstructed event-by-event columns
            - fov (float): Field of view in mm, used to scale pixel coordinates
            - focus_factor (float): A fcator that translates the position recorded on the sensor to the original position on the scintillator

        Returns:
            - df (pd.DataFrame): table with new stats columns
        """
                # Compute additional columns
        df["x3"] = (128 - df["x [px]"]) / 256 * fov *focus_factor
        df["y3"] = (128 - df["y [px]"]) / 256 * fov *focus_factor
        df["delta_x"] = df["x3"] - df["nx"]
        df["delta_y"] = df["y3"] - df["ny"]
        df["delta_r"] = np.sqrt(df["delta_x"]**2 + df["delta_y"]**2)

        return df


    def export_events(self, 
                    archive: Path=None,
                    enhanced_results: pd.DataFrame=None, 
                    verbosity: VerbosityLevel = VerbosityLevel.QUIET) -> bool:
        """
        Exports the events with their shape parameters to CSV files organized by their event file numbers.
        
        Args:
            enhanced_results: DataFrame containing the enhanced events data with shape parameters
            archive: Optional path to an alternative archive directory
            verbosity: Level of output verbosity
            
        Returns:
            bool: True if export was successful, False otherwise
        """

        
        # Use supplied archive or default to self.archive
        archive = archive if archive is not None else self.archive
        
        if enhanced_results is None or enhanced_results.empty:
            # Try to load from the default location if not provided
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
            # Create directories if they don't exist
            exported_events_dir = archive / "ExportedEvents"
            exported_events_dir.mkdir(parents=True, exist_ok=True)
            
            # Group events by file_number if available, otherwise create a single file
            if 'file_number' in enhanced_results.columns:
                # Group by file_number
                for file_num, group in enhanced_results.groupby('file_number'):
                    output_file = exported_events_dir / f"exported_event_data_{file_num}.csv"
                    
                    # Convert lists to strings for saving to CSV
                    group_save = group.copy()
                    if 'original_photon_ids' in group_save.columns:
                        group_save['original_photon_ids'] = group_save['original_photon_ids'].apply(
                            lambda x: str(x) if isinstance(x, list) else x
                        )
                    
                    group_save.to_csv(output_file, index=False)
                    
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Exported {len(group)} events to {output_file}")
                        
            else:
                # Single file export if no file_number available
                output_file = exported_events_dir / "exported_event_data.csv"
                
                # Convert lists to strings for saving to CSV
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
                                archive: Path=None,
                                enhanced_results: pd.DataFrame=None, 
                                verbosity: VerbosityLevel = VerbosityLevel.QUIET) -> bool:
        """
        Saves the reconstructed events with shape parameters to a dedicated directory.
        
        Args:
            enhanced_results: DataFrame containing the enhanced events with shape parameters
            archive: Optional path to an alternative archive directory
            verbosity: Level of output verbosity
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        
        # Use supplied archive or default to self.archive
        archive = archive if archive is not None else self.archive
        
        if enhanced_results is None or enhanced_results.empty:
            if verbosity >= VerbosityLevel.BASIC:
                print("No enhanced results provided to save.")
            return False
        
        try:
            # Create directory if it doesn't exist
            reconstructed_dir = archive / "ReconstructedEvents"
            reconstructed_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to the dedicated directory
            output_file = reconstructed_dir / "events_with_shape_parameters.csv"
            
            # Convert lists to strings for saving to CSV
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
                                temp_folders: list=None, 
                                verbosity: VerbosityLevel = VerbosityLevel.QUIET) -> bool:
        """
        Removes temporary folders used during processing.
        
        Args:
            temp_folders: List of folder paths to remove, relative to self.archive or absolute
            verbosity: Level of output verbosity
            
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        import shutil
        from pathlib import Path
        
        if temp_folders is None:
            # Default temporary folders to clean up
            temp_folders = ["temp", "tmp", "cache"]
        
        try:
            for folder in temp_folders:
                # Convert to Path if it's a string
                if isinstance(folder, str):
                    if Path(folder).is_absolute():
                        folder_path = Path(folder)
                    else:
                        folder_path = self.archive / folder
                else:
                    folder_path = folder
                
                # Check if folder exists
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

    def process_and_export_events(self,
                                archive: Path=None,    
                                results_df: pd.DataFrame=None,
                                verbosity: VerbosityLevel = VerbosityLevel.QUIET) -> pd.DataFrame:
        """
        Complete workflow to calculate event shape parameters, save reconstructed events,
        export to CSV files by event file number, and clean up temporary folders.
        
        Args:
            results_df: DataFrame containing the reconstructed events
            archive: Optional path to an alternative archive directory
            verbosity: Level of output verbosity
            
        Returns:
            DataFrame with the enhanced results (with shape parameters)
        """
        # Use supplied archive or default to self.archive
        archive = archive if archive is not None else self.archive
        
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Starting complete event processing workflow using archive: {archive}")
        
        # Step 1: Calculate ellipsoid shape parameters
        enhanced_results = self.calculate_event_ellipsoid_shape(
            results_df=results_df,
            verbosity=verbosity
        )
        
        if enhanced_results.empty:
            if verbosity >= VerbosityLevel.BASIC:
                print("No events were processed. Exiting workflow.")
            return enhanced_results
        
        # Step 2: Save to ReconstructedEvents directory
        saved = self.save_reconstructed_events(
            enhanced_results=enhanced_results,
            archive=archive,
            verbosity=verbosity
        )
        
        if not saved and verbosity >= VerbosityLevel.BASIC:
            print("Warning: Failed to save reconstructed events.")
        
        # Step 3: Export events by file number
        exported = self.export_events(
            enhanced_results=enhanced_results,
            archive=archive,
            verbosity=verbosity
        )
        
        if not exported and verbosity >= VerbosityLevel.BASIC:
            print("Warning: Failed to export events to CSV files.")
        
        # Step 4: Clean up temporary folders if requested
        try:
            cleaned = self.cleanup_temporary_folders(
                temp_folders=temp_folders,
                verbosity=verbosity
            )
        except:
            if verbosity >= VerbosityLevel.BASIC:
                print("Warning: Failed to clean up some temporary folders.")
        
        if verbosity >= VerbosityLevel.BASIC:
            print("Event processing workflow completed.")
        
        return enhanced_results