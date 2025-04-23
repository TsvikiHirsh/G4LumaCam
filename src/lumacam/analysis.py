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
            df["toa2"]*=1e-9
            df["px"] = (df["x2"] + 10)/10*128 # convert between px and mm
            df["py"] = (df["y2"] + 10)/10*128 # convert between px and mm
            df = df[["px", "py", "toa2"]]
            df.columns = ["x [px]", "y [px]", "t [s]"]
            df["t_relToExtTrigger [s]"] = df["t [s]"]
            df = df.loc[(df["t [s]"] >= 0) & (df["t [s]"] < 1)]

            # Sort and save
            temp_csv = self.archive / f"{file.stem}.csv"
            df.sort_values("t [s]").to_csv(temp_csv, index=False)

            # Output empirphot file
            empir_file = self.photon_files_dir / f"{file.stem}.empirphot"
            os.system(f"{self.empir_import_photons} {temp_csv} {empir_file} csv")
            if verbosity >= VerbosityLevel.BASIC:
                print(f"✔ Processed {file.name} → {empir_file.name}")
            temp_csv.unlink()  # Remove temporary CSV file

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
        # Create base directory for analysed results
        analysed_dir = self.archive / "AnalysedResults"
        analysed_dir.mkdir(parents=True, exist_ok=True)
        
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
            result_df.columns = ["stacks", "nPhotons","counts"]
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
                                suffix: str = "") -> pd.DataFrame:
        """
        Processes data event by event, grouping optical photons by neutron_id.
        
        This method:
        1. Combines self.data with neutron_id from self.sim_data
        2. Determines the number of batches from SimPhotons or TracedPhotons folder
        3. Groups traced photons by neutron_id and organizes them by batch
        4. Processes each neutron event independently
        5. Combines results by batch into separate files in a suffixed subfolder
        6. Optionally merges results with simulation and traced data, adding event_id
        
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
        
        Returns:
            DataFrame with processed event data (optionally merged with sim_data and traced_data)
        """
        
        # Create base directory for analysed results
        analysed_dir = self.archive / "AnalysedResults"
        analysed_dir.mkdir(parents=True, exist_ok=True)
        
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
        sim_photons_path = self.archive.parent / "SimPhotons"
        traced_photons_path = self.archive.parent / "TracedPhotons"
        
        if sim_photons_path.exists():
            batch_files = list(sim_photons_path.glob("*.simphot"))
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
        p2e_config.durationMax_s = durationMax_s  # Fixed: Removed invalid | operator
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
                df["toa2"] *= 1e-9
                df["px"] = (df["x2"] + 10) / 10 * 128
                df["py"] = (df["y2"] + 10) / 10 * 128
                df = df[["px", "py", "toa2"]]
                df.columns = ["x [px]", "y [px]", "t [s]"]
                df["t_relToExtTrigger [s]"] = df["t [s]"]
                df = df.loc[(df["t [s]"] >= 0) & (df["t [s]"] < 1)]
                
                if df.empty:
                    continue
                    
                event_prefix = f"event_{neutron_id}"
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
            
            def merge_sim_and_recon_data(sim_data, traced_data, recon_data):
                """
                Merge simulation, traced photon, and reconstruction dataframes based on neutron_id,
                spatial proximity (photon_px, photon_py vs x [px], y [px]), and temporal proximity (toa2 vs t [s]).
                The reconstructed event position is the center of mass of photon hit positions,
                and toa is the minimum of the photons in the group.
                Adds event_id for each reconstructed event per neutron_id.
                """
                sim_df = sim_data.copy()
                traced_df = traced_data.copy()
                recon_df = recon_data.copy()
                
                # Check for required columns in traced_df
                time_col = None
                for candidate in ['toa2', 'toa', 'time']:
                    if candidate in traced_df.columns:
                        time_col = candidate
                        break
                if time_col is None or not all(col in traced_df.columns for col in ['x2', 'y2']):
                    if verbosity >= 1:
                        print("Warning: traced_data missing required columns (x2, y2, toa2/toa/time). Skipping merge with traced data.")
                    sim_df['x2'] = np.nan
                    sim_df['y2'] = np.nan
                    sim_df['z2'] = np.nan
                    sim_df['toa2'] = np.nan
                    sim_df['photon_px'] = np.nan
                    sim_df['photon_py'] = np.nan
                else:
                    # Rename time column to toa2 for consistency
                    if time_col != 'toa2':
                        traced_df = traced_df.rename(columns={time_col: 'toa2'})
                    # Convert traced_df toa2 to nanoseconds if in seconds (assuming sim_df toa is in ns)
                    if traced_df['toa2'].max() < 1e3:  # Heuristic: if max toa2 < 1ms, assume seconds
                        traced_df['toa2'] = traced_df['toa2'] * 1e9
                    # Round toa and toa2 to avoid floating-point precision issues
                    sim_df['toa'] = sim_df['toa'].round(6)
                    traced_df['toa2'] = traced_df['toa2'].round(6)
                    # Merge sim_df with traced_df based on toa
                    if verbosity >= 2:
                        print(f"sim_df toa sample: {sim_df['toa'].head().tolist()}")
                        print(f"traced_df toa2 sample: {traced_df['toa2'].head().tolist()}")
                        print(f"Number of sim_df rows before merge: {len(sim_df)}")
                    sim_df = sim_df.merge(traced_df[['x2', 'y2', 'z2', 'toa2']], left_on='toa', right_on='toa2', how='left')
                    if verbosity >= 2:
                        print(f"Number of sim_df rows after merge: {len(sim_df)}")
                        print(f"Non-NaN x2 count: {sim_df['x2'].notna().sum()}")
                    # Compute photon_px, photon_py from x2, y2 for spatial matching
                    sim_df['photon_px'] = (sim_df['x2'] + 10) / 10 * 128
                    sim_df['photon_py'] = (sim_df['y2'] + 10) / 10 * 128
                
                # Initialize list to store merged rows
                merged_rows = []
                
                # Group simulation and reconstruction data by neutron_id
                sim_groups = sim_df.groupby('neutron_id')
                recon_groups = recon_df.groupby('neutron_id')
                
                # Iterate over all unique neutron_ids in simulation data
                for neutron_id in sorted(sim_df['neutron_id'].unique()):
                    sim_group = sim_groups.get_group(neutron_id) if neutron_id in sim_groups.groups else pd.DataFrame()
                    recon_group = recon_groups.get_group(neutron_id) if neutron_id in recon_groups.groups else pd.DataFrame()
                    
                    # Assign event_id to reconstruction events
                    if not recon_group.empty:
                        recon_group = recon_group.sort_values('t [s]').reset_index(drop=True)
                        recon_group['event_id'] = recon_group.index + 1
                    
                    # If no reconstruction data for this neutron_id, append sim rows with NaN for recon cols
                    if recon_group.empty:
                        for _, sim_row in sim_group.iterrows():
                            merged_row = sim_row.to_dict()
                            merged_row.update({col: np.nan for col in recon_df.columns if col != 'neutron_id'})
                            merged_row['event_id'] = np.nan
                            merged_row['time_diff_ns'] = np.nan
                            merged_row['spatial_diff_px'] = np.nan
                            merged_rows.append(merged_row)
                        continue
                    
                    # Track matched simulation indices to identify unmatched photons
                    matched_indices = []
                    
                    # Match reconstruction events to simulation photons
                    for recon_idx, recon_row in recon_group.iterrows():
                        n_photons = int(recon_row['nPhotons [1]'])  # Number of photons in this recon event
                        recon_time_ns = recon_row['t [s]'] * 1e9  # Convert to ns
                        recon_x = recon_row['x [px]']
                        recon_y = recon_row['y [px]']
                        event_id = recon_row['event_id']
                        
                        # Compute combined distance metric: temporal + spatial
                        sim_times = sim_group['toa'].values
                        sim_px = sim_group['photon_px'].values
                        sim_py = sim_group['photon_py'].values
                        
                        time_diffs = np.abs(sim_times - recon_time_ns)
                        spatial_diffs = np.sqrt((sim_px - recon_x)**2 + (sim_py - recon_y)**2)
                        
                        # Handle cases where photon_px, photon_py are NaN (no traced data)
                        if np.all(np.isnan(sim_px)) or np.all(np.isnan(sim_py)):
                            time_diffs_normalized = time_diffs / 1.0  # Use only time if spatial data is missing
                            combined_diffs = time_diffs_normalized
                            spatial_diffs = np.array([np.nan] * len(time_diffs))
                        else:
                            # Normalize distances for combined metric (1 ns ~ 1 px)
                            time_diffs_normalized = time_diffs / 1.0  # 1 ns
                            spatial_diffs_normalized = spatial_diffs / 1.0  # 1 px
                            combined_diffs = time_diffs_normalized + spatial_diffs_normalized
                        
                        if n_photons == 1:
                            if len(sim_group) > 0:
                                closest_idx = np.argmin(combined_diffs)
                                sim_row = sim_group.iloc[closest_idx]
                                merged_row = sim_row.to_dict()
                                for col in recon_df.columns:
                                    if col != 'neutron_id':
                                        merged_row[f'recon_{col}'] = recon_row[col]
                                merged_row['event_id'] = event_id
                                merged_row['time_diff_ns'] = time_diffs[closest_idx]
                                merged_row['spatial_diff_px'] = spatial_diffs[closest_idx]
                                merged_rows.append(merged_row)
                                matched_indices.append(closest_idx)
                        else:
                            # For n_photons > 1, select the top n_photons closest in combined distance
                            if len(sim_group) >= n_photons:
                                closest_indices = np.argsort(combined_diffs)[:n_photons]
                                # Verify center of mass
                                selected_px = sim_group.iloc[closest_indices]['photon_px']
                                selected_py = sim_group.iloc[closest_indices]['photon_py']
                                if np.all(np.isnan(selected_px)) or np.all(np.isnan(selected_py)):
                                    # Skip center-of-mass check if spatial data is missing
                                    for idx in closest_indices:
                                        sim_row = sim_group.iloc[idx]
                                        merged_row = sim_row.to_dict()
                                        for col in recon_df.columns:
                                            if col != 'neutron_id':
                                                merged_row[f'recon_{col}'] = recon_row[col]
                                        merged_row['event_id'] = event_id
                                        merged_row['time_diff_ns'] = time_diffs[idx]
                                        merged_row['spatial_diff_px'] = np.nan
                                        merged_rows.append(merged_row)
                                        matched_indices.append(idx)
                                else:
                                    com_x = selected_px.mean()
                                    com_y = selected_py.mean()
                                    com_dist = np.sqrt((com_x - recon_x)**2 + (com_y - recon_y)**2)
                                    if com_dist <= dSpace_px:  # Ensure center of mass is close enough
                                        for idx in closest_indices:
                                            sim_row = sim_group.iloc[idx]
                                            merged_row = sim_row.to_dict()
                                            for col in recon_df.columns:
                                                if col != 'neutron_id':
                                                    merged_row[f'recon_{col}'] = recon_row[col]
                                            merged_row['event_id'] = event_id
                                            merged_row['time_diff_ns'] = time_diffs[idx]
                                            merged_row['spatial_diff_px'] = spatial_diffs[idx]
                                            merged_rows.append(merged_row)
                                            matched_indices.append(idx)
                    
                    # Append unmatched simulation photons with NaN for reconstruction columns
                    unmatched_indices = [i for i in range(len(sim_group)) if i not in matched_indices]
                    for idx in unmatched_indices:
                        sim_row = sim_group.iloc[idx]
                        merged_row = sim_row.to_dict()
                        merged_row.update({col: np.nan for col in recon_df.columns if col != 'neutron_id'})
                        merged_row['event_id'] = np.nan
                        merged_row['time_diff_ns'] = np.nan
                        merged_row['spatial_diff_px'] = np.nan
                        merged_rows.append(merged_row)
                
                # Create merged dataframe
                merged_df = pd.DataFrame(merged_rows)
                
                # Ensure column order: simulation columns, traced columns, reconstruction columns
                sim_cols = [col for col in sim_df.columns if col not in ['x2', 'y2', 'z2', 'toa2', 'photon_px', 'photon_py']]
                traced_cols = ['x2', 'y2', 'z2', 'toa2', 'photon_px', 'photon_py']
                recon_cols = [f'recon_{col}' for col in recon_df.columns if col != 'neutron_id']
                final_cols = sim_cols + traced_cols + recon_cols + ['event_id', 'time_diff_ns', 'spatial_diff_px']
                
                # Reorder columns and handle missing columns
                merged_df = merged_df[[col for col in final_cols if col in merged_df.columns]]
                
                # Compute additional columns
                merged_df["x3"] = (128 - merged_df["recon_x [px]"]) / 256 * 120
                merged_df["y3"] = (128 - merged_df["recon_y [px]"]) / 256 * 120
                merged_df["delta_x"] = merged_df["x3"] - merged_df["nx"]
                merged_df["delta_y"] = merged_df["y3"] - merged_df["ny"]
                merged_df["delta_r"] = np.sqrt(merged_df["delta_x"]**2 + merged_df["delta_y"]**2)
                
                return merged_df
            
            def merge_sim_and_recon_data_no_prefix(sim_data, traced_data, recon_data):
                merged_df = merge_sim_and_recon_data(sim_data, traced_data, recon_data)
                rename_dict = {col: col.replace('recon_', '') for col in merged_df.columns if col.startswith('recon_')}
                merged_df = merged_df.rename(columns=rename_dict)
                return merged_df
            
            # Load traced data
            traced_data = pd.DataFrame()
            if traced_photons_path.exists():
                traced_files = list(traced_photons_path.glob("*.csv"))
                if traced_files:
                    traced_dfs = [pd.read_csv(f) for f in traced_files]
                    traced_data = pd.concat(traced_dfs, ignore_index=True)
                    if verbosity >= 1:
                        print(f"Loaded traced data with columns: {list(traced_data.columns)}")
                        print(f"Traced data toa2 sample: {traced_data.get('toa2', pd.Series()).head().tolist()}")
                else:
                    if verbosity >= 1:
                        print("Warning: No traced photon CSV files found in TracedPhotons folder.")
            
            merged_df = merge_sim_and_recon_data_no_prefix(self.sim_data, traced_data, combined_results)
            
            # Save merged results in suffixed folder
            merged_csv = suffix_dir / "merged_all_batches_results.csv"
            merged_df.to_csv(merged_csv, index=False)
            
            if verbosity >= 1:
                print(f"Merged results saved to {merged_csv}")
            
            return merged_df
        
        return combined_results


    def calculate_event_ellipsoid_shape(self, 
                                        results_df: pd.DataFrame=None,
                                        verbosity: VerbosityLevel = VerbosityLevel.QUIET) -> pd.DataFrame:
        """
        Calculates the ellipsoid shape parameters (major and minor axes) for each reconstructed event
        by identifying the original photon data points that correspond to each event.
        Also calculates parent track length and its X,Y projections, and includes the parent name.
        
        Args:
            results_df: DataFrame containing the reconstructed events (from process_data_event_by_event)
            verbosity: Level of output verbosity
        
        Returns:
            DataFrame with added ellipsoid shape parameters, track length metrics, and parent info for each event
        """
        from sklearn.decomposition import PCA

        if results_df is None or results_df.empty:
            # Check if the reconstructed events exist in the standard location
            reconstructed_events_path = self.archive / "all_batches_results.csv"
            if not reconstructed_events_path.exists():
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"No reconstructed events found at {reconstructed_events_path}")
                return pd.DataFrame()
            results_df = pd.read_csv(reconstructed_events_path)
        
        results_df.columns = [col.lstrip() for col in results_df.columns]
        
        if verbosity >= VerbosityLevel.BASIC:
            print("Calculating ellipsoid shape parameters and track lengths for reconstructed events...")
        
        try:
            # Combine data with neutron_id for reference
            combined_data = self.data.copy()
            
            # Check data length match
            if len(self.data) != len(self.sim_data):
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Warning: Data length mismatch: self.data has {len(self.data)} rows, self.sim_data has {len(self.sim_data)} rows")
                # Try to handle this by using only matching indices
                combined_data = self.data.iloc[:min(len(self.data), len(self.sim_data))].copy()
                sim_data_subset = self.sim_data.iloc[:min(len(self.data), len(self.sim_data))]
                combined_data['neutron_id'] = sim_data_subset['neutron_id']
            else:
                combined_data['neutron_id'] = self.sim_data['neutron_id']
            
            # Create a copy of results with added shape parameters
            enhanced_results = results_df.copy()
            enhanced_results['major_axis_px'] = np.nan
            enhanced_results['minor_axis_px'] = np.nan
            enhanced_results['orientation_deg'] = np.nan
            enhanced_results['original_photon_ids'] = None  # Will store lists of original photon indices
            
            # Add new columns for track length calculations
            enhanced_results['track_length_3d'] = np.nan
            enhanced_results['track_length_xy'] = np.nan
            enhanced_results['track_length_x'] = np.nan
            enhanced_results['track_length_y'] = np.nan
            enhanced_results['track_length_z'] = np.nan
            enhanced_results['parent_name'] = None  # Will store the parent particle name
            enhanced_results['parent_energy'] = np.nan  # Will store the parent particle energy if available
            
            # Group the original data by neutron_id for faster lookup
            neutron_groups = {}
            for neutron_id, group in combined_data.groupby('neutron_id'):
                neutron_groups[neutron_id] = group
            
            # Create a dictionary to store parent information by neutron_id for efficient lookup
            parent_info = {}
            try:
                # Group by neutron_id and select first row (parent info should be consistent within a neutron_id)
                for neutron_id, group in self.sim_data.groupby('neutron_id'):
                    first_row = group.iloc[0]
                    parent_info[neutron_id] = {
                        'parent_name': first_row.get('parentName', None),
                        'parent_energy': first_row.get('parentEnergy', None)
                    }
            except Exception as e:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Warning: Error building parent info dictionary: {str(e)}")
            
            # Process each reconstructed event
            for idx, event in tqdm(enhanced_results.iterrows(), total=len(enhanced_results), desc="Processing reconstructed events"):
                try:
                    neutron_id = event['neutron_id']
                    event_time = event['t [s]']
                    n_photons = event['nPhotons [1]']
                    
                    # Add parent information from our dictionary
                    if neutron_id in parent_info:
                        enhanced_results.at[idx, 'parent_name'] = parent_info[neutron_id]['parent_name']
                        enhanced_results.at[idx, 'parent_energy'] = parent_info[neutron_id]['parent_energy']
                    
                    # Skip if the event has only one photon (can't form an ellipse)
                    if n_photons <= 1:
                        continue
                    
                    # Get all photons for this neutron_id
                    if neutron_id not in neutron_groups:
                        if verbosity >= VerbosityLevel.BASIC:
                            print(f"Neutron ID {neutron_id} not found in original data")
                        continue
                        
                    neutron_data = neutron_groups[neutron_id]
                    
                    # Calculate track length from sim_data
                    try:
                        # Find the corresponding sim data for this neutron_id
                        sim_neutron_data = self.sim_data[self.sim_data['neutron_id'] == neutron_id]
                        
                        if len(sim_neutron_data) > 0:
                            # We only need one row since track info should be consistent for all photons of same neutron
                            first_row = sim_neutron_data.iloc[0]
                            
                            # Check if necessary columns exist
                            if all(col in first_row for col in ['nx', 'ny', 'nz', 'px', 'py', 'pz']):
                                # Calculate the 3D track length
                                nx, ny, nz = first_row['nx'], first_row['ny'], first_row['nz']
                                px, py, pz = first_row['px'], first_row['py'], first_row['pz']
                                
                                # 3D track length
                                track_length_3d = np.sqrt((nx - px)**2 + (ny - py)**2 + (nz - pz)**2)
                                
                                # XY projection length
                                track_length_xy = np.sqrt((nx - px)**2 + (ny - py)**2)
                                
                                # Individual axis projections
                                track_length_x = abs(nx - px)
                                track_length_y = abs(ny - py)
                                track_length_z = abs(nz - pz)
                                
                                # Store track length results
                                enhanced_results.at[idx, 'track_length_3d'] = track_length_3d
                                enhanced_results.at[idx, 'track_length_xy'] = track_length_xy
                                enhanced_results.at[idx, 'track_length_x'] = track_length_x
                                enhanced_results.at[idx, 'track_length_y'] = track_length_y
                                enhanced_results.at[idx, 'track_length_z'] = track_length_z
                                
                                if verbosity >= VerbosityLevel.DETAILED:
                                    print(f"Track length for event {idx} (neutron_id {neutron_id}): 3D={track_length_3d:.3f}, XY={track_length_xy:.3f}")
                    except Exception as e:
                        if verbosity >= VerbosityLevel.BASIC:
                            print(f"Error calculating track length for event {idx}: {str(e)}")
                    
                    # Convert TOA to seconds to match the event time format
                    neutron_data = neutron_data.copy()
                    
                    try:
                        neutron_data['toa_s'] = neutron_data['toa2'] * 1e-9
                    except KeyError:
                        # Handle case where 'toa2' column might be missing or named differently
                        if 'toa' in neutron_data.columns:
                            neutron_data['toa_s'] = neutron_data['toa'] * 1e-9
                        else:
                            if verbosity >= VerbosityLevel.BASIC:
                                print(f"Could not find time-of-arrival column for neutron ID {neutron_id}")
                            continue
                    
                    # Calculate the time window for this event (with a small buffer)
                    time_buffer = 1e-9  # 1 nanosecond buffer
                    
                    # First try: match by time proximity to the event time
                    time_diff = np.abs(neutron_data['toa_s'] - event_time)
                    
                    # Safely get the N smallest values, handling case where n_photons > available datapoints
                    n_indices = min(int(n_photons), len(time_diff))
                    if n_indices < 2:  # Need at least 2 points for an ellipse
                        continue
                        
                    closest_indices = time_diff.nsmallest(n_indices).index
                    
                    # Get the closest photons
                    photons_subset = neutron_data.loc[closest_indices]
                    
                    # Convert the photon positions to pixel coordinates
                    try:
                        photons_subset["px"] = (photons_subset["x2"] + 10) / 10 * 128
                        photons_subset["py"] = (photons_subset["y2"] + 10) / 10 * 128
                    except KeyError:
                        # Try alternative column names
                        if 'x' in photons_subset.columns and 'y' in photons_subset.columns:
                            photons_subset["px"] = (photons_subset["x"] + 10) / 10 * 128
                            photons_subset["py"] = (photons_subset["y"] + 10) / 10 * 128
                        else:
                            if verbosity >= VerbosityLevel.BASIC:
                                print(f"Could not find position columns for neutron ID {neutron_id}")
                            continue
                    
                    # Create a matrix of positions for PCA
                    if len(photons_subset) >= 2:
                        positions = photons_subset[['px', 'py']].values
                        
                        # Use PCA to find the principal axes
                        pca = PCA(n_components=2)
                        try:
                            pca.fit(positions)
                            
                            # The eigenvalues of the covariance matrix give us the variance along each principal axis
                            eigenvalues = pca.explained_variance_
                            
                            # Safety check for valid eigenvalues
                            if np.any(eigenvalues <= 0):
                                if verbosity >= VerbosityLevel.DETAILED:
                                    print(f"Invalid eigenvalues for event {idx}: {eigenvalues}")
                                continue
                            
                            # Calculate the major and minor axes (2 standard deviations = 95% confidence interval)
                            major_axis = 2.0 * np.sqrt(eigenvalues[0])
                            minor_axis = 2.0 * np.sqrt(eigenvalues[1])
                            
                            # Get the orientation (angle of the first principal component)
                            v = pca.components_[0]
                            angle_rad = np.arctan2(v[1], v[0])
                            angle_deg = np.degrees(angle_rad) % 180  # Convert to degrees, range 0-180
                            
                            # Store the results
                            enhanced_results.at[idx, 'major_axis_px'] = major_axis
                            enhanced_results.at[idx, 'minor_axis_px'] = minor_axis
                            enhanced_results.at[idx, 'orientation_deg'] = angle_deg
                            enhanced_results.at[idx, 'original_photon_ids'] = list(closest_indices)
                            
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"Event {idx} (neutron_id {neutron_id}): major={major_axis:.2f}px, minor={minor_axis:.2f}px, angle={angle_deg:.2f}°")
                        
                        except Exception as e:
                            if verbosity >= VerbosityLevel.BASIC:
                                print(f"PCA error for event {idx}: {str(e)}")
                            continue
                
                except Exception as e:
                    if verbosity >= VerbosityLevel.BASIC:
                        print(f"Error processing event {idx}: {str(e)}")
                    continue
            
            # Safely calculate derived metrics
            try:
                # Calculate ellipticity (ratio of major to minor axis)
                enhanced_results['ellipticity'] = enhanced_results['major_axis_px'] / enhanced_results['minor_axis_px']
                
                # Calculate area of the ellipse
                enhanced_results['area_px2'] = np.pi * enhanced_results['major_axis_px'] * enhanced_results['minor_axis_px']
            except Exception as e:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Error calculating derived metrics: {str(e)}")
            
            # Alternative approach: Join parent info to enhanced_results as a batch operation
            try:
                # Create a small dataframe with just neutron_id, parent_name and parent_energy
                parent_df = pd.DataFrame([
                    {'neutron_id': nid, 'parent_name': info['parent_name'], 'parent_energy': info['parent_energy']}
                    for nid, info in parent_info.items()
                ])
                
                # Merge with enhanced_results
                if not parent_df.empty:
                    # Use left join to keep all rows from enhanced_results
                    enhanced_results = pd.merge(
                        enhanced_results,
                        parent_df,
                        on='neutron_id',
                        how='left',
                        suffixes=('', '_new')
                    )
                    
                    # If we already added parent_name and parent_energy directly, we need to reconcile
                    if 'parent_name_new' in enhanced_results.columns:
                        # Fill any missing values from the direct assignment with the joined values
                        enhanced_results['parent_name'] = enhanced_results['parent_name'].fillna(enhanced_results['parent_name_new'])
                        enhanced_results['parent_energy'] = enhanced_results['parent_energy'].fillna(enhanced_results['parent_energy_new'])
                        
                        # Drop the duplicate columns
                        enhanced_results = enhanced_results.drop(columns=['parent_name_new', 'parent_energy_new'], errors='ignore')
                
                if verbosity >= VerbosityLevel.BASIC:
                    parent_count = enhanced_results['parent_name'].notna().sum()
                    print(f"Added parent information for {parent_count} out of {len(enhanced_results)} events")
            except Exception as e:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Error joining parent information: {str(e)}")
        
        except Exception as e:
            if verbosity >= VerbosityLevel.BASIC:
                print(f"Fatal error in calculate_event_ellipsoid_shape: {str(e)}")
            # Return the original DataFrame if we encountered a serious error
            return results_df
        
        return enhanced_results

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