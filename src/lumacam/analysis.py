import pandas as pd
import numpy as np
import os
import subprocess
from pathlib import Path
from tqdm.notebook import tqdm
from enum import IntEnum
from dataclasses import dataclass, field
import json
from typing import Dict, Any, Optional, Union
from multiprocessing import Pool
import glob
import shutil
import subprocess
from functools import partial
import lumacamTesting as lct

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

    def _process_single_file(self, 
                            file, 
                            verbosity: VerbosityLevel = VerbosityLevel.QUIET,
                            events_per_frame: int = None,
                            frame_time_s: float = None):
        try:
            df = pd.read_csv(file)
            if df.empty:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"⚠️ Skipping empty file: {file.name}")
                return
            processed_df = self._process_photon_dataframe(
                df=df,
                file_stem=file.stem,
                events_per_frame=events_per_frame,
                frame_time_s=frame_time_s,
                verbosity=verbosity
            )
            if processed_df.empty:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"⚠️ No valid data after processing {file.name}")
                return
            imported_photons_dir = self.archive / "ImportedPhotons"
            imported_photons_dir.mkdir(exist_ok=True)
            output_csv = imported_photons_dir / f"imported_{file.stem}.csv"
            processed_df.sort_values("t [s]").to_csv(output_csv, index=False)
            empir_file = self.photon_files_dir / f"{file.stem}.empirphot"
            result = os.system(f"{self.empir_import_photons} {output_csv} {empir_file} csv")
            if result == 0:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"✔ Processed {file.name} → {empir_file.name}")
            else:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"❌ empir_import_photons failed for {file.name}")
        except Exception as e:
            if verbosity >= VerbosityLevel.BASIC: 
                print(f"❌ Error processing {file.name}: {e}")

    def _run_import_photons(self, 
                            parallel=True,
                            verbosity: VerbosityLevel = VerbosityLevel.QUIET,
                            events_per_frame: int = None,
                            frame_time_s: float = None):
        traced_files = sorted(self.traced_dir.glob("traced_sim_data_*.csv"))
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Processing {len(traced_files)} traced photon files...")
        if parallel:
            with Pool() as pool:
                list(tqdm(
                    pool.imap_unordered(
                        partial(self._process_single_file, 
                                verbosity=verbosity, 
                                events_per_frame=events_per_frame, 
                                frame_time_s=frame_time_s),
                        traced_files
                    ),
                    total=len(traced_files),
                    desc="Processing files"
                ))
        else:
            for file in tqdm(traced_files, desc="Processing files"):
                self._process_single_file(file, verbosity, events_per_frame, frame_time_s)
        if verbosity >= VerbosityLevel.BASIC:
            print("✅ Finished processing all files!")

    def _run_photon2event(self, archive: str = None, 
                          config: Photon2EventConfig = None,
                          verbosity: VerbosityLevel = VerbosityLevel.BASIC,
                          **config_kwargs):
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
        if archive is None:
            archive = self.archive
        archive = Path(archive)
        input_folder = archive / "EventFiles"
        if not input_folder.exists():
            raise FileNotFoundError(f"No eventFiles folder found at {input_folder}")
        params_file = archive / "parameterEvents.json"
        if config is None:
            config = EventBinningConfig(**config_kwargs)
        config.write(params_file)
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
        if archive is None:
            archive = self.archive
        archive = Path(archive)
        binned_data = pd.read_csv(archive/"binned.empirevent", header=0)
        return binned_data

    def _process_photon_dataframe(self, 
                                 df: pd.DataFrame, 
                                 file_stem: str = None,
                                 events_per_frame: int = None,
                                 frame_time_s: float = None,
                                 verbosity: VerbosityLevel = VerbosityLevel.QUIET) -> pd.DataFrame:
        try:
            initial_rows = len(df)
            df = df[["x2", "y2", "toa2"]].dropna().copy()
            if verbosity >= VerbosityLevel.DETAILED and len(df) < initial_rows:
                print(f"Dropped {initial_rows - len(df)} rows with NaN values")
            if df.empty:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"⚠️ Empty DataFrame after cleaning")
                return df
            df["toa2"] *= 1e-9
            df["px"] = (df["x2"] + 10) / 10 * 128
            df["py"] = (df["y2"] + 10) / 10 * 128
            df = df[["px", "py", "toa2"]]
            df.columns = ["x [px]", "y [px]", "t [s]"]
            df["t_relToExtTrigger [s]"] = df["t [s]"]
            if events_per_frame is not None and frame_time_s is not None:
                if events_per_frame <= 0:
                    if verbosity >= VerbosityLevel.BASIC:
                        print(f"⚠️ Invalid events_per_frame ({events_per_frame}). Skipping frame-based time offsets.")
                    events_per_frame = None
                elif frame_time_s < 0:
                    if verbosity >= VerbosityLevel.BASIC:
                        print(f"⚠️ Invalid frame_time_s ({frame_time_s}). Skipping frame-based time offsets.")
                    frame_time_s = None
                else:
                    if 'neutron_id' not in self.sim_data.columns:
                        if verbosity >= VerbosityLevel.BASIC:
                            print(f"⚠️ neutron_id not found in sim_data. Assigning arbitrary neutron IDs")
                        avg_photons_per_event = 10
                        total_events = max(1, len(df) // avg_photons_per_event)
                        events_per_group = min(total_events, events_per_frame)
                        if events_per_group == 0:
                            events_per_group = 1
                        df['neutron_id'] = (df.index * events_per_group // len(df)).astype(int)
                    else:
                        if len(self.sim_data) < len(df):
                            if verbosity >= VerbosityLevel.BASIC:
                                print(f"⚠️ sim_data has fewer rows ({len(self.sim_data)}) than df ({len(df)}). Truncating.")
                            df = df.iloc[:len(self.sim_data)].copy()
                            df['neutron_id'] = self.sim_data['neutron_id'].iloc[:len(df)].values
                        else:
                            df['neutron_id'] = self.sim_data['neutron_id'].iloc[:len(df)].values
                    neutron_ids = df['neutron_id'].values
                    unique_neutron_ids, indices = np.unique(neutron_ids, return_index=True)
                    sorted_indices = np.argsort(indices)
                    unique_neutron_ids = unique_neutron_ids[sorted_indices]
                    neutron_to_frame = {nid: i // events_per_frame for i, nid in enumerate(unique_neutron_ids)}
                    df['frame'] = df['neutron_id'].map(neutron_to_frame)
                    df['t [s]'] = df['t [s]'] + df['frame'] * frame_time_s
                    df = df.drop(columns=['neutron_id', 'frame'])
            initial_rows = len(df)
            df = df.loc[(df["t [s]"] >= 0) & (df["t [s]"] < 1)]
            if verbosity >= VerbosityLevel.DETAILED and len(df) < initial_rows:
                print(f"Dropped {initial_rows - len(df)} rows due to time filter (t [s] < 0 or >= 1)")
            return df
        except Exception as e:
            if verbosity >= VerbosityLevel.BASIC:
                print(f"❌ Error processing DataFrame: {e}")
            return pd.DataFrame()

    def convert_traced_to_imported(self, traced_df, verbosity=VerbosityLevel.QUIET):
        """Convert traced photon data to imported format using the provided conversion logic"""
        if verbosity >= VerbosityLevel.DETAILED:
            print("Converting traced data to imported format...")
        df = traced_df[["x2", "y2", "toa2"]].dropna().copy()
        if df.empty:
            if verbosity >= VerbosityLevel.BASIC:
                print("⚠️ Empty DataFrame after cleaning traced data")
            return df
        df["toa2"] *= 1e-9
        df["px"] = (df["x2"] + 10) / 10 * 128
        df["py"] = (df["y2"] + 10) / 10 * 128
        df = df[["px", "py", "toa2"]]
        df.columns = ["x [px]", "y [px]", "t [s]"]
        df["t_relToExtTrigger [s]"] = df["t [s]"]
        initial_rows = len(df)
        df = df.loc[(df["t [s]"] >= 0) & (df["t [s]"] < 1)]
        if verbosity >= VerbosityLevel.DETAILED and len(df) < initial_rows:
            print(f"Dropped {initial_rows - len(df)} rows due to time filter (t [s] < 0 or >= 1)")
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Converted {len(df)} traced photons to imported format")
        return df

    def merge_simulation_with_traced(self, sim_df, imported_df, verbosity=VerbosityLevel.QUIET):
        """Merge simulation data with traced/imported data by matching toa values"""
        if verbosity >= VerbosityLevel.BASIC:
            print("Merging simulation with traced data...")
        if len(sim_df) != len(imported_df):
            if verbosity >= VerbosityLevel.BASIC:
                print(f"Length mismatch: sim={len(sim_df)}, traced={len(imported_df)}")
            min_len = min(len(sim_df), len(imported_df))
            sim_df = sim_df.iloc[:min_len].copy()
            imported_df = imported_df.iloc[:min_len].copy()
        merged_df = sim_df.copy()
        for col in imported_df.columns:
            merged_df[f"imported_{col}"] = imported_df[col].values
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Successfully merged {len(merged_df)} photons")
        return merged_df

    def fix_time_ordering(self, df, dataset_name, verbosity=VerbosityLevel.QUIET):
        """Ensure strict time ordering by adding small increments to duplicate times"""
        time_diff = df['t_s'].diff()
        non_increasing = (time_diff <= 0).sum()
        if non_increasing > 0:
            for i in range(1, len(df)):
                if df.loc[i, 't_s'] <= df.loc[i-1, 't_s']:
                    df.loc[i, 't_s'] = df.loc[i-1, 't_s'] + 1e-12
        return df

    def calculate_association_parameters(self, imported_events, processed_events, verbosity=VerbosityLevel.QUIET):
        """Calculate appropriate parameters for event association"""
        all_x = pd.concat([imported_events['x_px'], processed_events['x_px']])
        all_y = pd.concat([imported_events['y_px'], processed_events['y_px']])
        all_t = pd.concat([imported_events['t_s'], processed_events['t_s']])
        spatial_scale = np.sqrt(np.var(all_x) + np.var(all_y))
        temporal_scale = np.std(all_t)
        weight_px_in_s = temporal_scale / spatial_scale
        max_dist_s = 3 * temporal_scale
        if verbosity >= VerbosityLevel.DETAILED:
            print(f"Association parameters:")
            print(f"  Spatial scale: {spatial_scale:.1f} px")
            print(f"  Temporal scale: {temporal_scale*1e9:.1f} ns")
            print(f"  Weight (px→s): {weight_px_in_s:.2e}")
            print(f"  Max distance: {max_dist_s*1e9:.1f} ns")
        return weight_px_in_s, max_dist_s

    def run_event_association(self, merged_sim_traced, processed_df, verbosity=VerbosityLevel.QUIET):
        """Run event association between imported photons and processed events"""
        if verbosity >= VerbosityLevel.BASIC:
            print("Running event association...")
        imported_events = pd.DataFrame({
            't_s': merged_sim_traced['imported_t [s]'],
            'x_px': merged_sim_traced['imported_x [px]'],
            'y_px': merged_sim_traced['imported_y [px]']
        })
        processed_events = pd.DataFrame({
            't_s': processed_df['t [s]'],
            'x_px': processed_df['x [px]'],
            'y_px': processed_df['y [px]']
        })
        imported_events = self.fix_time_ordering(imported_events.sort_values('t_s').reset_index(drop=True), "imported", verbosity)
        processed_events = self.fix_time_ordering(processed_events.sort_values('t_s').reset_index(drop=True), "processed", verbosity)
        weight_px_in_s, max_dist_s = self.calculate_association_parameters(imported_events, processed_events, verbosity)
        eventAssociation = lct.EventAssociation.make_individualShortestConnection(
            weight_px_in_s, max_dist_s,
            imported_events[['t_s', 'x_px', 'y_px']],
            processed_events[['t_s', 'x_px', 'y_px']]
        )
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Association complete: {len(eventAssociation.clusters)} clusters found")
            if verbosity >= VerbosityLevel.DETAILED:
                print("Cluster summary:")
                print(eventAssociation.clusters)
        return {
            'association': eventAssociation,
            'imported_events': imported_events,
            'processed_events': processed_events
        }

    def create_tidy_dataframe(self, merged_sim_traced, processed_df, association_results, verbosity=VerbosityLevel.QUIET):
        """Create final tidy dataframe with simulation data + associated event info"""
        if verbosity >= VerbosityLevel.BASIC:
            print("Creating tidy merged dataframe...")
        final_df = merged_sim_traced.copy()
        processed_columns = ['assoc_t [s]', 'assoc_x [px]', 'assoc_y [px]', 'assoc_nPhotons [1]', 
                            'assoc_PSD_value', 'assoc_cluster_id']
        for col in processed_columns:
            final_df[col] = np.nan
        eventAssociation = association_results['association']
        imported_events = association_results['imported_events']
        processed_events = association_results['processed_events']
        cluster_associations = eventAssociation.clusterAssociation_groundTruth
        for cluster_id in eventAssociation.clusters.index:
            photon_indices = np.where(cluster_associations == cluster_id)[0]
            processed_indices = np.where(eventAssociation.clusterAssociation_toTest == cluster_id)[0]
            if len(processed_indices) > 0:
                proc_idx = processed_indices[0]
                proc_event = processed_events.iloc[proc_idx]
                proc_original = processed_df.iloc[proc_idx]
                final_df.loc[photon_indices, 'assoc_t [s]'] = proc_event['t_s']
                final_df.loc[photon_indices, 'assoc_x [px]'] = proc_event['x_px']
                final_df.loc[photon_indices, 'assoc_y [px]'] = proc_event['y_px']
                final_df.loc[photon_indices, 'assoc_nPhotons [1]'] = proc_original['nPhotons [1]']
                final_df.loc[photon_indices, 'assoc_PSD_value'] = proc_original['PSD value']
                final_df.loc[photon_indices, 'assoc_cluster_id'] = cluster_id
        total_photons = len(final_df)
        associated_photons = final_df['assoc_cluster_id'].notna().sum()
        if verbosity >= VerbosityLevel.BASIC:
            print(f"Association summary:")
            print(f"  Total photons: {total_photons}")
            print(f"  Associated photons: {associated_photons} ({associated_photons/total_photons*100:.1f}%)")
            print(f"  Unassociated photons: {total_photons - associated_photons}")
        return final_df

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
                     events_per_frame: int = None,
                     frame_time_s: float = None,
                     save_to_imported_photons: bool = False,
                     method: str = "default") -> pd.DataFrame:
        """
        Streamlined method to process data through the complete analysis pipeline.
        If method="lct", uses lumacamTesting for event association.
        
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
            events_per_frame: Number of neutron events per frame for time delay
            frame_time_s: Time delay between frames in seconds
            save_to_imported_photons: If True, save photon data to ImportedPhotons
            method: Processing method ("default" or "lct" for lumacamTesting)
            
        Returns:
            DataFrame with processed data
        """
        analysed_dir = self.archive / "AnalysedResults"
        analysed_dir.mkdir(parents=True, exist_ok=True)
        imported_photons_dir = self.archive / "ImportedPhotons"
        imported_photons_dir.mkdir(parents=True, exist_ok=True)
        suffix_dir = analysed_dir / (suffix.strip("_") if suffix else "default")
        suffix_dir.mkdir(parents=True, exist_ok=True)

        if method == "lct":
            if verbosity >= VerbosityLevel.BASIC:
                print("Processing data with lumacamTesting (lct) method...")
            try:
                # Load simulation data if not already loaded
                sim_files = sorted(self.sim_dir.glob("sim_data_*.csv"))
                if not sim_files:
                    raise FileNotFoundError(f"No sim_data_*.csv files found in {self.sim_dir}")
                sim_df = pd.concat([pd.read_csv(f) for f in sim_files], ignore_index=True)

                # Convert traced data to imported format
                imported_df = self.convert_traced_to_imported(self.data, verbosity)

                # Merge simulation with imported data
                merged_sim_traced = self.merge_simulation_with_traced(sim_df, imported_df, verbosity)

                # Run photon2event and event binning to get processed events
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
                
                processed_df = self._read_binned_data()
                if nPhotons_bins is None:
                    processed_df.columns = ["stacks", "counts"]
                else:
                    processed_df.columns = ["stacks", "nPhotons", "counts"]
                
                # Run lumacamTesting event association
                association_results = self.run_event_association(merged_sim_traced, processed_df, verbosity)
                
                # Create final tidy dataframe
                result_df = self.create_tidy_dataframe(merged_sim_traced, processed_df, association_results, verbosity)
                
                # Save results
                output_csv = suffix_dir / "lct_results.csv"
                result_df.to_csv(output_csv, index=False)
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"LCT processed data saved to {output_csv}")
                
                return result_df

            except Exception as e:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Error in lct processing: {e}")
                return pd.DataFrame()
        
        else:
            if verbosity >= VerbosityLevel.BASIC:
                print("Processing data with default method...")
            if save_to_imported_photons:
                if verbosity >= VerbosityLevel.BASIC:
                    print("Processing and saving photon data to ImportedPhotons...")
                if self.data is None or self.data.empty or not all(col in self.data.columns for col in ["x2", "y2", "toa2"]):
                    traced_files = sorted(self.traced_dir.glob("traced_sim_data_*.csv"))
                    if not traced_files:
                        raise FileNotFoundError(f"No traced simulation data found in {self.traced_dir}")
                    for file in tqdm(traced_files, desc="Processing traced files", disable=(verbosity < VerbosityLevel.BASIC)):
                        df = pd.read_csv(file)
                        if df.empty:
                            if verbosity >= VerbosityLevel.BASIC:
                                print(f"⚠️ Skipping empty file: {file.name}")
                            continue
                        processed_df = self._process_photon_dataframe(
                            df=df,
                            file_stem=file.stem,
                            events_per_frame=events_per_frame,
                            frame_time_s=frame_time_s,
                            verbosity=verbosity
                        )
                        if not processed_df.empty:
                            output_csv = imported_photons_dir / f"imported_{file.stem}.csv"
                            processed_df.sort_values("t [s]").to_csv(output_csv, index=False)
                            if verbosity >= VerbosityLevel.BASIC:
                                print(f"Saved photon data to {output_csv}")
                            empir_file = self.photon_files_dir / f"{file.stem}.empirphot"
                            result = os.system(f"{self.empir_import_photons} {output_csv} {empir_file} csv")
                            if result == 0 and verbosity >= VerbosityLevel.BASIC:
                                print(f"✔ Generated {empir_file.name}")
                            elif result != 0 and verbosity >= VerbosityLevel.BASIC:
                                print(f"❌ empir_import_photons failed for {file.name}")
                else:
                    processed_df = self._process_photon_dataframe(
                        df=self.data,
                        file_stem="combined_data",
                        events_per_frame=events_per_frame,
                        frame_time_s=frame_time_s,
                        verbosity=verbosity
                    )
                    if not processed_df.empty:
                        output_csv = imported_photons_dir / "imported_combined_data.csv"
                        processed_df.sort_values("t [s]").to_csv(output_csv, index=False)
                        if verbosity >= VerbosityLevel.BASIC:
                            print(f"Saved photon data to {output_csv}")
                        empir_file = self.photon_files_dir / "combined_data.empirphot"
                        result = os.system(f"{self.empir_import_photons} {output_csv} {empir_file} csv")
                        if result == 0 and verbosity >= VerbosityLevel.BASIC:
                            print(f"✔ Generated {empir_file.name}")
                        elif result != 0 and verbosity >= VerbosityLevel.BASIC:
                            print(f"❌ empir_import_photons failed for combined_data")
            else:
                self._run_import_photons(
                    parallel=True,
                    verbosity=verbosity,
                    events_per_frame=events_per_frame,
                    frame_time_s=frame_time_s
                )
            p2e_config = self.Photon2EventConfig()
            p2e_config.dSpace_px = dSpace_px
            p2e_config.dTime_s = dTime_s
            p2e_config.durationMax_s = durationMax_s
            p2e_config.dTime_ext = dTime_ext
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

    def process_data_advanced(self,
                             photon2event_config: dict = None,
                             event_binning_config: dict = None,
                             binning_type: str = "time",
                             verbosity: VerbosityLevel = VerbosityLevel.QUIET) -> pd.DataFrame:
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
                                   focus_factor: float = 1.2) -> pd.DataFrame:
        analysed_dir = self.archive / "AnalysedResults"
        analysed_dir.mkdir(parents=True, exist_ok=True)
        imported_photons_dir = self.archive / "ImportedPhotons"
        imported_photons_dir.mkdir(parents=True, exist_ok=True)
        suffix_dir = analysed_dir / (suffix.strip("_") if suffix else "default")
        suffix_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = self.archive / "temp_events"
        temp_dir.mkdir(parents=True, exist_ok=True)
        result_dir = self.archive / "EventResults"
        result_dir.mkdir(parents=True, exist_ok=True)
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
        neutron_groups = combined_data.groupby('neutron_id')
        batch_results = {i: [] for i in range(num_batches)}
        p2e_config = self.Photon2EventConfig()
        p2e_config.dSpace_px = dSpace_px
        p2e_config.dTime_s = dTime_s
        p2e_config.durationMax_s = durationMax_s
        p2e_config.dTime_ext = dTime_ext
        params_file = suffix_dir / "parameterSettings.json"
        p2e_config.write(params_file)
        binning_config = self.EventBinningConfig().time_binning()
        binning_config.binning_t.nBins = nBins
        binning_config.binning_t.resolution_s = binning_time_resolution
        binning_config.binning_t.offset_s = binning_offset
        event_params_file = suffix_dir / "parameterEvents.json"
        binning_config.write(event_params_file)
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
                imported_csv = imported_photons_dir / f"imported_{event_prefix}.csv"
                df.sort_values("t [s]").to_csv(imported_csv, index=False)
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
        combined_csv = suffix_dir / "all_batches_results.csv"
        combined_results.to_csv(combined_csv, index=False)
        if verbosity >= 1:
            print(f"All results saved to {combined_csv}")
            print(f"Processed {len(combined_results)} neutron events across {len(batch_results)} batches")
        if merge:
            if verbosity >= 1:
                print("Merging processed results with simulation and traced data...")
            def merge_sim_and_recon_data(sim_data, traced_data, recon_data, focus_factor: float = 1.2):
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
                        if verbosity >= 1:
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
                        if verbosity >= 2:
                            print(f"Assigned traced columns. Non-NaN counts: x2={sim_df['x2'].notna().sum()}, toa2={sim_df['toa2'].notna().sum()}")
                    else:
                        if verbosity >= 1:
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
                        sim_times = sim_group['toa2'].values
                        sim_px = sim_group['photon_px'].values
                        sim_py = sim_group['photon_py'].values
                        time_diffs = np.abs(sim_times * 1e-9 - recon_time_s) * 1e9
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
                merged_df = self.calculate_reconstruction_stats(merged_df, focus_factor=focus_factor)
                sim_cols = [col for col in sim_df.columns if col not in ['x2', 'y2', 'z2', 'toa2', 'photon_px', 'photon_py']]
                traced_cols = ['x2', 'y2', 'z2', 'toa2', 'photon_px', 'photon_py']
                recon_cols = [col for col in recon_df.columns if col != 'neutron_id']
                final_cols = sim_cols + traced_cols + recon_cols + ['event_id', 'time_diff_ns', 'spatial_diff_px', 'x3', 'y3', 'delta_x', 'delta_y', 'delta_r']
                merged_df = merged_df[[col for col in final_cols if col in merged_df.columns]]
                return merged_df
            traced_data = pd.DataFrame()
            if traced_photons_path.exists():
                traced_files = list(traced_photons_path.glob("*.csv"))
                if traced_files:
                    traced_dfs = [pd.read_csv(f) for f in traced_files]
                    traced_data = pd.concat(traced_dfs, ignore_index=True)
                    if verbosity >= 1:
                        print(f"Loaded traced data with columns: {list(traced_data.columns)}")
                else:
                    if verbosity >= 1:
                        print("Warning: No traced photon CSV files found in TracedPhotons folder.")
            merged_df = merge_sim_and_recon_data(self.sim_data, traced_data, combined_results, focus_factor=focus_factor)
            merged_csv = suffix_dir / "merged_all_batches_results.csv"
            merged_df.to_csv(merged_csv, index=False)
            if verbosity >= 1:
                print(f"Merged results saved to {merged_csv}")
            return merged_df
        return combined_results

    def calculate_reconstruction_stats(self, df: pd.DataFrame, focus_factor: float = 1.2):
        df["x3"] = (128 - df["x [px]"]) / 256 * 120 * focus_factor
        df["y3"] = (128 - df["y [px]"]) / 256 * 120 * focus_factor
        df["delta_x"] = df["x3"] - df["nx"]
        df["delta_y"] = df["y3"] - df["ny"]
        df["delta_r"] = np.sqrt(df["delta_x"]**2 + df["delta_y"]**2)
        return df

    def calculate_event_ellipsoid_shape(self, 
                                       results_df: pd.DataFrame = None,
                                       verbosity: VerbosityLevel = VerbosityLevel.QUIET) -> pd.DataFrame:
        from sklearn.decomposition import PCA
        if results_df is None or results_df.empty:
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
            combined_data = self.data.copy()
            if len(self.data) != len(self.sim_data):
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Warning: Data length mismatch: self.data has {len(self.data)} rows, self.sim_data has {len(self.sim_data)} rows")
                combined_data = self.data.iloc[:min(len(self.data), len(self.sim_data))].copy()
                sim_data_subset = self.sim_data.iloc[:min(len(self.data), len(self.sim_data))]
                combined_data['neutron_id'] = sim_data_subset['neutron_id']
            else:
                combined_data['neutron_id'] = self.sim_data['neutron_id']
            enhanced_results = results_df.copy()
            enhanced_results['major_axis_px'] = np.nan
            enhanced_results['minor_axis_px'] = np.nan
            enhanced_results['orientation_deg'] = np.nan
            enhanced_results['original_photon_ids'] = None
            enhanced_results['track_length_3d'] = np.nan
            enhanced_results['track_length_xy'] = np.nan
            enhanced_results['track_length_x'] = np.nan
            enhanced_results['track_length_y'] = np.nan
            enhanced_results['track_length_z'] = np.nan
            enhanced_results['parent_name'] = None
            enhanced_results['parent_energy'] = np.nan
            neutron_groups = {}
            for neutron_id, group in combined_data.groupby('neutron_id'):
                neutron_groups[neutron_id] = group
            parent_info = {}
            try:
                for neutron_id, group in self.sim_data.groupby('neutron_id'):
                    first_row = group.iloc[0]
                    parent_info[neutron_id] = {
                        'parent_name': first_row.get('parentName', None),
                        'parent_energy': first_row.get('parentEnergy', None)
                    }
            except Exception as e:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Warning: Error building parent info dictionary: {str(e)}")
            for idx, event in tqdm(enhanced_results.iterrows(), total=len(enhanced_results), desc="Processing reconstructed events"):
                try:
                    neutron_id = event['neutron_id']
                    event_time = event['t [s]']
                    n_photons = event['nPhotons [1]']
                    if neutron_id in parent_info:
                        enhanced_results.at[idx, 'parent_name'] = parent_info[neutron_id]['parent_name']
                        enhanced_results.at[idx, 'parent_energy'] = parent_info[neutron_id]['parent_energy']
                    if n_photons <= 1:
                        continue
                    if neutron_id not in neutron_groups:
                        if verbosity >= VerbosityLevel.BASIC:
                            print(f"Neutron ID {neutron_id} not found in original data")
                        continue
                    neutron_data = neutron_groups[neutron_id]
                    try:
                        sim_neutron_data = self.sim_data[self.sim_data['neutron_id'] == neutron_id]
                        if len(sim_neutron_data) > 0:
                            first_row = sim_neutron_data.iloc[0]
                            if all(col in first_row for col in ['nx', 'ny', 'nz', 'px', 'py', 'pz']):
                                nx, ny, nz = first_row['nx'], first_row['ny'], first_row['nz']
                                px, py, pz = first_row['px'], first_row['py'], first_row['pz']
                                track_length_3d = np.sqrt((nx - px)**2 + (ny - py)**2 + (nz - pz)**2)
                                track_length_xy = np.sqrt((nx - px)**2 + (ny - py)**2)
                                track_length_x = abs(nx - px)
                                track_length_y = abs(ny - py)
                                track_length_z = abs(nz - pz)
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
                    neutron_data = neutron_data.copy()
                    try:
                        neutron_data['toa_s'] = neutron_data['toa2'] * 1e-9
                    except KeyError:
                        if 'toa' in neutron_data.columns:
                            neutron_data['toa_s'] = neutron_data['toa'] * 1e-9
                        else:
                            if verbosity >= VerbosityLevel.BASIC:
                                print(f"Could not find time-of-arrival column for neutron ID {neutron_id}")
                            continue
                    time_buffer = 1e-9
                    time_diff = np.abs(neutron_data['toa_s'] - event_time)
                    n_indices = min(int(n_photons), len(time_diff))
                    if n_indices < 2:
                        continue
                    closest_indices = time_diff.nsmallest(n_indices).index
                    photons_subset = neutron_data.loc[closest_indices]
                    try:
                        photons_subset["px"] = (photons_subset["x2"] + 10) / 10 * 128
                        photons_subset["py"] = (photons_subset["y2"] + 10) / 10 * 128
                    except KeyError:
                        if 'x' in photons_subset.columns and 'y' in photons_subset.columns:
                            photons_subset["px"] = (photons_subset["x"] + 10) / 10 * 128
                            photons_subset["py"] = (photons_subset["y"] + 10) / 10 * 128
                        else:
                            if verbosity >= VerbosityLevel.BASIC:
                                print(f"Could not find position columns for neutron ID {neutron_id}")
                            continue
                    if len(photons_subset) >= 2:
                        positions = photons_subset[['px', 'py']].values
                        pca = PCA(n_components=2)
                        try:
                            pca.fit(positions)
                            eigenvalues = pca.explained_variance_
                            if np.any(eigenvalues <= 0):
                                if verbosity >= VerbosityLevel.DETAILED:
                                    print(f"Invalid eigenvalues for event {idx}: {eigenvalues}")
                                continue
                            major_axis = 2.0 * np.sqrt(eigenvalues[0])
                            minor_axis = 2.0 * np.sqrt(eigenvalues[1])
                            v = pca.components_[0]
                            angle_rad = np.arctan2(v[1], v[0])
                            angle_deg = np.degrees(angle_rad) % 180
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
            try:
                enhanced_results['ellipticity'] = enhanced_results['major_axis_px'] / enhanced_results['minor_axis_px']
                enhanced_results['area_px2'] = np.pi * enhanced_results['major_axis_px'] * enhanced_results['minor_axis_px']
            except Exception as e:
                if verbosity >= VerbosityLevel.BASIC:
                    print(f"Error calculating derived metrics: {str(e)}")
            try:
                parent_df = pd.DataFrame([
                    {'neutron_id': nid, 'parent_name': info['parent_name'], 'parent_energy': info['parent_energy']}
                    for nid, info in parent_info.items()
                ])
                if not parent_df.empty:
                    enhanced_results = pd.merge(
                        enhanced_results,
                        parent_df,
                        on='neutron_id',
                        how='left',
                        suffixes=('', '_new')
                    )
                    if 'parent_name_new' in enhanced_results.columns:
                        enhanced_results['parent_name'] = enhanced_results['parent_name'].fillna(enhanced_results['parent_name_new'])
                        enhanced_results['parent_energy'] = enhanced_results['parent_energy'].fillna(enhanced_results['parent_energy_new'])
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
            return results_df
        return enhanced_results

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
        import shutil
        from pathlib import Path
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