import os
import subprocess
import re
from dataclasses import dataclass
from importlib import resources
import shutil
from typing import Optional, Tuple, List
from pathlib import Path
from enum import IntEnum
from tqdm.notebook import tqdm
import pandas as pd
import threading
import queue
import time
import glob

class VerbosityLevel(IntEnum):
    """Verbosity levels for simulation output."""
    QUIET = 0    # Show nothing except progress bar
    BASIC = 1    # Show progress bar and basic info
    DETAILED = 2 # Show everything

@dataclass
class Config:
    """Configuration for Geant4 simulation."""
    # GPS parameters
    particle: str = "neutron"
    energy: float = 10.0
    energy_unit: str = "MeV"
    energy_type: str = "Mono"  # Can be "Mono", "Lin", or "Hist" for histogram distribution
    energy_min: Optional[float] = None  # Used for Lin distribution
    energy_max: Optional[float] = None  # Used for Lin distribution
    energy_gradient: Optional[float] = None  # Used for Lin distribution
    energy_intercept: Optional[float] = None  # Used for Lin distribution
    energy_histogram: Optional[List[Tuple[float, float]]] = None  # Used for Hist distribution: (energy, intensity)
    position_x: float = 0
    position_y: float = 0
    position_z: float = -1059
    position_unit: str = "cm"
    direction_x: float = 0
    direction_y: float = 0
    direction_z: float = 1
    shape: str = "Rectangle"
    halfx: float = 60
    halfy: float = 60
    shape_unit: str = "mm"
    angle_type: str = "iso"
    max_theta: float = 0
    min_theta: float = 0
    angle_unit: str = "deg"
    
    # Time spread parameters
    tmin: float = 0.0  # Minimum time in ns
    tmax: float = 0.0  # Maximum time in ns
    time_unit: str = "ns"
    
    # Pulse parameters
    flux: Optional[float] = None  # Neutron flux in n/cm²/s
    freq: Optional[float] = None  # Pulse frequency in Hz
    
    sample_material: str = "G4_Galactic"  # Material of the sample
    scintillator: str = "EJ200"  # Scintillator type: PVT, EJ-200, GS20
    sample_thickness: float = 0.2  # Sample thickness in cm (default 0.2 cm = 200 microns)
    sample_width: float = 12.0  # Sample width in cm (default 12 cm)  
    scintillator_thickness: float = 20  # Scintillator thickness in mm (default is 20 mm)
    csv_batch_size: int = 0
    # Ion parameters for radioactive decay
    ion_z: Optional[int] = None  # Atomic number
    ion_a: Optional[int] = None  # Mass number
    ion_excitation: float = 0.0  # Excitation energy in keV
    
    # Run parameters
    num_events: int = 100000
    progress_interval: int = 100
    csv_filename: str = "sim_data.csv"

    @classmethod
    def neutrons_uniform(cls) -> 'Config':
        """Default neutron configuration with uniform distribution."""
        return cls(
            particle="neutron",
            energy=10.0,
            energy_unit="MeV",
            position_z=-1059,
            position_unit="cm",
            halfx=60,
            halfy=60,
            shape_unit="mm",
            num_events=100000,
            progress_interval=100,
            csv_filename="sim_data.csv",
            sample_material="G4_GRAPHITE",
            scintillator="EJ200",
            sample_thickness=2,
            csv_batch_size=1000,
        )
    
    @classmethod
    def neutrons_uniform_energy(cls) -> 'Config':
        """Neutron configuration with uniform spatial and energy distribution."""
        return cls(
            particle="neutron",
            energy_type="Lin",
            energy_min=2.0,
            energy_max=10.0,
            energy_gradient=0.0,
            energy_intercept=1.0,
            energy_unit="MeV",
            position_z=-1059,
            position_unit="cm",
            halfx=60,
            halfy=60,
            shape_unit="mm",
            num_events=10000,
            progress_interval=100,
            csv_filename="sim_data.csv",
            sample_material="G4_GRAPHITE",
            scintillator="EJ200",
            sample_thickness=2,
            csv_batch_size=1000,
        )

    @classmethod
    def opticalphoton_point(cls) -> 'Config':
        """Point source optical photon configuration."""
        return cls(
            particle="opticalphoton",
            energy=3,
            energy_unit="eV",
            position_z=20.,
            position_unit="mm",
            halfx=0.0001,
            halfy=0.0001,
            shape_unit="um",
            num_events=10000,
            max_theta=180,
            min_theta=177,
            progress_interval=1000,
            csv_filename="sim_data.csv",
            sample_material="G4_Galactic",
            scintillator="EJ200",
            sample_thickness=2,
            csv_batch_size=1000,
        )

    @classmethod
    def opticalphoton_uniform(cls) -> 'Config':
        """Uniform source optical photon configuration."""
        return cls(
            particle="opticalphoton",
            energy=3,
            energy_unit="eV",
            position_z=20,
            position_unit="mm",
            halfx=60,
            halfy=60,
            max_theta=180,
            min_theta=177,
            shape_unit="mm",
            num_events=100000,
            progress_interval=1000,
            csv_filename="sim_data.csv",
            sample_material="G4_Galactic",
            scintillator="EJ200",
            sample_thickness=2,
            csv_batch_size=1000,
        )

    @classmethod
    def point_ion(cls, ion_z: int = 39, ion_a: int = 88, ion_excitation: float = 0.0) -> 'Config':
        """Point source configuration for a gamma-emitting isotope using radioactive decay."""
        return cls(
            particle="ion",
            ion_z=ion_z,
            ion_a=ion_a,
            ion_excitation=ion_excitation,
            position_z=0,
            position_unit="mm",
            halfx=0.0001,
            halfy=0.0001,
            shape_unit="um",
            angle_type="iso",
            max_theta=180,
            min_theta=0,
            num_events=100000,
            progress_interval=100,
            csv_filename="sim_data.csv",
            sample_material="G4_Galactic",
            scintillator="EJ200",
            sample_thickness=2,
            csv_batch_size=1000,
        )

    @classmethod
    def point_gamma_lines(cls, gamma_lines: List[Tuple[float, float]] = None) -> 'Config':
        """Point source configuration for gamma rays with specified energies and intensities."""
        if gamma_lines is None:
            # Default to Y-88 gamma lines
            gamma_lines = [(0.898, 93), (1.836, 99), (2.734, 0.6)]
        
        # Normalize intensities to sum to 1
        total_intensity = sum(intensity for energy, intensity in gamma_lines)
        normalized_lines = [(energy, intensity / total_intensity) for energy, intensity in gamma_lines]
        
        return cls(
            particle="gamma",
            energy_type="Hist",
            energy_histogram=normalized_lines,
            energy_unit="MeV",
            position_z=0,
            position_unit="mm",
            halfx=0.0001,
            halfy=0.0001,
            shape_unit="um",
            angle_type="iso",
            max_theta=180,
            min_theta=0,
            num_events=100000,
            progress_interval=100,
            csv_filename="sim_data.csv",
            sample_material="G4_Galactic",
            scintillator="EJ200",
            sample_thickness=2,
            csv_batch_size=1000,
        )

    @classmethod
    def uniform_epithermal_neutrons(cls, pulse_width: float = 1000.0) -> 'Config':
        """Neutron configuration with uniform epithermal energy distribution.
        Args:
            pulse_width (float): Width of the neutron pulse in ns. Default is 1000 ns.
        """
        return cls(
            particle="neutron",
            energy_type="Lin",
            energy_min=0.5,
            energy_max=100.0,
            energy_gradient=0.0,
            energy_intercept=1.0,
            energy_unit="eV",
            position_z=-900,
            position_unit="cm",
            halfx=60,
            halfy=60,
            flux=1e4,  # Neutron flux in n/cm²/s
            freq=20,  # Pulse frequency in Hz (20 Hz)
            tmin=0.0,
            tmax=pulse_width,
            shape_unit="mm",
            num_events=10000,
            progress_interval=100,
            csv_filename="sim_data.csv",
            sample_material="G4_TUNGSTEN",
            scintillator="GS20",
            sample_thickness=0.05,
            scintillator_thickness=1,
            csv_batch_size=1000,
        )

    @classmethod
    def pulsed_neutron_source(cls, pulse_width: float = 1000.0) -> 'Config':
        """Example configuration for a pulsed neutron source with time spread."""
        return cls(
            particle="neutron",
            energy=10.0,
            energy_unit="MeV",
            position_z=-1059,
            position_unit="cm",
            halfx=60,
            halfy=60,
            shape_unit="mm",
            tmin=0.0,
            tmax=pulse_width,
            time_unit="ns",
            num_events=100000,
            progress_interval=100,
            csv_filename="sim_data.csv",
            sample_material="G4_GRAPHITE",
            scintillator="EJ200",
            sample_thickness=2,
            csv_batch_size=1000,
        )

    @classmethod
    def neutrons_tof(cls, energy_min: Optional[float] = 1.0, energy_max: Optional[float] = 10.) -> 'Config':
        """Neutron configuration for time-of-flight with pulsed structure and optional energy range."""
        if energy_min is not None and energy_max is not None:
            # Use linear energy distribution
            return cls(
                particle="neutron",
                energy_type="Lin",
                energy_min=energy_min,
                energy_max=energy_max,
                energy_gradient=0.0,
                energy_intercept=1.0,
                energy_unit="MeV",
                position_z=-1085,  # Flight path of 10.85 m
                position_unit="cm",
                halfx=60,  # 12 cm FOV = 120 mm, halfx = 60 mm
                halfy=60,  # 12 cm FOV = 120 mm, halfy = 60 mm
                shape_unit="mm",
                flux=1e4,  # Neutron flux in n/cm²/s
                freq=200000,  # Pulse frequency in Hz (200 kHz)
                num_events=10000,
                progress_interval=100,
                csv_filename="sim_data.csv",
                sample_material="G4_GRAPHITE",
                scintillator="EJ200",
                sample_thickness=7.5,  # 7.5 cm = 75 mm
                sample_width=12.0,  # 12 cm = 120 mm
                scintillator_thickness=2,  # 2 cm = 20 mm
                csv_batch_size=10000,
            )
        else:
            # Default to monoenergetic 10 MeV
            return cls(
                particle="neutron",
                energy=10.0,
                energy_type="Mono",
                energy_unit="MeV",
                position_z=-1085,  # Flight path of 10.85 m
                position_unit="cm",
                halfx=60,  # 12 cm FOV = 120 mm, halfx = 60 mm
                halfy=60,  # 12 cm FOV = 120 mm, halfy = 60 mm
                shape_unit="mm",
                flux=1e4,  # Neutron flux in n/cm²/s
                freq=200000,  # Pulse frequency in Hz (200 kHz)
                num_events=10000,
                progress_interval=100,
                csv_filename="sim_data.csv",
                sample_material="G4_GRAPHITE",
                scintillator="EJ200",
                sample_thickness=7.5,  # 7.5 cm = 75 mm
                scintillator_thickness=2,  # 2 cm = 20 mm
                csv_batch_size=10000,
            )

    def write(self, output_file: str) -> str:
        """
        Write configuration to a Geant4 macro file.
        """
        macro_content = ""
        if self.particle == "ion" and self.ion_z is not None and self.ion_a is not None:
            macro_content += f"""
/gps/particle ion
/gps/ion {self.ion_z} {self.ion_a} 0 {self.ion_excitation}
/grdm/nucleusLimits {self.ion_a} {self.ion_a} {self.ion_z} {self.ion_z}
/grdm/applyICM true
/grdm/applyARM true
"""
        else:
            macro_content += f"""
/gps/particle {self.particle}
"""
            if self.energy_type == "Mono":
                macro_content += f"/gps/energy {self.energy} {self.energy_unit}\n"
            elif self.energy_type == "Lin":
                macro_content += f"""
/gps/ene/type Lin
/gps/ene/min {self.energy_min} {self.energy_unit}
/gps/ene/max {self.energy_max} {self.energy_unit}
/gps/ene/gradient {self.energy_gradient}
/gps/ene/intercept {self.energy_intercept}
"""
            elif self.energy_type == "Hist" and self.energy_histogram is not None:
                macro_content += f"""
/gps/ene/type User
/gps/hist/type energy
"""
                for energy, intensity in self.energy_histogram:
                    macro_content += f"/gps/hist/point {energy} {intensity}\n"

        # Add time spread configuration
        if self.tmax > self.tmin:
            macro_content += f"""
/lumacam/tmin {self.tmin} {self.time_unit}
/lumacam/tmax {self.tmax} {self.time_unit}
"""

        # Add pulse configuration
        if self.flux is not None and self.freq is not None:
            macro_content += f"""
/lumacam/flux {self.flux}
/lumacam/freq {self.freq}
"""

        macro_content += f"""
/gps/position {self.position_x} {self.position_y} {self.position_z} {self.position_unit}
/gps/direction {self.direction_x} {self.direction_y} {self.direction_z}
/gps/pos/shape {self.shape}
/gps/pos/halfx {self.halfx} {self.shape_unit}
/gps/pos/halfy {self.halfy} {self.shape_unit}
/gps/pos/type Plane
/gps/ang/type {self.angle_type}
/gps/ang/maxtheta {self.max_theta} {self.angle_unit}
/gps/ang/mintheta {self.min_theta} {self.angle_unit}
/run/printProgress {self.progress_interval}
/lumacam/scintMaterial {self.scintillator}
/lumacam/sampleThickness {self.sample_thickness} cm
/lumacam/sampleWidth {self.sample_width} cm
/lumacam/scintThickness {self.scintillator_thickness} cm
/lumacam/sampleMaterial {self.sample_material}
/lumacam/batchSize {self.csv_batch_size}
/control/verbose 2
/run/beamOn {self.num_events}
"""
        with open(output_file, 'w') as f:
            f.write(macro_content.strip())
            
        return output_file

    def __str__(self) -> str:
        """Return a human-readable string representation of the configuration."""
        source_info = ""
        if self.particle == "ion" and self.ion_z is not None and self.ion_a is not None:
            source_info = f"Source: Ion (Z={self.ion_z}, A={self.ion_a}, Excitation={self.ion_excitation} keV)\n"
        else:
            energy_info = ""
            if self.energy_type == "Mono":
                energy_info = f"Energy: {self.energy} {self.energy_unit}\n"
            elif self.energy_type == "Lin":
                energy_info = (f"Energy: uniform distribution from {self.energy_min} to {self.energy_max} {self.energy_unit}\n"
                              f"  (gradient: {self.energy_gradient}, intercept: {self.energy_intercept})\n")
            elif self.energy_type == "Hist" and self.energy_histogram is not None:
                energy_info = "Energy: discrete lines\n"
                for energy, intensity in self.energy_histogram:
                    energy_info += f"  {energy} {self.energy_unit} (intensity: {intensity:.3f})\n"
            source_info = f"Particle: {self.particle}\n  {energy_info}"
        
        # Add time spread info
        time_info = ""
        if self.tmax > self.tmin:
            time_info = f"  Time spread: {self.tmin} to {self.tmax} {self.time_unit}\n"
        elif self.tmin != 0.0 or self.tmax != 0.0:
            time_info = f"  Time: {self.tmin} {self.time_unit}\n"
        
        # Add pulse info
        pulse_info = ""
        if self.flux is not None and self.freq is not None:
            pulse_info = f"  Neutron Flux: {self.flux} n/cm²/s\n  Pulse Frequency: {self.freq/1000} kHz\n"
            
        return (
            f"Configuration:\n"
            f"  {source_info}"
            f"{time_info}"
            f"{pulse_info}"
            f"  Position: ({self.position_x}, {self.position_y}, {self.position_z}) {self.position_unit}\n"
            f"  Direction: ({self.direction_x}, {self.direction_y}, {self.direction_z})\n"
            f"  Shape: {self.shape} ({self.halfx}x{self.halfy} {self.shape_unit})\n"
            f"  Angle: {self.angle_type} (max theta: {self.max_theta} {self.angle_unit})\n"
            f"  Sample Material: {self.sample_material}\n"
            f"  Sample Thickness: {self.sample_thickness} cm\n"
            f"  Scintillator: {self.scintillator}\n"
            f"  Scintillator Thickness: {self.scintillator_thickness} cm\n"
            f"  CSV Batch Size: {self.csv_batch_size}\n"
            f"  Progress Interval: {self.progress_interval}\n"
            f"  Events: {self.num_events}\n"
            f"  Output: {self.csv_filename}"
        )

    def __repr__(self) -> str:
        """Return a string representation of the configuration."""
        return str(self)
        
class Simulate:
    """Class to simulate the lumacam executable."""
    def __init__(self, archive: str = "archive/test"):
        """
        Initialize the Simulate object.
        """
        self.archive = Path(archive).absolute()
        self.archive.mkdir(exist_ok=True, parents=True)
        
        self.sim_dir = self.archive / "SimPhotons"
        self.sim_dir.mkdir(exist_ok=True, parents=True)

        with resources.path('G4LumaCam', 'bin') as bin_path:
            self.lumacam_executable = os.path.join(bin_path, "lumacam")

    def _process_output(self, process, output_queue, verbosity):
        """Process the output from the simulation in real-time."""
        event_pattern = re.compile(r'--> Event (\d+) starts\.')
        final_event_pattern = re.compile(r'Simulating Event: (\d+)')
        cleanup_pattern = re.compile(r'Graphics systems deleted\.')
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
                
            line = line.strip()
            if line:
                match = event_pattern.match(line)
                if match:
                    event_num = int(match.group(1))
                    output_queue.put(('progress', event_num))
                    continue

                match = final_event_pattern.search(line)
                if match:
                    event_num = int(match.group(1))
                    output_queue.put(('progress', event_num))
                    continue

                if cleanup_pattern.search(line):
                    output_queue.put(('complete', None))
                
                if verbosity >= VerbosityLevel.DETAILED:
                    output_queue.put(('output', line))
                elif verbosity >= VerbosityLevel.BASIC and ('starts.' in line or 'Run' in line or 'G4Exception' in line):
                    output_queue.put(('output', line))

    def clear_subfolders(self, verbosity: VerbosityLevel = VerbosityLevel.BASIC):
        """Remove all contents of SimPhotons subfolder if it exists.
        This ensures that old simulation data does not interfere with new runs.
        Args:
            verbosity (VerbosityLevel): Level of verbosity for print statements.           
        """
        if self.sim_dir.exists():
            for item in self.sim_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Cleared contents of {self.sim_dir}")

    def run(self, 
            config_or_file: Optional[str | Config] = None, 
            verbosity: VerbosityLevel = VerbosityLevel.BASIC) -> pd.DataFrame:
        """
        Run the lumacam executable with either a Config object or a macro file.
        """
        if not os.path.exists(self.lumacam_executable):
            raise FileNotFoundError(f"lumacam executable not found at {self.lumacam_executable}")

        # Check write permissions
        if not os.access(self.sim_dir, os.W_OK):
            raise PermissionError(f"No write permission in {self.sim_dir}")

        # Clear SimPhotons subfolder
        self.clear_subfolders(verbosity=verbosity)

        temp_macro = None
        macro_file = None
        num_events = None
        progress_interval = None
        csv_filename = "sim_data.csv"
        
        if isinstance(config_or_file, Config):
            temp_macro = self.sim_dir / "macro.mac"
            macro_file = config_or_file.write(str(temp_macro))
            num_events = config_or_file.num_events
            progress_interval = config_or_file.progress_interval
            csv_filename = config_or_file.csv_filename
            shutil.copy(str(temp_macro), str(self.archive / "macro.mac"))
        elif isinstance(config_or_file, str):
            if not os.path.exists(config_or_file):
                raise FileNotFoundError(f"Macro file not found at {config_or_file}")
            macro_file = str(self.sim_dir / "macro.mac")
            shutil.copy(config_or_file, macro_file)
            with open(macro_file, 'r') as f:
                content = f.read()
                match = re.search(r'/run/beamOn\s+(\d+)', content)
                if match:
                    num_events = int(match.group(1))
                match = re.search(r'/run/printProgress\s+(\d+)', content)
                if match:
                    progress_interval = int(match.group(1))
                match = re.search(r'/lumacam/csvFilename\s+(\S+)', content)
                if match:
                    csv_filename = match.group(1)

        original_dir = os.getcwd()
        os.chdir(str(self.archive))

        if verbosity >= VerbosityLevel.DETAILED:
            print(f"Current working directory: {os.getcwd()}")
            print(f"Expected CSV output: {csv_filename}")

        try:
            process = subprocess.Popen(
                [self.lumacam_executable, "macro.mac"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            output_queue = queue.Queue()
            output_thread = threading.Thread(
                target=self._process_output,
                args=(process, output_queue, verbosity)
            )
            output_thread.daemon = True
            output_thread.start()

            if num_events is not None:
                pbar = tqdm(total=num_events, desc="Simulating", unit="events")
            else:
                pbar = None

            last_event = 0
            run_completed = False
            last_update = 0

            while process.poll() is None or not output_queue.empty():
                try:
                    msg_type, content = output_queue.get(timeout=0.1)
                    if msg_type == 'progress':
                        current_event = content
                        if pbar is not None and progress_interval:
                            if current_event - last_update >= progress_interval:
                                pbar.n = min(current_event, num_events)
                                pbar.refresh()
                                last_update = current_event
                        last_event = current_event
                    elif msg_type == 'complete':
                        run_completed = True
                        if pbar is not None:
                            pbar.n = num_events
                            pbar.refresh()
                    elif msg_type == 'output':
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(content)
                except queue.Empty:
                    continue

            if pbar is not None:
                if run_completed or last_event >= num_events - 1:
                    pbar.n = num_events
                pbar.refresh()
                pbar.close()

            stderr = process.stderr.read()
            if process.returncode != 0 and verbosity > 1:
                print(f"lumacam execution failed with error:\n{stderr}")

            base_name = csv_filename.rsplit('.', 1)[0]
            extension = csv_filename.rsplit('.', 1)[1] if '.' in csv_filename else "csv"
            
            dfs = []
            # Use glob to find all CSV files in SimPhotons directory
            csv_pattern = os.path.join(str(self.sim_dir), f"{base_name}*.{extension}")
            csv_files = sorted(glob.glob(csv_pattern))
            
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Looking for CSV files with pattern: {csv_pattern}")
                print(f"Found CSV files: {csv_files}")
            
            for csv_file in csv_files:
                csv_path = Path(csv_file)
                if csv_path.exists():
                    file_size = csv_path.stat().st_size
                    if verbosity >= VerbosityLevel.DETAILED:
                        print(f"Processing CSV file: {csv_path} (size: {file_size} bytes)")
                    
                    if file_size == 0:
                        csv_path.unlink()
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"Removed empty CSV file: {csv_path}")
                        continue
                    
                    try:
                        df = pd.read_csv(csv_path)
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"CSV file {csv_path}: {df.shape[0]} rows, {df.shape[1]} columns")
                        
                        if df.shape[0] == 0:
                            csv_path.unlink()
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"Removed header-only CSV file: {csv_path}")
                        else:
                            dfs.append(df)
                            if verbosity >= VerbosityLevel.DETAILED:
                                print(f"Added {df.shape[0]} rows from {csv_path}")
                    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"Removing malformed/empty CSV file {csv_path}: {e}")
                        csv_path.unlink()
                    except Exception as e:
                        if verbosity >= VerbosityLevel.DETAILED:
                            print(f"Error reading CSV file {csv_path}: {e}")
                        csv_path.unlink()
                else:
                    print(f"CSV file does not exist: {csv_path}")
            
            if not dfs:
                print(f"No valid (non-empty) CSV files found in {self.sim_dir}. Check EventProcessor output logic or simulation configuration.")
                return pd.DataFrame()  # Return empty DataFrame instead of raising an error
            
            combined_df = pd.concat(dfs, ignore_index=True)
            if verbosity >= VerbosityLevel.DETAILED:
                print(f"Combined DataFrame: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
            return combined_df

        finally:
            os.chdir(original_dir)