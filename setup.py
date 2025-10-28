import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.install import install

class BuildGeant4Simulation(build_py):
    """Custom build step to compile the Geant4 simulation."""
    def run(self):
        # Create build directory
        build_dir = os.path.join(os.getcwd(), "build")
        os.makedirs(build_dir, exist_ok=True)

        # Run CMake and build
        subprocess.check_call(["cmake", "../src/G4LumaCam"], cwd=build_dir)
        subprocess.check_call(["cmake", "--build", "."], cwd=build_dir)

        # Path to the lumacam executable
        lumacam_executable = os.path.join(build_dir, "lumacam")

        # Debug output
        print("Build directory:", build_dir)
        print("Looking for lumacam executable at:", lumacam_executable)

        # Check if the executable exists
        if not os.path.exists(lumacam_executable):
            raise FileNotFoundError("lumacam executable not found in build directory.")

        # Create the bin directory in the package
        bin_dir = os.path.join(self.build_lib, "G4LumaCam", "bin")
        os.makedirs(bin_dir, exist_ok=True)

        # Copy the executable to the package directory
        subprocess.check_call(["cp", lumacam_executable, bin_dir])

        # Make the executable executable
        executable_path = os.path.join(bin_dir, "lumacam")
        os.chmod(executable_path, 0o755)

        # Continue with normal build process
        super().run()

class CustomInstall(install):
    """Custom install command to ensure the executable is properly installed and configure EMPIR path."""
    def run(self):
        install.run(self)
        
        # Create or update config file with EMPIR path if specified
        empir_path = os.environ.get('EMPIR_PATH')
        if empir_path:
            try:
                # Determine the installation directory
                site_packages_dir = self.install_lib
                config_dir = os.path.join(site_packages_dir, 'G4LumaCam', 'config')
                os.makedirs(config_dir, exist_ok=True)
                
                config_file = os.path.join(config_dir, 'paths.py')
                with open(config_file, 'w') as f:
                    f.write(f"EMPIR_PATH = '{empir_path}'\n")
                
                print(f"EMPIR_PATH configured as {empir_path}")
            except Exception as e:
                print(f"Warning: Could not configure EMPIR_PATH: {e}")
        else:
            print("Note: EMPIR_PATH environment variable not set. Using default './empir' path.")
            print("To set EMPIR_PATH, run: export EMPIR_PATH=/path/to/empir/executables before installation")

setup(
    name="G4LumaCam",
    version="0.3.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        'G4LumaCam': ['bin/*', 'config/*'],
        'lumacam': ['data/*.zmx'],  # Include .zmx files in lumacam/data
    },
    install_requires=[
        "rayoptics",
        "tqdm",
        "pandas",
        "scikit-learn",
        "lmfit",
        "matplotlib",
        "tifffile",
        "roifile"
    ],
    dependency_links=[
        "git+https://github.com/TsvikiHirsh/neutron_event_analyzer.git#egg=neutron_event_analyzer",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "lumacam=G4LumaCam.run_lumacam:main",
        ]
    },
    cmdclass={
        "build_py": BuildGeant4Simulation,
        "install": CustomInstall,
    },
)