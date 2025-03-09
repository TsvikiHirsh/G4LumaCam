import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.command.install import install

class BuildGeant4Simulation(build_py):
    """Custom build step to compile the Geant4 simulation with SSLG4 and OPSim."""
    def run(self):
        build_dir = os.path.join(os.getcwd(), "build")
        os.makedirs(build_dir, exist_ok=True)

        # Define SSLG4 and OPSim paths (relative to src/)
        sslg4_base_dir = os.path.abspath(os.path.join(os.getcwd(), "src", "SSLG4", "sslg4"))
        sslg4_include_dir = os.path.join(sslg4_base_dir, "include")
        sslg4_source_dir = os.path.join(sslg4_base_dir, "src")

        opsim_base_dir = os.path.abspath(os.path.join(os.getcwd(), "src", "OPSim", "OPSim"))  # Adjusted to OPSim/OPSim/
        opsim_include_dir = os.path.join(opsim_base_dir, "include")
        opsim_source_dir = os.path.join(opsim_base_dir, "src")

        # Check if directories exist
        for dir_path, name in [
            (sslg4_include_dir, "SSLG4 include"),
            (opsim_include_dir, "OPSim include")
        ]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"{name} directory not found at {dir_path}. Please ensure submodules are initialized (git submodule update --init --recursive).")

        # Configure CMake with SSLG4 and OPSim paths
        cmake_args = [
            "cmake",
            "../src/G4LumaCam",
            f"-DSSLG4_INCLUDE_DIR={sslg4_include_dir}",
            f"-DSSLG4_SOURCE_DIR={sslg4_source_dir}",
            f"-DOPSIM_INCLUDE_DIR={opsim_include_dir}",
            f"-DOPSIM_SOURCE_DIR={opsim_source_dir}"
        ]
        subprocess.check_call(cmake_args, cwd=build_dir)

        # Build the project
        subprocess.check_call(["cmake", "--build", "."], cwd=build_dir)

        lumacam_executable = os.path.join(build_dir, "lib", "lumacam")  # Matches CMake output dir
        print("Build directory:", build_dir)
        print("Looking for lumacam executable at:", lumacam_executable)

        if not os.path.exists(lumacam_executable):
            raise FileNotFoundError("lumacam executable not found in build directory.")

        bin_dir = os.path.join(self.build_lib, "G4LumaCam", "bin")
        os.makedirs(bin_dir, exist_ok=True)
        subprocess.check_call(["cp", lumacam_executable, bin_dir])

        executable_path = os.path.join(bin_dir, "lumacam")
        os.chmod(executable_path, 0o755)

        super().run()

class CustomInstall(install):
    """Custom install command to ensure the executable is properly installed and configure EMPIR path."""
    def run(self):
        install.run(self)
        
        empir_path = os.environ.get('EMPIR_PATH')
        if empir_path:
            try:
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
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        'G4LumaCam': ['bin/*', 'config/*'],
    },
    install_requires=[
        "rayoptics",
        "tqdm",
        "pandas",
        "scikit-learn"
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
    }
)