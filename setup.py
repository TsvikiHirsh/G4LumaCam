import os
import subprocess
import sys
import shutil
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

        opsim_base_dir = os.path.abspath(os.path.join(os.getcwd(), "src", "OPSim", "OPSim"))
        opsim_include_dir = os.path.join(opsim_base_dir, "include")
        opsim_source_dir = os.path.join(opsim_base_dir, "src")

        # Check if directories exist
        for dir_path, name in [
            (sslg4_include_dir, "SSLG4 include"),
            (opsim_include_dir, "OPSim include")
        ]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"{name} directory not found at {dir_path}. Please ensure submodules are initialized (git submodule update --init --recursive).")

        # Configure CMake with SSLG4 and OPSim paths, set output to build/
        cmake_args = [
            "cmake",
            "../src/G4LumaCam",
            f"-DSSLG4_INCLUDE_DIR={sslg4_include_dir}",
            f"-DSSLG4_SOURCE_DIR={sslg4_source_dir}",
            f"-DOPSIM_INCLUDE_DIR={opsim_include_dir}",
            f"-DOPSIM_SOURCE_DIR={opsim_source_dir}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={build_dir}"  # Output to build/
        ]
        try:
            subprocess.check_call(cmake_args, cwd=build_dir)
        except subprocess.CalledProcessError as e:
            print("CMake configuration failed. Output:")
            subprocess.run(cmake_args, cwd=build_dir, capture_output=False)
            raise

        # Build the project
        try:
            subprocess.check_call(["cmake", "--build", ".", "--verbose"], cwd=build_dir)
        except subprocess.CalledProcessError as e:
            print("CMake build failed.")
            raise

        # Check for the executable at build/lumacam
        expected_path = os.path.join(build_dir, "lumacam")
        print("Build directory:", build_dir)
        print("Expected lumacam path:", expected_path)
        print("Directory contents (build/):", os.listdir(build_dir))
        if os.path.exists(expected_path):
            lumacam_executable = expected_path
            print("Found lumacam at expected path.")
        else:
            raise FileNotFoundError("lumacam executable not found at expected path: " + expected_path)

        # Set up the target directory and copy the executable and macros
        bin_dir = os.path.join(self.build_lib, "G4LumaCam", "bin")
        macros_dir = os.path.join(self.build_lib, "G4LumaCam", "macros")
        data_dir = os.path.join(self.build_lib, "G4LumaCam", "data")
        print("Target bin directory:", bin_dir)
        print("Target macros directory:", macros_dir)
        print("Target data directory:", data_dir)
        os.makedirs(bin_dir, exist_ok=True)
        os.makedirs(macros_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        destination_path = os.path.join(bin_dir, "lumacam")
        print("Copying lumacam to:", destination_path)
        subprocess.check_call(["cp", "-f", lumacam_executable, destination_path])
        os.chmod(destination_path, 0o755)

        # Copy macros from build/macros to package
        build_macros_dir = os.path.join(build_dir, "macros")
        if os.path.exists(build_macros_dir):
            print(f"Copying macros from {build_macros_dir} to {macros_dir}")
            subprocess.check_call(["cp", "-r", f"{build_macros_dir}/.", macros_dir])
        else:
            print("Warning: No macros found in build/macros to copy.")

        # Copy data from build/data to package
        build_data_dir = os.path.join(build_dir, "data")
        if os.path.exists(build_data_dir):
            print(f"Copying data from {build_data_dir} to {data_dir}")
            subprocess.check_call(["cp", "-r", f"{build_data_dir}/.", data_dir])
        else:
            print("Warning: No data found in build/data to copy.")

        # Clean up any stray lumacam file in build/lib/
        stray_path = os.path.join(self.build_lib, "lumacam")
        if os.path.exists(stray_path):
            print(f"Removing stray file at {stray_path} to avoid conflict.")
            if os.path.isfile(stray_path):
                os.remove(stray_path)
            elif os.path.isdir(stray_path):
                shutil.rmtree(stray_path)

        # Run the parent build_py logic
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
        'G4LumaCam': ['bin/lumacam', 'macros/*/*.mac', 'data/*/*/*', 'config/*'],
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